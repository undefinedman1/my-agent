
from __future__ import annotations

import inspect
import logging
import math
import textwrap
import time
from datetime import datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any, Callable

import requests
import streamlit as st

logger = logging.getLogger(__name__)

# ==========================================================================
# 1. 常量配置
# ==========================================================================
REQUEST_TIMEOUT = (3.0, 5.0)   # (连接超时, 读取超时) 秒
CACHE_TTL_SECONDS = 10         # 行情缓存有效期
CACHE_MAX_ENTRIES = 16         # 缓存条目上限，防止频繁刷新导致无限增长
MIN_REFRESH_INTERVAL = 1.0     # 两次手动刷新的最小间隔（秒），防抖
AUTO_REFRESH_SECONDS = 30      # 自动刷新间隔（run_every，需 Streamlit >= 1.37）
STALE_WARNING_SECONDS = 600    # 陈旧数据告警阈值（10 分钟）

COINGECKO_MARKETS_URL = "https://api.coingecko.com/api/v3/coins/markets"
BINANCE_TICKER_URL = "https://api.binance.com/api/v3/ticker/24hr"

HTTP_HEADERS = {
    "User-Agent": "btc-price-dashboard/1.1 (Streamlit)",
    "Accept": "application/json",
}

CUSTOM_CSS = textwrap.dedent(
    """
    <style>
    .btc-card{
        background:linear-gradient(135deg,#1e293b 0%,#0f172a 100%);
        border:1px solid rgba(148,163,184,.18);
        border-radius:18px;
        padding:30px 34px;
        text-align:center;
        box-shadow:0 12px 30px rgba(15,23,42,.28);
        margin-bottom:14px;
    }
    .btc-card-dim{opacity:.65;}
    .btc-card-label{
        color:#94a3b8;font-size:15px;letter-spacing:.05em;margin-bottom:10px;
    }
    .btc-card-price{
        color:#f8fafc;font-weight:700;line-height:1.1;letter-spacing:-1.2px;
        font-size:clamp(32px,9vw,54px);font-variant-numeric:tabular-nums;
    }
    .btc-card-change{
        margin-top:12px;font-size:19px;font-weight:600;
        font-variant-numeric:tabular-nums;
    }
    .btc-card-pct{margin-left:6px;}
    .btc-card-period{
        color:#94a3b8;font-size:14px;font-weight:500;margin-left:8px;
    }
    .up{color:#22c55e;}
    .down{color:#ef4444;}
    .flat{color:#94a3b8;}
    </style>
    """
).strip()


# ==========================================================================
# 2. 数据获取层
# ==========================================================================
class PriceFetchError(Exception):
    """行情数据获取 / 解析失败。调用方可据此切换到备用数据源。"""


def _http_get_json(url: str, params: dict[str, Any]) -> Any:
    """发起 GET 请求并解析 JSON，所有网络异常统一转换为 PriceFetchError。"""
    try:
        response = requests.get(
            url, params=params, timeout=REQUEST_TIMEOUT, headers=HTTP_HEADERS
        )
    except requests.exceptions.Timeout as exc:
        raise PriceFetchError("请求超时") from exc
    except requests.exceptions.SSLError as exc:
        raise PriceFetchError("SSL 证书校验失败") from exc
    except requests.exceptions.ConnectionError as exc:
        raise PriceFetchError("网络连接失败") from exc
    except requests.exceptions.RequestException as exc:
        raise PriceFetchError(f"请求异常：{exc}") from exc

    if response.status_code == 429:
        raise PriceFetchError("请求被接口限流 (HTTP 429)")
    if response.status_code != 200:
        raise PriceFetchError(f"接口返回异常状态码 {response.status_code}")

    try:
        return response.json()
    except ValueError as exc:
        raise PriceFetchError("接口返回内容不是合法 JSON") from exc


def _to_float(value: Any, field_name: str) -> float:
    """安全地把接口字段转成有限的 float，失败时抛出可读错误。

    注意：显式拒绝 ``bool``，因为 ``float(True) == 1.0`` 会把接口异常静默放行。
    """
    if isinstance(value, bool):
        raise PriceFetchError(f"字段 {field_name} 类型异常（{value!r}）")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise PriceFetchError(f"字段 {field_name} 缺失或格式异常（{value!r}）") from exc
    if not math.isfinite(number):
        raise PriceFetchError(f"字段 {field_name} 不是有效数字")
    return number


def fetch_from_coingecko() -> dict[str, float]:
    """主数据源：CoinGecko。一次性返回价格、24h 涨跌额与涨跌幅。

    接口偶尔只返回其中一个涨跌字段，此时互相推导；但**任何推导失败都不会
    拖垮整个数据源**（避免 float(None) 抛错导致级联降级）。
    """
    payload = _http_get_json(
        COINGECKO_MARKETS_URL,
        {
            "vs_currency": "usd",
            "ids": "bitcoin",
            "price_change_percentage": "24h",
        },
    )
    if not isinstance(payload, list) or not payload:
        raise PriceFetchError("返回数据为空")
    item = payload[0]
    if not isinstance(item, dict):
        raise PriceFetchError("返回结构异常")

    price = _to_float(item.get("current_price"), "current_price")

    change_val: float | None = None
    pct_val: float | None = None
    if item.get("price_change_24h") is not None:
        change_val = _to_float(item["price_change_24h"], "price_change_24h")
    if item.get("price_change_percentage_24h") is not None:
        pct_val = _to_float(
            item["price_change_percentage_24h"], "price_change_percentage_24h"
        )

    # 互相推导（带分母/除零保护，极端值也不会抛错）
    if change_val is None and pct_val is not None:
        denom = 1 + pct_val / 100
        if denom > 0.01:
            change_val = price - price / denom
    if pct_val is None and change_val is not None:
        prev_price = price - change_val
        if prev_price:
            pct_val = (price - prev_price) / prev_price * 100

    if change_val is None and pct_val is None:
        raise PriceFetchError("24h 涨跌数据全部缺失")

    return {
        "price": price,
        "change_24h": change_val if change_val is not None else 0.0,
        "change_pct_24h": pct_val if pct_val is not None else 0.0,
    }


def fetch_from_binance() -> dict[str, float]:
    """备用数据源：Binance。字段为字符串，需显式转换。"""
    payload = _http_get_json(BINANCE_TICKER_URL, {"symbol": "BTCUSDT"})
    if not isinstance(payload, dict):
        raise PriceFetchError("返回结构异常")
    return {
        "price": _to_float(payload.get("lastPrice"), "lastPrice"),
        "change_24h": _to_float(payload.get("priceChange"), "priceChange"),
        "change_pct_24h": _to_float(
            payload.get("priceChangePercent"), "priceChangePercent"
        ),
    }


# 数据源按优先级排列，前面的失败后自动降级到后面的
DATA_SOURCES: tuple[tuple[str, Callable[[], dict[str, float]]], ...] = (
    ("CoinGecko", fetch_from_coingecko),
    ("Binance", fetch_from_binance),
)


@st.cache_data(ttl=CACHE_TTL_SECONDS, max_entries=CACHE_MAX_ENTRIES, show_spinner=False)
def get_btc_quote(
    refresh_token: float,
) -> tuple[dict[str, Any] | None, str | None]:
    """按优先级尝试各数据源，返回 ``(行情数据, 错误信息)``，二者有且仅有一个为 None。

    ``refresh_token`` 仅参与缓存 key：用户点击「刷新」时写入当前时间戳，
    使缓存立即失效，从而保证「点刷新必拉到新数据」。

    注意：本函数**不抛异常**，失败信息也随返回值一起缓存，
    避免接口宕机时每次页面交互都重复等待 8 秒超时。
    """
    _ = refresh_token  # 仅用作缓存 key，不参与业务逻辑

    failures: list[str] = []
    for source_name, fetcher in DATA_SOURCES:
        try:
            quote = fetcher()
        except PriceFetchError as exc:
            logger.warning("数据源 %s 失败: %s", source_name, exc)
            failures.append(f"{source_name}：{exc}")
        except Exception as exc:  # noqa: BLE001 - 兜底，绝不让意外异常冒到 UI
            logger.error(
                "数据源 %s 出现未知错误", source_name, exc_info=True
            )
            failures.append(f"{source_name}：未知错误 {type(exc).__name__}: {exc}")
        else:
            quote["source"] = source_name
            quote["fetched_at"] = datetime.now()
            return quote, None

    return None, "；".join(failures) or "未知错误"


# ==========================================================================
# 3. 格式化工具
# ==========================================================================
def _normalize(value: float, digits: int = 2) -> float:
    """四舍五入（HALF_UP）并消除 ``-0.0`` 这类显示问题。

    金融展示用 Decimal 量化，避免 Python ``round()`` 的银行家舍入带来的反直觉结果。
    """
    try:
        quantum = Decimal(1).scaleb(-digits)  # 如 digits=2 -> Decimal('0.01')
        quantized = Decimal(str(float(value))).quantize(quantum, rounding=ROUND_HALF_UP)
    except (InvalidOperation, ValueError, TypeError):
        return 0.0
    result = float(quantized)
    return 0.0 if result == 0 else result


def _format_price(value: float) -> str:
    return f"${_normalize(value):,.2f}"


def _format_signed_usd(value: float) -> str:
    value = _normalize(value)
    sign = "+" if value > 0 else ("-" if value < 0 else "")
    return f"{sign}${abs(value):,.2f}"


def _format_signed_pct(value: float) -> str:
    value = _normalize(value)
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.2f}%"


def _format_age(delta: timedelta) -> str:
    """把时间差格式化为「X 秒前 / X 分钟前 / X 小时前 / X 天前」。"""
    seconds = max(0, int(delta.total_seconds()))
    if seconds < 60:
        return f"{seconds} 秒前"
    if seconds < 3600:
        return f"{seconds // 60} 分钟前"
    if seconds < 86400:
        return f"{seconds // 3600} 小时前"
    return f"{seconds // 86400} 天前"


# ==========================================================================
# 4. UI 层
# ==========================================================================
def _inject_css() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def _init_state() -> None:
    """初始化会话状态（幂等）。"""
    st.session_state.setdefault("refresh_token", 0.0)
    st.session_state.setdefault("last_refresh_ts", 0.0)
    st.session_state.setdefault("last_good_quote", None)


def _render_quote_card(quote: dict[str, Any], dimmed: bool = False) -> None:
    """渲染大字号价格卡片。"""
    price = _normalize(quote["price"])
    change = _normalize(quote["change_24h"])
    change_pct = _normalize(quote["change_pct_24h"])

    if change > 0:
        trend_class, arrow = "up", "▲"
    elif change < 0:
        trend_class, arrow = "down", "▼"
    else:
        trend_class, arrow = "flat", "—"

    html = (
        f'<div class="btc-card{" btc-card-dim" if dimmed else ""}">'
        '<div class="btc-card-label">比特币 BTC · 当前价格 (USD)</div>'
        f'<div class="btc-card-price">{_format_price(price)}</div>'
        f'<div class="btc-card-change {trend_class}">'
        f"{arrow} {_format_signed_usd(change)}"
        f'<span class="btc-card-pct">({_format_signed_pct(change_pct)})</span>'
        '<span class="btc-card-period">· 近 24 小时</span>'
        "</div>"
        "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def _render_success(quote: dict[str, Any]) -> None:
    _render_quote_card(quote)
    st.caption(
        f"数据源：**{quote['source']}** · "
        f"最后更新：{quote['fetched_at']:%Y-%m-%d %H:%M:%S}（本地时间）"
    )


def _render_stale(stale: dict[str, Any], error: str | None) -> None:
    """接口全失败但有历史数据：降级展示，并明确告知陈旧程度。"""
    age = datetime.now() - stale["fetched_at"]
    age_str = _format_age(age)

    if age.total_seconds() > STALE_WARNING_SECONDS:
        st.warning(
            f"⚠️ 实时行情获取失败，当前展示的是 **{age_str}** 的旧数据。"
            f"数据已超过 {STALE_WARNING_SECONDS // 60} 分钟，可能严重过期，请谨慎参考。"
        )
    else:
        st.warning(f"⚠️ 实时行情获取失败，当前展示的是 **{age_str}** 的最后一次成功数据。")

    _render_quote_card(stale, dimmed=True)
    st.caption(
        f"数据源：{stale['source']} · "
        f"数据时间：{stale['fetched_at']:%Y-%m-%d %H:%M:%S}（{age_str}）"
    )
    if error:
        with st.expander("查看错误详情"):
            st.code(error)


def _render_empty_error(error: str | None) -> None:
    """接口全失败且无任何历史数据：友好提示 + 折叠技术细节。"""
    st.error(
        "😥 **暂时无法获取比特币价格** —— 请检查网络连接后点击上方「🔄 刷新价格」重试。"
    )
    if error:
        with st.expander("查看错误详情"):
            st.code(error)


def _handle_manual_refresh() -> None:
    """处理手动刷新：防抖 + 更新缓存 key。"""
    now = time.time()
    last = float(st.session_state.get("last_refresh_ts", 0.0))
    if now - last < MIN_REFRESH_INTERVAL:
        st.warning("⏳ 刷新过于频繁，请稍后再试。")
        return
    st.session_state["last_refresh_ts"] = now
    st.session_state["refresh_token"] = now  # 缓存 key 变化 => 强制拉新数据


def _detect_auto_refresh_support() -> bool:
    """探测当前 Streamlit 是否支持 ``st.fragment(run_every=...)``（需 >= 1.37）。"""
    if not hasattr(st, "fragment"):
        return False
    try:
        return "run_every" in inspect.signature(st.fragment).parameters
    except (TypeError, ValueError):
        return False


_SUPPORTS_AUTO_REFRESH = _detect_auto_refresh_support()


def _render_controls() -> None:
    """刷新按钮 + 自动刷新开关（开关仅在运行环境支持时展示）。"""
    col_button, col_auto = st.columns([1, 1])
    with col_button:
        if st.button(
            "🔄 刷新价格",
            type="primary",
            use_container_width=True,
            help="立即拉取最新行情（绕过本地缓存）",
        ):
            _handle_manual_refresh()
    with col_auto:
        if _SUPPORTS_AUTO_REFRESH:
            st.checkbox(
                f"自动刷新（每 {AUTO_REFRESH_SECONDS} 秒）",
                key="auto_refresh",
                value=False,
                help="开启后页面会定时自动更新行情",
            )


def _render_dashboard() -> None:
    """取数 + 分派到成功/陈旧/空数据三个渲染分支。"""
    _init_state()

    with st.spinner("正在获取比特币最新行情…"):
        quote, error = get_btc_quote(st.session_state["refresh_token"])

    if quote is not None:
        st.session_state["last_good_quote"] = quote
        _render_success(quote)
        return

    stale = st.session_state.get("last_good_quote")
    if stale is not None:
        _render_stale(stale, error)
    else:
        _render_empty_error(error)


# ---------------------------------------------------------------------------
# 自动刷新 fragment：模块级固定 run_every，避免每次 rerun 动态创建导致定时器叠加
# ---------------------------------------------------------------------------
if _SUPPORTS_AUTO_REFRESH:

    @st.fragment(run_every=AUTO_REFRESH_SECONDS)
    def _auto_refresh_fragment() -> None:
        # 开关关闭时 fragment 空转（不取数、不渲染），由主流程走普通渲染分支
        if st.session_state.get("auto_refresh", False):
            _render_dashboard()

else:  # pragma: no cover - 低版本兜底，仅直出一次

    def _auto_refresh_fragment() -> None:
        _render_dashboard()


# ==========================================================================
# 5. 入口
# ==========================================================================
def main() -> None:
    st.set_page_config(
        page_title="比特币价格看板",
        page_icon="📈",
        layout="centered",
        initial_sidebar_state="collapsed",
    )
    _inject_css()
    _init_state()

    st.title("₿ 比特币价格看板")
    st.caption("实时行情 · 数据来自 CoinGecko / Binance 公共接口")

    _render_controls()
    st.divider()

    if _SUPPORTS_AUTO_REFRESH and st.session_state.get("auto_refresh", False):
        _auto_refresh_fragment()
    else:
        _render_dashboard()

    st.divider()
    st.caption(
        "⚠️ 行情数据仅供参考，不构成任何投资建议。"
        "公共接口存在限流，刷新请适度。"
    )


if __name__ == "__main__":
    main()
