# -*- coding: utf-8 -*-
"""US morning stock scanner with one concise Telegram report."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from zoneinfo import ZoneInfo

import pandas as pd

logger = logging.getLogger(__name__)

MAX_GEMINI_STOCKS = 10
TELEGRAM_TARGET_CHARS = 5800
DEFAULT_TOP_N = 10
DEFAULT_PERIOD = "6mo"
QUALITY_LIMITS = {
    "best": 1,
    "second": 2,
    "watch": 5,
    "avoid": 3,
}
DEFAULT_UNIVERSE = (
    "AAPL,MSFT,NVDA,AMZN,META,GOOGL,GOOG,AVGO,TSLA,BRK-B,JPM,LLY,V,UNH,XOM,MA,COST,JNJ,HD,PG,"
    "NFLX,ABBV,BAC,KO,PLTR,PM,CRM,ORCL,CSCO,WMT,AMD,CVX,GE,ABT,IBM,MRK,LIN,MCD,INTU,DIS,NOW,"
    "AXP,GS,TMO,UBER,PEP,ADBE,QCOM,AMGN,CAT,TXN,ISRG,SPGI,RTX,NEE,PFE,LOW,AMAT,BLK,SYK,BA,"
    "HON,DE,VRTX,MDT,ETN,ADI,CB,MMC,LMT,UPS,REGN,ADP,BSX,KLAC,CI,MDLZ,SO,MO,DUK,ICE,SHW,"
    "PANW,EQIX,CDNS,SNPS,WM,APH,TT,PH,MU,ANET,ELV,CMG,MS,INTC,C,COF,NKE"
)

MARKET_INDICES = {
    "S&P 500": "^GSPC",
    "Nasdaq 100": "^NDX",
    "Dow": "^DJI",
    "VIX": "^VIX",
    "美国10年期国债收益率": "^TNX",
    "美元指数": "DX-Y.NYB",
    "比特币": "BTC-USD",
}

SECTOR_ETFS = {
    "科技": "XLK",
    "可选消费": "XLY",
    "通信服务": "XLC",
    "工业": "XLI",
    "金融": "XLF",
    "能源": "XLE",
    "公用事业": "XLU",
    "必需消费": "XLP",
    "医疗保健": "XLV",
    "房地产": "XLRE",
    "材料": "XLB",
}


@dataclass(frozen=True)
class StockSignal:
    symbol: str
    price: float
    daily_return_pct: float
    return_5d_pct: float
    return_20d_pct: float
    return_60d_pct: float
    volume_ratio: float
    ma20: float
    ma50: float
    volatility_20d_pct: float
    rsi_14: float
    score: float
    reasons: tuple[str, ...] = field(default_factory=tuple)
    action: str = "Watch"
    current_price: float = 0.0
    entry_zone_low: float | None = None
    entry_zone_high: float | None = None
    entry_zone_status: str = "暂无可靠价位"
    breakout_trigger: float | None = None
    stop_loss: float | None = None
    target_1: float | None = None
    target_2: float | None = None
    risk_reward_ratio: float | None = None
    level_basis: str = "暂无可靠价位"
    data_timestamp: str = ""


@dataclass(frozen=True)
class NewsItem:
    title: str
    url: str
    source: str = ""
    published_date: str = ""
    snippet: str = ""


@dataclass(frozen=True)
class MarketMetric:
    name: str
    symbol: str
    value: float | None = None
    change_pct: float | None = None
    data_timestamp: str = "暂无数据"
    unit: str = ""


@dataclass(frozen=True)
class MarketContext:
    metrics: dict[str, MarketMetric] = field(default_factory=dict)
    strongest_sector: MarketMetric | None = None
    weakest_sector: MarketMetric | None = None
    sector_metrics: list[MarketMetric] = field(default_factory=list)
    data_timestamp: str = "暂无数据"


@dataclass(frozen=True)
class CandidateReason:
    technical: str
    catalyst: str
    risk: str


@dataclass(frozen=True)
class CandidateView:
    signal: StockSignal
    confidence: float
    holding_period: str
    reason: CandidateReason
    avoid_category: str = ""


@dataclass(frozen=True)
class ReportOutcome:
    ranked: list[StockSignal]
    gemini_candidates: list[StockSignal]
    news: dict[str, list[NewsItem]]
    market_context: MarketContext
    market_news: list[NewsItem]
    gemini_summary: str | None
    report_text: str
    telegram_text: str
    report_path: Path | None
    telegram_sent: bool


MarketFetcher = Callable[[Sequence[str], str], Mapping[str, pd.DataFrame]]
NewsFetcher = Callable[[Sequence[StockSignal]], Mapping[str, list[NewsItem]]]
MarketContextFetcher = Callable[[], MarketContext]
MarketNewsFetcher = Callable[[], list[NewsItem]]
GeminiSummarizer = Callable[..., str | None]
TelegramSender = Callable[[str], bool]


def split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    normalized = value.replace("\n", ",").replace(";", ",").replace("，", ",").replace("、", ",")
    symbols: list[str] = []
    seen: set[str] = set()
    for part in normalized.split(","):
        symbol = part.strip().upper()
        if symbol and symbol not in seen:
            seen.add(symbol)
            symbols.append(symbol)
    return symbols


def split_env_list(value: str | None) -> list[str]:
    if not value:
        return []
    normalized = value.replace("\n", ",").replace(";", ",").replace("，", ",").replace("、", ",")
    values: list[str] = []
    seen: set[str] = set()
    for part in normalized.split(","):
        item = part.strip()
        if item and item not in seen:
            seen.add(item)
            values.append(item)
    return values


def clamp_top_n(value: int | str | None) -> int:
    try:
        parsed = int(value) if value is not None else DEFAULT_TOP_N
    except (TypeError, ValueError):
        parsed = DEFAULT_TOP_N
    return max(1, min(parsed, MAX_GEMINI_STOCKS))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _pct_change(series: pd.Series, periods: int) -> float:
    clean = series.dropna()
    if len(clean) <= periods:
        return 0.0
    start = _safe_float(clean.iloc[-periods - 1])
    end = _safe_float(clean.iloc[-1])
    if start <= 0:
        return 0.0
    return (end / start - 1.0) * 100.0


def _rsi(close: pd.Series, window: int = 14) -> float:
    clean = close.dropna()
    if len(clean) <= window:
        return 50.0
    delta = clean.diff()
    gains = delta.clip(lower=0).rolling(window).mean()
    losses = (-delta.clip(upper=0)).rolling(window).mean()
    latest_loss = _safe_float(losses.iloc[-1])
    if latest_loss == 0:
        return 100.0
    rs = _safe_float(gains.iloc[-1]) / latest_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _ema(close: pd.Series, span: int) -> float:
    clean = close.dropna()
    if len(clean) < span:
        return 0.0
    return _safe_float(clean.ewm(span=span, adjust=False).mean().iloc[-1])


def _atr(frame: pd.DataFrame, window: int = 14) -> float:
    if len(frame) < window + 1 or not {"High", "Low", "Close"}.issubset(frame.columns):
        return 0.0
    high = pd.to_numeric(frame["High"], errors="coerce")
    low = pd.to_numeric(frame["Low"], errors="coerce")
    close = pd.to_numeric(frame["Close"], errors="coerce")
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return _safe_float(true_range.rolling(window).mean().iloc[-1])


def _latest_timestamp(frame: pd.DataFrame) -> str:
    if frame.empty:
        return datetime.now(timezone.utc).date().isoformat()
    latest = frame.index[-1]
    try:
        return pd.Timestamp(latest).date().isoformat()
    except Exception:
        return datetime.now(timezone.utc).date().isoformat()


def _round_price(value: float | None) -> float | None:
    if value is None or not math.isfinite(value) or value <= 0:
        return None
    if value >= 100:
        return round(value, 2)
    if value >= 10:
        return round(value, 2)
    return round(value, 3)


def _degraded_levels(symbol: str, current_price: float, data_timestamp: str, reason: str, score: float) -> dict[str, Any]:
    action = "Watch" if score >= 55 else "Avoid"
    return {
        "action": action,
        "current_price": _round_price(current_price) or 0.0,
        "entry_zone_low": None,
        "entry_zone_high": None,
        "entry_zone_status": "暂无可靠价位",
        "breakout_trigger": None,
        "stop_loss": None,
        "target_1": None,
        "target_2": None,
        "risk_reward_ratio": None,
        "level_basis": f"{reason}，{symbol} 暂不生成交易价位",
        "data_timestamp": data_timestamp,
    }


def calculate_trade_levels(
    symbol: str,
    frame: pd.DataFrame,
    *,
    score: float,
    volatility_20d_pct: float,
) -> dict[str, Any]:
    """Calculate deterministic trade levels from Yahoo Finance technical data."""
    data_timestamp = _latest_timestamp(frame)
    if frame is None or frame.empty or not {"High", "Low", "Close"}.issubset(frame.columns):
        return _degraded_levels(symbol, 0.0, data_timestamp, "数据不足", score)

    close = pd.to_numeric(frame["Close"], errors="coerce").dropna()
    high = pd.to_numeric(frame["High"], errors="coerce").dropna()
    low = pd.to_numeric(frame["Low"], errors="coerce").dropna()
    if len(close) < 50 or len(high) < 20 or len(low) < 20:
        current = _safe_float(close.iloc[-1]) if not close.empty else 0.0
        return _degraded_levels(symbol, current, data_timestamp, "数据不足", score)

    current_price = _safe_float(close.iloc[-1])
    ema20 = _ema(close, 20)
    ema50 = _ema(close, 50)
    atr14 = _atr(frame, 14)
    high_20 = _safe_float(high.tail(20).max())
    low_20 = _safe_float(low.tail(20).min())
    support_10 = _safe_float(low.tail(10).min())
    technical_values = [current_price, ema20, ema50, atr14, high_20, low_20, support_10]
    if any(value <= 0 or not math.isfinite(value) for value in technical_values):
        return _degraded_levels(symbol, current_price, data_timestamp, "技术指标异常", score)

    bullish = current_price > ema20 and ema20 >= ema50
    if not bullish:
        return _degraded_levels(symbol, current_price, data_timestamp, "趋势不明确", score)

    entry_low = max(ema20, current_price - 0.75 * atr14)
    entry_high = current_price - 0.25 * atr14
    breakout_trigger = high_20 * 1.002
    stop_loss = min(support_10, entry_low - 1.0 * atr14)
    target_1 = current_price + 1.5 * atr14
    target_2 = current_price + 2.5 * atr14
    near_20d_high = current_price >= high_20 * 0.985

    values = [entry_low, entry_high, breakout_trigger, stop_loss, target_1, target_2]
    if any(value <= 0 or not math.isfinite(value) for value in values):
        return _degraded_levels(symbol, current_price, data_timestamp, "价位计算异常", score)
    if entry_low > entry_high:
        return _degraded_levels(symbol, current_price, data_timestamp, "买入区间异常", score)
    if not (stop_loss < entry_low or stop_loss < breakout_trigger):
        return _degraded_levels(symbol, current_price, data_timestamp, "止损位置异常", score)
    if not (target_1 > entry_high and target_2 > target_1):
        return _degraded_levels(symbol, current_price, data_timestamp, "目标价异常", score)

    entry_reference = breakout_trigger if near_20d_high else entry_high
    risk = entry_reference - stop_loss
    reward = target_1 - entry_reference
    if risk <= 0 or reward <= 0:
        return _degraded_levels(symbol, current_price, data_timestamp, "风险收益异常", score)
    risk_reward = reward / risk

    if near_20d_high:
        action = "Watch"
        entry_status = "等待回调"
        basis = (
            f"价格接近20日高点，回调区参考 EMA20/ATR，突破触发为20日高点上方0.2%；"
            f"ATR14={atr14:.2f}，波动率={volatility_20d_pct:.1f}%"
        )
    elif risk_reward >= 1.5:
        action = "Buy"
        entry_status = "回调买入区"
        basis = (
            f"偏多趋势且站上 EMA20，买入区由 EMA20 与 ATR14 回撤计算；"
            f"ATR14={atr14:.2f}，20日高点={high_20:.2f}"
        )
    else:
        action = "Watch"
        entry_status = "回调买入区"
        basis = f"趋势偏多但风险回报比 {risk_reward:.2f} 低于 1.5，暂不标记 Buy"

    return {
        "action": action,
        "current_price": _round_price(current_price) or 0.0,
        "entry_zone_low": _round_price(entry_low),
        "entry_zone_high": _round_price(entry_high),
        "entry_zone_status": entry_status,
        "breakout_trigger": _round_price(breakout_trigger),
        "stop_loss": _round_price(stop_loss),
        "target_1": _round_price(target_1),
        "target_2": _round_price(target_2),
        "risk_reward_ratio": round(risk_reward, 2),
        "level_basis": basis,
        "data_timestamp": data_timestamp,
    }


def download_yfinance_history(symbols: Sequence[str], period: str = DEFAULT_PERIOD) -> dict[str, pd.DataFrame]:
    """Fetch daily bars from Yahoo Finance before any Tavily or Gemini calls."""
    import yfinance as yf

    if not symbols:
        return {}

    data = yf.download(
        tickers=" ".join(symbols),
        period=period,
        interval="1d",
        group_by="ticker",
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if data.empty:
        return {}

    frames: dict[str, pd.DataFrame] = {}
    if isinstance(data.columns, pd.MultiIndex):
        for symbol in symbols:
            if symbol in data.columns.get_level_values(0):
                frame = data[symbol].dropna(how="all")
                if not frame.empty:
                    frames[symbol] = frame
    elif len(symbols) == 1:
        frames[symbols[0]] = data.dropna(how="all")

    return frames


def _download_yfinance_group(symbols: Sequence[str], period: str = "5d") -> dict[str, pd.DataFrame]:
    import yfinance as yf

    if not symbols:
        return {}
    data = yf.download(
        tickers=" ".join(symbols),
        period=period,
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        progress=False,
        threads=True,
    )
    if data.empty:
        return {}
    frames: dict[str, pd.DataFrame] = {}
    if isinstance(data.columns, pd.MultiIndex):
        for symbol in symbols:
            if symbol in data.columns.get_level_values(0):
                frame = data[symbol].dropna(how="all")
                if not frame.empty:
                    frames[symbol] = frame
    elif len(symbols) == 1:
        frames[symbols[0]] = data.dropna(how="all")
    return frames


def _metric_from_frame(name: str, symbol: str, frame: pd.DataFrame, *, unit: str = "") -> MarketMetric:
    if frame is None or frame.empty or "Close" not in frame:
        return MarketMetric(name=name, symbol=symbol, unit=unit)
    close = pd.to_numeric(frame["Close"], errors="coerce").dropna()
    if close.empty:
        return MarketMetric(name=name, symbol=symbol, unit=unit)
    value = _safe_float(close.iloc[-1])
    if symbol == "^TNX" and value > 20:
        value = value / 10.0
    change_pct = None
    if len(close) >= 2:
        previous = _safe_float(close.iloc[-2])
        if previous > 0:
            change_pct = (value / previous - 1.0) * 100.0
    return MarketMetric(
        name=name,
        symbol=symbol,
        value=_round_price(value),
        change_pct=round(change_pct, 2) if change_pct is not None and math.isfinite(change_pct) else None,
        data_timestamp=_latest_timestamp(frame),
        unit=unit,
    )


def fetch_market_context() -> MarketContext:
    symbols = list(MARKET_INDICES.values()) + list(SECTOR_ETFS.values())
    frames = _download_yfinance_group(symbols, period="5d")

    metrics = {
        name: _metric_from_frame(
            name,
            symbol,
            frames.get(symbol, pd.DataFrame()),
            unit="%" if symbol == "^TNX" else "",
        )
        for name, symbol in MARKET_INDICES.items()
    }

    sector_metrics = [
        _metric_from_frame(name, symbol, frames.get(symbol, pd.DataFrame()))
        for name, symbol in SECTOR_ETFS.items()
    ]
    available_sectors = [item for item in sector_metrics if item.change_pct is not None]
    strongest = max(available_sectors, key=lambda item: item.change_pct) if available_sectors else None
    weakest = min(available_sectors, key=lambda item: item.change_pct) if available_sectors else None
    timestamps = [item.data_timestamp for item in list(metrics.values()) + sector_metrics if item.data_timestamp != "暂无数据"]
    return MarketContext(
        metrics=metrics,
        strongest_sector=strongest,
        weakest_sector=weakest,
        sector_metrics=sector_metrics,
        data_timestamp=max(timestamps) if timestamps else "暂无数据",
    )


def rank_market_data(history_by_symbol: Mapping[str, pd.DataFrame]) -> list[StockSignal]:
    signals: list[StockSignal] = []
    for symbol, frame in history_by_symbol.items():
        if frame is None or frame.empty:
            continue
        if "Close" not in frame or "Volume" not in frame:
            continue

        close = pd.to_numeric(frame["Close"], errors="coerce").dropna()
        volume = pd.to_numeric(frame["Volume"], errors="coerce").dropna()
        if len(close) < 35 or len(volume) < 20:
            continue

        price = _safe_float(close.iloc[-1])
        prev_price = _safe_float(close.iloc[-2]) if len(close) >= 2 else price
        if price <= 0 or prev_price <= 0:
            continue

        daily_return = (price / prev_price - 1.0) * 100.0
        return_5d = _pct_change(close, 5)
        return_20d = _pct_change(close, 20)
        return_60d = _pct_change(close, min(60, len(close) - 2))
        ma20 = _safe_float(close.tail(20).mean())
        ma50 = _safe_float(close.tail(50).mean()) if len(close) >= 50 else ma20
        avg_volume = _safe_float(volume.iloc[-21:-1].mean())
        volume_ratio = _safe_float(volume.iloc[-1]) / avg_volume if avg_volume > 0 else 1.0
        returns = close.pct_change().dropna().tail(20)
        volatility = _safe_float(returns.std()) * math.sqrt(252) * 100.0 if not returns.empty else 0.0
        rsi_14 = _rsi(close)

        score = 50.0
        score += max(-20.0, min(25.0, return_20d * 1.4))
        score += max(-12.0, min(15.0, return_60d * 0.45))
        score += 12.0 if price > ma20 > ma50 else (-8.0 if price < ma20 else 2.0)
        score += max(-5.0, min(12.0, (volume_ratio - 1.0) * 8.0))
        score += max(-8.0, min(8.0, daily_return * 1.2))
        score += 5.0 if 45.0 <= rsi_14 <= 70.0 else (-6.0 if rsi_14 > 80.0 else -2.0)
        score -= max(0.0, min(10.0, (volatility - 35.0) * 0.25))
        score = max(0.0, min(100.0, score))

        reasons: list[str] = []
        if price > ma20 > ma50:
            reasons.append("uptrend above 20/50-day averages")
        if return_20d > 5:
            reasons.append(f"20d momentum {return_20d:.1f}%")
        if volume_ratio >= 1.4:
            reasons.append(f"volume {volume_ratio:.1f}x normal")
        if 45 <= rsi_14 <= 70:
            reasons.append("RSI constructive")
        if not reasons:
            reasons.append("balanced technical profile")

        trade_levels = calculate_trade_levels(
            symbol,
            frame,
            score=score,
            volatility_20d_pct=volatility,
        )

        signals.append(
            StockSignal(
                symbol=symbol,
                price=price,
                daily_return_pct=daily_return,
                return_5d_pct=return_5d,
                return_20d_pct=return_20d,
                return_60d_pct=return_60d,
                volume_ratio=volume_ratio,
                ma20=ma20,
                ma50=ma50,
                volatility_20d_pct=volatility,
                rsi_14=rsi_14,
                score=score,
                reasons=tuple(reasons[:3]),
                **trade_levels,
            )
        )

    return sorted(signals, key=lambda item: item.score, reverse=True)


def fetch_tavily_news(candidates: Sequence[StockSignal]) -> dict[str, list[NewsItem]]:
    keys = _tavily_keys()
    if not keys or not candidates:
        return {}

    try:
        from tavily import TavilyClient
    except ImportError:
        logger.warning("Tavily package is unavailable; continuing without news enrichment")
        return {}

    client = TavilyClient(api_key=keys[0])
    news: dict[str, list[NewsItem]] = {}
    for candidate in candidates[:MAX_GEMINI_STOCKS]:
        query = f"{candidate.symbol} stock latest earnings analyst catalyst news"
        try:
            response = client.search(
                query=query,
                topic="news",
                search_depth="basic",
                max_results=2,
                include_answer=False,
                include_raw_content=False,
                days=3,
            )
        except Exception as exc:  # noqa: BLE001 - third-party enrichment is fail-open.
            logger.warning("Tavily news lookup failed for %s: %s", candidate.symbol, type(exc).__name__)
            continue

        items: list[NewsItem] = []
        for row in response.get("results", [])[:2]:
            url = str(row.get("url") or "")
            items.append(
                NewsItem(
                    title=str(row.get("title") or "").strip()[:180],
                    url=url,
                    source=_domain(url),
                    published_date=str(row.get("published_date") or row.get("publishedDate") or ""),
                    snippet=str(row.get("content") or "").strip()[:240],
                )
            )
        if items:
            news[candidate.symbol] = items
    return news


def _tavily_keys() -> list[str]:
    return split_env_list(os.getenv("TAVILY_API_KEY")) or split_env_list(os.getenv("TAVILY_API_KEYS"))


def fetch_market_news() -> list[NewsItem]:
    keys = _tavily_keys()
    if not keys:
        return []
    try:
        from tavily import TavilyClient
    except ImportError:
        logger.warning("Tavily package is unavailable; continuing without market news")
        return []

    client = TavilyClient(api_key=keys[0])
    query = (
        "US stock market overnight catalysts S&P 500 Nasdaq Dow VIX Treasury yield dollar Bitcoin sector performance"
    )
    try:
        response = client.search(
            query=query,
            topic="news",
            search_depth="basic",
            max_results=3,
            include_answer=False,
            include_raw_content=False,
            days=2,
        )
    except Exception as exc:  # noqa: BLE001 - market news enrichment is fail-open.
        logger.warning("Tavily market news lookup failed: %s", type(exc).__name__)
        return []

    items: list[NewsItem] = []
    for row in response.get("results", [])[:3]:
        url = str(row.get("url") or "")
        items.append(
            NewsItem(
                title=str(row.get("title") or "").strip()[:180],
                url=url,
                source=_domain(url),
                published_date=str(row.get("published_date") or row.get("publishedDate") or ""),
                snippet=str(row.get("content") or "").strip()[:260],
            )
        )
    return items


def _domain(url: str) -> str:
    if "://" not in url:
        return ""
    return url.split("://", 1)[1].split("/", 1)[0].replace("www.", "")


def _first_configured_key(*names: str) -> str:
    for name in names:
        keys = split_env_list(os.getenv(name))
        if keys:
            return keys[0]
    return ""


def summarize_with_gemini(
    candidates: Sequence[StockSignal],
    news: Mapping[str, list[NewsItem]],
    market_context: MarketContext | None = None,
    market_news: Sequence[NewsItem] | None = None,
) -> str | None:
    if not candidates:
        return None
    api_key = _first_configured_key(
        "GEMINI_API_KEY",
        "GEMINI_API_KEYS",
        "LLM_GEMINI_API_KEY",
        "LLM_GEMINI_API_KEYS",
    )
    if not api_key:
        return None

    try:
        from litellm import completion
    except ImportError:
        logger.warning("LiteLLM is unavailable; continuing without Gemini summary")
        return None

    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash"
    model = model_name if model_name.startswith(("gemini/", "vertex_ai/")) else f"gemini/{model_name}"
    payload = [
        {
            "symbol": item.symbol,
            "score": round(item.score, 1),
            "price": round(item.price, 2),
            "daily_return_pct": round(item.daily_return_pct, 2),
            "return_20d_pct": round(item.return_20d_pct, 2),
            "return_60d_pct": round(item.return_60d_pct, 2),
            "volume_ratio": round(item.volume_ratio, 2),
            "rsi_14": round(item.rsi_14, 1),
            "reasons": list(item.reasons),
            "program_calculated_levels": {
                "action": item.action,
                "current_price": item.current_price,
                "entry_zone_low": item.entry_zone_low,
                "entry_zone_high": item.entry_zone_high,
                "entry_zone_status": item.entry_zone_status,
                "breakout_trigger": item.breakout_trigger,
                "stop_loss": item.stop_loss,
                "target_1": item.target_1,
                "target_2": item.target_2,
                "risk_reward_ratio": item.risk_reward_ratio,
                "level_basis": item.level_basis,
                "data_timestamp": item.data_timestamp,
            },
            "news": [
                {"title": news_item.title, "source": news_item.source, "snippet": news_item.snippet}
                for news_item in news.get(item.symbol, [])[:2]
            ],
        }
        for item in candidates[:MAX_GEMINI_STOCKS]
    ]
    market_payload = {
        "metrics": {
            name: {
                "value": metric.value,
                "change_pct": metric.change_pct,
                "data_timestamp": metric.data_timestamp,
                "unit": metric.unit,
            }
            for name, metric in (market_context.metrics.items() if market_context else [])
        },
        "strongest_sector": market_context.strongest_sector.name if market_context and market_context.strongest_sector else None,
        "weakest_sector": market_context.weakest_sector.name if market_context and market_context.weakest_sector else None,
        "market_news": [
            {
                "title": item.title,
                "source": item.source,
                "published_date": item.published_date,
                "snippet": item.snippet,
            }
            for item in list(market_news or [])[:3]
        ],
    }
    prompt = (
        "You are preparing a concise pre-market US stock opportunity report. "
        "Use only the supplied ranked candidates. Your job is limited to explaining news/catalysts, "
        "summarizing risks, and briefly commenting on the program-calculated technical setup. "
        "Do not overwrite, recalculate, or invent current price, entry zones, breakout triggers, "
        "stop losses, targets, risk/reward ratios, index levels, VIX, yields, DXY, BTC, or sector changes. "
        "Return Chinese text only: prioritize the three most important catalysts and explain why they matter. "
        "Do not output any numeric market data unless it appears exactly in the JSON. "
        "Do not mention API keys or internal implementation details.\n\n"
        f"Market context JSON:\n{json.dumps(market_payload, ensure_ascii=False)}\n\n"
        f"Ranked candidates JSON:\n{json.dumps(payload, ensure_ascii=False)}"
    )
    try:
        response = completion(
            model=model,
            messages=[
                {"role": "system", "content": "You write concise trading briefs, not investment advice."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=900,
            api_key=api_key,
        )
    except Exception as exc:  # noqa: BLE001 - fail open to quantitative report.
        logger.warning("Gemini summary failed: %s", type(exc).__name__)
        return None

    content = response.choices[0].message.content if getattr(response, "choices", None) else None
    return str(content).strip() if content else None


def build_report(
    ranked: Sequence[StockSignal],
    *,
    news: Mapping[str, list[NewsItem]] | None = None,
    gemini_summary: str | None = None,
    generated_at: datetime | None = None,
) -> str:
    news = news or {}
    generated_at = generated_at or datetime.now(timezone.utc)
    lines = [
        f"# US Morning Stock Report - {generated_at.strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        f"Scanned candidates shown: {len(ranked)}",
        "",
    ]
    if gemini_summary:
        lines.extend(["## Gemini catalyst/risk notes", gemini_summary.strip(), ""])

    lines.append("## Top Opportunities")
    for index, item in enumerate(ranked[:MAX_GEMINI_STOCKS], 1):
        reason = "; ".join(item.reasons)
        lines.append(
            f"{index}. {item.symbol} | score {item.score:.1f} | {item.action} | "
            f"${item.current_price:.2f} | "
            f"1d {item.daily_return_pct:+.1f}% | 20d {item.return_20d_pct:+.1f}% | "
            f"vol {item.volume_ratio:.1f}x | RSI {item.rsi_14:.0f}"
        )
        lines.append(f"   Entry zone: {_format_entry_zone(item)}")
        lines.append(f"   Breakout trigger: {_format_price(item.breakout_trigger)}")
        lines.append(f"   Stop loss: {_format_price(item.stop_loss)}")
        lines.append(f"   Target 1 / 2: {_format_price(item.target_1)} / {_format_price(item.target_2)}")
        lines.append(f"   Risk/reward: {_format_ratio(item.risk_reward_ratio)}")
        lines.append(f"   Basis: {item.level_basis}")
        lines.append(f"   Action cue: {reason}.")
        for news_item in news.get(item.symbol, [])[:2]:
            source = f" ({news_item.source})" if news_item.source else ""
            lines.append(f"   News: {news_item.title}{source}")
        lines.append("")

    lines.extend(
        [
            "Risk note: informational scan only; confirm liquidity, upcoming events, and personal risk limits before trading.",
        ]
    )
    return "\n".join(lines).strip()


def _format_price(value: float | None) -> str:
    return f"${value:.2f}" if value is not None and value > 0 else "暂无可靠价位"


def _format_ratio(value: float | None) -> str:
    return f"{value:.2f}" if value is not None and value > 0 else "暂无可靠价位"


def _format_metric(metric: MarketMetric | None) -> str:
    if metric is None or metric.value is None:
        return "暂无数据（时间：暂无数据）"
    value = f"{metric.value:.2f}{metric.unit}"
    change = f"，涨跌幅 {metric.change_pct:+.2f}%" if metric.change_pct is not None else "，涨跌幅 暂无数据"
    return f"{value}{change}（时间：{metric.data_timestamp}）"


def _format_sector(metric: MarketMetric | None) -> str:
    if metric is None or metric.change_pct is None:
        return "暂无数据（时间：暂无数据）"
    value_text = f"{metric.value:.2f}" if metric.value is not None else "暂无数据"
    return f"{metric.name} {metric.change_pct:+.2f}%（ETF {metric.symbol} 最新 {value_text}，时间：{metric.data_timestamp}）"


def _format_entry_zone(item: StockSignal) -> str:
    if item.entry_zone_low is None or item.entry_zone_high is None:
        return "暂无可靠价位"
    zone = f"${item.entry_zone_low:.2f}-${item.entry_zone_high:.2f}"
    if item.entry_zone_status == "等待回调":
        return f"等待回调至 {zone}"
    return zone


def calculate_confidence(signal: StockSignal, news_items: Sequence[NewsItem] | None = None) -> float:
    """Deterministic confidence score. Gemini must not generate this."""
    news_items = list(news_items or [])
    technical_score = max(0.0, min(100.0, signal.score))

    if signal.risk_reward_ratio is None or signal.risk_reward_ratio <= 0:
        rr_score = 0.0
    else:
        rr_score = max(0.0, min(100.0, signal.risk_reward_ratio / 2.5 * 100.0))

    reliable_level_count = sum(
        1
        for value in (
            signal.current_price,
            signal.entry_zone_low,
            signal.entry_zone_high,
            signal.breakout_trigger,
            signal.stop_loss,
            signal.target_1,
            signal.target_2,
        )
        if value is not None and value > 0
    )
    data_quality = 100.0 if reliable_level_count >= 7 and signal.data_timestamp else 45.0 if signal.current_price > 0 else 0.0

    weights = {"technical": 50.0, "rr": 20.0, "data": 10.0}
    weighted_total = technical_score * weights["technical"] + rr_score * weights["rr"] + data_quality * weights["data"]
    weight_total = sum(weights.values())

    if news_items:
        best_news = news_items[0]
        text_len = len(f"{best_news.title} {best_news.snippet}".strip())
        news_score = 55.0
        if best_news.source:
            news_score += 15.0
        if best_news.published_date:
            news_score += 10.0
        if text_len >= 80:
            news_score += 20.0
        weighted_total += min(100.0, news_score) * 20.0
        weight_total += 20.0

    if weight_total <= 0:
        return 0.0
    return round(max(0.0, min(100.0, weighted_total / weight_total)), 1)


def estimate_holding_period(signal: StockSignal) -> str:
    """Estimate holding period from target distance and realized volatility."""
    if not (signal.current_price > 0 and signal.target_1 and signal.target_1 > signal.current_price):
        return "等待确认"
    target_distance_pct = (signal.target_1 / signal.current_price - 1.0) * 100.0
    daily_vol_pct = max(0.4, signal.volatility_20d_pct / math.sqrt(252.0))
    expected_days = target_distance_pct / daily_vol_pct if daily_vol_pct > 0 else 10.0
    if expected_days <= 5:
        return "2-5天"
    if expected_days <= 10:
        return "3-10天"
    return "1-3周"


def _macro_risk_phrase(market_context: MarketContext | None) -> str:
    if market_context is None:
        return "宏观数据缺失，需降低仓位。"
    vix = _metric(market_context, "VIX")
    ten_year = _metric(market_context, "美国10年期国债收益率")
    dxy = _metric(market_context, "美元指数")
    risks: list[str] = []
    if vix and vix.value is not None and vix.value > 22:
        risks.append(f"VIX {vix.value:.2f} 偏高")
    if ten_year and ten_year.change_pct is not None and ten_year.change_pct > 1.0:
        risks.append(f"10年期收益率上行 {ten_year.change_pct:+.2f}%")
    if dxy and dxy.change_pct is not None and dxy.change_pct > 0.35:
        risks.append(f"美元指数走强 {dxy.change_pct:+.2f}%")
    return "，".join(risks) + "可能压制风险偏好。" if risks else "若跌破支撑或新闻反转，交易条件失效。"


def build_candidate_reason(
    signal: StockSignal,
    news_items: Sequence[NewsItem] | None = None,
    market_context: MarketContext | None = None,
) -> CandidateReason:
    news_items = list(news_items or [])
    technical = _one_line("；".join(signal.reasons[:2]) or signal.level_basis, 72)
    if signal.risk_reward_ratio is not None and signal.risk_reward_ratio < 1.5:
        technical = _one_line(f"{technical}；但风险回报比 {signal.risk_reward_ratio:.2f} 未达 Buy 门槛", 82)

    if news_items:
        top_news = news_items[0]
        source = f"（{top_news.source}）" if top_news.source else ""
        catalyst = _one_line(f"{top_news.title}{source}", 82)
    else:
        catalyst = "暂无明确 Tavily 新闻催化剂，confidence 已按缺失新闻重新归一化。"

    if signal.entry_zone_status == "暂无可靠价位":
        risk = _one_line(f"{signal.level_basis}。", 82)
    elif signal.volatility_20d_pct > 45:
        risk = _one_line(f"20日年化波动率 {signal.volatility_20d_pct:.1f}% 偏高，容易击穿止损。", 82)
    elif signal.stop_loss and signal.entry_zone_low:
        risk = _one_line(f"若跌破 ${signal.stop_loss:.2f} 或无法守住支撑，交易失效；{_macro_risk_phrase(market_context)}", 96)
    else:
        risk = _one_line(_macro_risk_phrase(market_context), 82)

    return CandidateReason(
        technical=technical or "技术面暂无可靠信号。",
        catalyst=catalyst,
        risk=risk,
    )


def _avoid_category(signal: StockSignal, news_items: Sequence[NewsItem] | None = None) -> str:
    news_items = list(news_items or [])
    if signal.entry_zone_status == "暂无可靠价位" or signal.current_price <= 0:
        return "数据不足"
    if signal.risk_reward_ratio is None or signal.risk_reward_ratio < 1.5:
        return "风险回报不足"
    if signal.volatility_20d_pct > 55:
        return "波动过高"
    if not news_items:
        return "催化不足"
    if "趋势不明确" in signal.level_basis or signal.action == "Avoid":
        return "趋势弱"
    return "风险回报不足"


def build_candidate_views(
    ranked: Sequence[StockSignal],
    news: Mapping[str, list[NewsItem]] | None = None,
    market_context: MarketContext | None = None,
) -> list[CandidateView]:
    news = news or {}
    views: list[CandidateView] = []
    for signal in ranked[:MAX_GEMINI_STOCKS]:
        stock_news = news.get(signal.symbol, [])
        confidence = calculate_confidence(signal, stock_news)
        views.append(
            CandidateView(
                signal=signal,
                confidence=confidence,
                holding_period=estimate_holding_period(signal),
                reason=build_candidate_reason(signal, stock_news, market_context),
                avoid_category=_avoid_category(signal, stock_news),
            )
        )
    return views


def group_candidate_views(views: Sequence[CandidateView]) -> dict[str, list[CandidateView]]:
    groups = {"best": [], "second": [], "watch": [], "avoid": []}
    ordered = sorted(views, key=lambda view: (view.confidence, view.signal.score), reverse=True)

    high_quality = [
        view
        for view in ordered
        if view.signal.action == "Buy"
        and view.confidence >= 70
        and view.signal.risk_reward_ratio is not None
        and view.signal.risk_reward_ratio >= 1.5
    ]
    groups["best"] = high_quality[: QUALITY_LIMITS["best"]]
    used = {view.signal.symbol for view in groups["best"]}
    groups["second"] = [
        view for view in high_quality if view.signal.symbol not in used
    ][: QUALITY_LIMITS["second"]]
    used.update(view.signal.symbol for view in groups["second"])

    groups["watch"] = [
        view
        for view in ordered
        if view.signal.symbol not in used
        and view.signal.action == "Watch"
        and view.confidence >= 45
    ][: QUALITY_LIMITS["watch"]]
    used.update(view.signal.symbol for view in groups["watch"])

    groups["avoid"] = [
        view
        for view in ordered
        if view.signal.symbol not in used
        and (
            view.signal.action == "Avoid"
            or view.confidence < 45
            or view.signal.entry_zone_status == "暂无可靠价位"
            or (view.signal.risk_reward_ratio is not None and view.signal.risk_reward_ratio < 1.5)
        )
    ][: QUALITY_LIMITS["avoid"]]
    return groups


def _one_line(text: str, max_chars: int) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 1].rstrip() + "…"


def _split_sentences(text: str, limit: int) -> list[str]:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    if not cleaned:
        return []
    parts = [part.strip(" -•") for part in re.split(r"(?<=[.!?。！？])\s+|\n+", cleaned) if part.strip(" -•")]
    return [_one_line(part, 160) for part in parts[:limit]]


def _safe_gemini_sentences(text: str | None, limit: int) -> list[str]:
    deny_markers = (
        "$",
        "current price",
        "entry",
        "stop",
        "target",
        "现价",
        "买入区",
        "止损",
        "目标",
        "突破",
    )
    safe: list[str] = []
    for sentence in _split_sentences(text or "", 8):
        lowered = sentence.lower()
        if any(marker in lowered for marker in deny_markers):
            continue
        safe.append(sentence)
        if len(safe) >= limit:
            break
    return safe


def _metric(context: MarketContext | None, name: str) -> MarketMetric | None:
    if context is None:
        return None
    return context.metrics.get(name)


def _market_bias(context: MarketContext | None, ranked: Sequence[StockSignal]) -> tuple[str, str]:
    if context is None:
        return "中性", "关键市场数据不足，先按个股价位执行，避免用主观判断放大仓位。"

    score = 0
    reasons: list[str] = []
    spx = _metric(context, "S&P 500")
    ndx = _metric(context, "Nasdaq 100")
    dow = _metric(context, "Dow")
    vix = _metric(context, "VIX")
    ten_year = _metric(context, "美国10年期国债收益率")
    dxy = _metric(context, "美元指数")
    btc = _metric(context, "比特币")

    index_changes = [item.change_pct for item in (spx, ndx, dow) if item and item.change_pct is not None]
    if index_changes:
        avg_index = sum(index_changes) / len(index_changes)
        if avg_index > 0.35:
            score += 2
            reasons.append(f"三大指数平均涨幅 {avg_index:+.2f}%")
        elif avg_index < -0.35:
            score -= 2
            reasons.append(f"三大指数平均跌幅 {avg_index:+.2f}%")
        else:
            reasons.append(f"三大指数平均变动 {avg_index:+.2f}%")

    if vix and vix.value is not None:
        if vix.value < 18:
            score += 1
            reasons.append(f"VIX {vix.value:.2f} 处于较低风险溢价")
        elif vix.value > 22:
            score -= 1
            reasons.append(f"VIX {vix.value:.2f} 显示避险升温")

    if ten_year and ten_year.change_pct is not None:
        if ten_year.change_pct > 1.0:
            score -= 1
            reasons.append(f"10年期收益率上行 {ten_year.change_pct:+.2f}% 压制估值")
        elif ten_year.change_pct < -1.0:
            score += 1
            reasons.append(f"10年期收益率回落 {ten_year.change_pct:+.2f}% 缓和估值压力")

    if dxy and dxy.change_pct is not None:
        if dxy.change_pct > 0.35:
            score -= 1
            reasons.append(f"美元指数走强 {dxy.change_pct:+.2f}%")
        elif dxy.change_pct < -0.35:
            score += 1
            reasons.append(f"美元指数走弱 {dxy.change_pct:+.2f}%")

    if btc and btc.change_pct is not None:
        if btc.change_pct > 1.5:
            score += 1
            reasons.append(f"比特币上涨 {btc.change_pct:+.2f}% 风险偏好改善")
        elif btc.change_pct < -1.5:
            score -= 1
            reasons.append(f"比特币下跌 {btc.change_pct:+.2f}%")

    buy_count = sum(1 for item in ranked[:MAX_GEMINI_STOCKS] if item.action == "Buy")
    avoid_count = sum(1 for item in ranked[:MAX_GEMINI_STOCKS] if item.action == "Avoid")
    if buy_count >= 4:
        score += 1
        reasons.append(f"扫描 Top 10 中 Buy 有 {buy_count} 只")
    if avoid_count >= 4:
        score -= 1
        reasons.append(f"Avoid 有 {avoid_count} 只")

    if score >= 2:
        bias = "偏多"
    elif score <= -2:
        bias = "偏空"
    else:
        bias = "中性"
    return bias, "；".join(reasons[:4]) if reasons else "可用数据未形成一致方向。"


def _news_line(items: Sequence[NewsItem], index: int) -> str:
    if index >= len(items):
        return f"{index + 1}. 暂无数据"
    item = items[index]
    date = f"，时间：{item.published_date}" if item.published_date else "，时间：暂无数据"
    source = f"，来源：{item.source}" if item.source else ""
    snippet = f"；{_one_line(item.snippet, 45)}" if item.snippet else ""
    return f"{index + 1}. {_one_line(item.title, 90)}{source}{date}{snippet}"


def build_market_summary(
    ranked: Sequence[StockSignal],
    *,
    scanned_count: int,
    market_context: MarketContext | None = None,
    market_news: Sequence[NewsItem] | None = None,
    gemini_summary: str | None = None,
) -> str:
    market_news = list(market_news or [])
    top = list(ranked[:MAX_GEMINI_STOCKS])
    buy_count = sum(1 for item in top if item.action == "Buy")
    watch_count = sum(1 for item in top if item.action == "Watch")
    avoid_count = sum(1 for item in top if item.action == "Avoid")
    bias, bias_reason = _market_bias(market_context, top)
    top_reasons = []
    for item in top[:3]:
        top_reasons.append(
            f"{item.symbol}：{item.action}，评分 {item.score:.1f}，{_one_line(item.level_basis, 32)}"
        )
    gemini_lines = _safe_gemini_sentences(gemini_summary, 2)
    gemini_note = "；".join(gemini_lines) if gemini_lines else "暂无额外模型解读，执行以程序价位和新闻事实为准。"

    lines = [
        f"1. 大盘表现：S&P 500 {_format_metric(_metric(market_context, 'S&P 500'))}；Nasdaq 100 {_format_metric(_metric(market_context, 'Nasdaq 100'))}；Dow {_format_metric(_metric(market_context, 'Dow'))}。",
        f"2. 波动率：VIX {_format_metric(_metric(market_context, 'VIX'))}。若 VIX 反向上行，Buy 只等买入区。",
        f"3. 利率与美元：美国10年期国债收益率 {_format_metric(_metric(market_context, '美国10年期国债收益率'))}；美元指数 {_format_metric(_metric(market_context, '美元指数'))}。同步上行压制成长股估值。",
        f"4. 风险资产：比特币 {_format_metric(_metric(market_context, '比特币'))}。BTC 只作风险偏好参考。",
        f"5. 板块强弱：昨夜最强板块 {_format_sector(market_context.strongest_sector if market_context else None)}；最弱板块 {_format_sector(market_context.weakest_sector if market_context else None)}。",
        f"6. 重要新闻/催化剂：{_news_line(market_news, 0)}；{_news_line(market_news, 1)}；{_news_line(market_news, 2)}。",
        f"7. 扫描结构：扫描 {scanned_count} 只，深入分析 {len(top)} 只；Buy {buy_count}、Watch {watch_count}、Avoid {avoid_count}。Buy 代表价位和 RR 过关。",
        f"8. Top 3 入选原因：{'；'.join(top_reasons) if top_reasons else '暂无数据'}。",
        f"9. 环境判断：当前定性为{bias}，原因是{bias_reason}。若这些条件在盘前反转，以实际价位触发为准。",
        f"10. 今日主要交易策略：Top 3 只在买入区或突破触发附近执行；第4至第10名候选观察；RR<1.5 或数据不足不买。",
        f"11. 今日最大风险：新闻或宏观数据引发开盘跳空，使买入区失效；高开远离买入区则等待回踩。模型补充：{gemini_note}",
    ]
    return "\n".join(lines)


def _market_conclusion(gemini_summary: str | None) -> str:
    sentences = _safe_gemini_sentences(gemini_summary, 2)
    if sentences:
        return " ".join(sentences)
    return "以 Yahoo Finance 技术面排序为主，优先观察趋势偏多且风险回报达标的标的。缺少可靠价位或风险回报不足时保持 Watch/Avoid。"


def _risk_lines(ranked: Sequence[StockSignal], gemini_summary: str | None) -> list[str]:
    lines = _safe_gemini_sentences(gemini_summary, 3)
    defaults = [
        "盘前消息、财报和宏观数据可能导致开盘跳空。",
        "若价格直接远离买入区，避免追高。",
        "单笔风险需按账户承受能力重新校准。",
    ]
    merged: list[str] = []
    for line in lines + defaults:
        if line and line not in merged:
            merged.append(line)
        if len(merged) >= 3:
            break
    return merged


def _strategy_lines(ranked: Sequence[StockSignal]) -> list[str]:
    buy_count = sum(1 for item in ranked[:MAX_GEMINI_STOCKS] if item.action == "Buy")
    watch_count = sum(1 for item in ranked[:MAX_GEMINI_STOCKS] if item.action == "Watch")
    return [
        f"今日优先处理 {buy_count} 个 Buy 候选，其余 {watch_count} 个 Watch 等回调或突破确认。",
        "只在买入区或突破触发附近执行，价格偏离后等待下一轮信号。",
        "若跌破止损，按计划退出，不用新闻叙事替代风控。",
    ]


def build_telegram_report(
    ranked: Sequence[StockSignal],
    *,
    scanned_count: int,
    news: Mapping[str, list[NewsItem]] | None = None,
    market_context: MarketContext | None = None,
    market_news: Sequence[NewsItem] | None = None,
    gemini_summary: str | None = None,
    generated_at: datetime | None = None,
) -> str:
    news = news or {}
    generated_at = generated_at or datetime.now(ZoneInfo("Asia/Kuala_Lumpur"))
    top = list(ranked[:MAX_GEMINI_STOCKS])
    views = build_candidate_views(top, news=news, market_context=market_context)
    groups = group_candidate_views(views)
    lines = [
        f"📊 US Morning Scanner｜{generated_at.strftime('%Y-%m-%d')}",
        f"扫描：{scanned_count}",
        f"深入分析：{len(top)}",
        "",
        "🌍 Market Overview",
        *_market_overview_lines(top, market_context, gemini_summary),
        "",
        "📰 Overnight News",
        *_overnight_news_lines(market_news or []),
        "",
        "🔥 Best Trade",
        *_trade_group_lines(groups["best"], empty_text="今天没有高质量交易机会"),
        "",
        "🥈 Second Choice",
        *_trade_group_lines(groups["second"], empty_text="暂无第二梯队机会"),
        "",
        "👀 Watch",
        *_watch_group_lines(groups["watch"]),
        "",
        "🚫 Avoid",
        *_avoid_group_lines(groups["avoid"]),
        "",
        "📊 Scanner Statistics",
        *_scanner_statistics_lines(top, groups),
        "",
        "🎯 Today’s Plan",
        *_todays_plan_lines(groups, market_context),
    ]
    return "\n".join(lines).strip()


def _market_overview_lines(
    ranked: Sequence[StockSignal],
    market_context: MarketContext | None,
    gemini_summary: str | None,
) -> list[str]:
    bias, reason = _market_bias(market_context, ranked)
    model_note = "；".join(_safe_gemini_sentences(gemini_summary, 1))
    lines = [
        f"- S&P 500：{_format_metric(_metric(market_context, 'S&P 500'))}",
        f"- Nasdaq 100：{_format_metric(_metric(market_context, 'Nasdaq 100'))}；Dow：{_format_metric(_metric(market_context, 'Dow'))}",
        f"- VIX：{_format_metric(_metric(market_context, 'VIX'))}；10Y：{_format_metric(_metric(market_context, '美国10年期国债收益率'))}",
        f"- DXY：{_format_metric(_metric(market_context, '美元指数'))}；BTC：{_format_metric(_metric(market_context, '比特币'))}",
        f"- 板块：最强 {_format_sector(market_context.strongest_sector if market_context else None)}；最弱 {_format_sector(market_context.weakest_sector if market_context else None)}",
        f"- 环境：{bias}。理由：{_one_line(reason, 120)}",
    ]
    if model_note:
        lines.append(f"- 模型压缩：{_one_line(model_note, 100)}")
    return lines


def _overnight_news_lines(market_news: Sequence[NewsItem]) -> list[str]:
    if not market_news:
        return ["- 暂无数据"]
    return [f"- {_news_line(market_news, index)}" for index in range(min(3, len(market_news)))]


def _reason_lines(view: CandidateView) -> list[str]:
    return [
        f"Reason-技术面：{view.reason.technical}",
        f"Reason-催化剂：{view.reason.catalyst}",
        f"Reason-风险：{view.reason.risk}",
    ]


def _full_trade_lines(view: CandidateView) -> list[str]:
    item = view.signal
    lines = [
        f"{item.symbol}｜Score {item.score:.1f}｜Confidence {view.confidence:.1f}｜{item.action}",
        f"现价：${item.current_price:.2f}｜买入区：{_format_entry_zone(item)}｜突破：{_format_price(item.breakout_trigger)}",
        f"止损：{_format_price(item.stop_loss)}｜目标1：{_format_price(item.target_1)}｜目标2：{_format_price(item.target_2)}｜RR：{_format_ratio(item.risk_reward_ratio)}",
        f"建议持有周期：{view.holding_period}",
    ]
    lines.extend(_reason_lines(view))
    return lines


def _trade_group_lines(views: Sequence[CandidateView], *, empty_text: str) -> list[str]:
    if not views:
        return [empty_text]
    lines: list[str] = []
    for view in views:
        lines.extend(_full_trade_lines(view))
        lines.append("")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def _watch_group_lines(views: Sequence[CandidateView]) -> list[str]:
    if not views:
        return ["暂无值得等待的候选股"]
    lines: list[str] = []
    for view in views[: QUALITY_LIMITS["watch"]]:
        item = view.signal
        trigger = _format_entry_zone(item)
        if trigger == "暂无可靠价位":
            trigger = f"突破 {_format_price(item.breakout_trigger)}"
        wait_condition = "等待回调进入买入区或放量突破触发价"
        if item.risk_reward_ratio is not None and item.risk_reward_ratio < 1.5:
            wait_condition = "等待风险回报比修复到 1.5 以上"
        lines.extend(
            [
                f"{item.symbol}｜{trigger}｜止损 {_format_price(item.stop_loss)}｜目标1 {_format_price(item.target_1)}",
                f"Reason-技术面：{view.reason.technical}",
                f"等待条件：{wait_condition}",
            ]
        )
    return lines


def _avoid_group_lines(views: Sequence[CandidateView]) -> list[str]:
    if not views:
        return ["暂无明确 Avoid"]
    lines: list[str] = []
    for view in views[: QUALITY_LIMITS["avoid"]]:
        item = view.signal
        lines.append(f"{item.symbol}｜Reason：{view.avoid_category}。{view.reason.technical}；{view.reason.risk}")
    return lines


def _scanner_statistics_lines(top: Sequence[StockSignal], groups: Mapping[str, list[CandidateView]]) -> list[str]:
    action_counts = {
        "Buy": sum(1 for item in top if item.action == "Buy"),
        "Watch": sum(1 for item in top if item.action == "Watch"),
        "Avoid": sum(1 for item in top if item.action == "Avoid"),
    }
    return [
        f"- Top 10 原始动作：Buy {action_counts['Buy']}｜Watch {action_counts['Watch']}｜Avoid {action_counts['Avoid']}",
        f"- 输出分组：Best {len(groups['best'])}｜Second {len(groups['second'])}｜Watch {len(groups['watch'])}｜Avoid {len(groups['avoid'])}",
    ]


def _todays_plan_lines(groups: Mapping[str, list[CandidateView]], market_context: MarketContext | None) -> list[str]:
    if not groups["best"] and not groups["second"]:
        plan = "今天没有高质量交易机会，优先等待回调、突破确认或新的催化剂。"
    else:
        symbols = [view.signal.symbol for view in groups["best"] + groups["second"]]
        plan = f"只处理 {', '.join(symbols)} 的买入区/突破触发，其他股票不追价。"
    return [
        f"- {plan}",
        "- 所有交易以程序计算的止损和目标为准，Gemini 只作新闻解释。",
        f"- 主要风险：{_macro_risk_phrase(market_context)}",
    ]


def split_telegram_message(content: str, max_chars: int = TELEGRAM_TARGET_CHARS) -> list[str]:
    text = content.strip()
    if len(text) <= max_chars:
        return [text]

    preferred_marker = "\n\n👀 Watch"
    if preferred_marker in text:
        first, rest = text.split(preferred_marker, 1)
        second = "👀 Watch" + rest
        if len(first) <= max_chars and len(second) <= max_chars:
            return [first, second]

    blocks = text.split("\n\n")
    top_blocks_seen = 0
    split_index = None
    for index, block in enumerate(blocks):
        if block.startswith("🥇 "):
            top_blocks_seen += 1
            if top_blocks_seen == 3:
                split_index = index + 1
                break
    if split_index is not None:
        first = "\n\n".join(block.strip() for block in blocks[:split_index] if block.strip())
        second = "\n\n".join(block.strip() for block in blocks[split_index:] if block.strip())
        if first and second and len(first) <= max_chars and len(second) <= max_chars:
            return [first, second]

    chunks: list[str] = []
    current = ""
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        candidate = block if not current else f"{current}\n\n{block}"
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
        if len(block) <= max_chars:
            current = block
        else:
            lines = block.splitlines()
            current = ""
            for line in lines:
                line_candidate = line if not current else f"{current}\n{line}"
                if len(line_candidate) <= max_chars:
                    current = line_candidate
                else:
                    if current:
                        chunks.append(current)
                    current = line[:max_chars]
    if current:
        chunks.append(current)
    return chunks or [text[:max_chars]]


def _call_gemini_summarizer(
    summarizer: GeminiSummarizer,
    candidates: Sequence[StockSignal],
    news: Mapping[str, list[NewsItem]],
    market_context: MarketContext,
    market_news: Sequence[NewsItem],
) -> str | None:
    try:
        return summarizer(candidates, news, market_context, market_news)
    except TypeError:
        return summarizer(candidates, news)


def send_telegram_summary(content: str) -> bool:
    import requests

    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    thread_id = os.getenv("TELEGRAM_MESSAGE_THREAD_ID", "").strip()
    if not token or not chat_id:
        logger.warning("Telegram is not configured; report was generated but not sent")
        return False

    chunks = split_telegram_message(content, max_chars=TELEGRAM_TARGET_CHARS)
    api_url = f"https://api.telegram.org/bot{token}/sendMessage"
    for index, text in enumerate(chunks, 1):
        payload: dict[str, Any] = {
            "chat_id": chat_id,
            "text": text,
            "disable_web_page_preview": True,
        }
        if thread_id:
            payload["message_thread_id"] = thread_id
        response = requests.post(api_url, json=payload, timeout=15)
        if not (response.status_code == 200 and response.json().get("ok")):
            logger.error("Telegram summary chunk %s/%s failed with HTTP %s", index, len(chunks), response.status_code)
            return False
    logger.info("Telegram summary sent in %s chunk(s)", len(chunks))
    return True


def run_us_morning_report(
    *,
    symbols: Sequence[str] | None = None,
    period: str = DEFAULT_PERIOD,
    top_n: int | str | None = DEFAULT_TOP_N,
    market_fetcher: MarketFetcher = download_yfinance_history,
    market_context_fetcher: MarketContextFetcher = fetch_market_context,
    news_fetcher: NewsFetcher = fetch_tavily_news,
    market_news_fetcher: MarketNewsFetcher = fetch_market_news,
    gemini_summarizer: GeminiSummarizer = summarize_with_gemini,
    telegram_sender: TelegramSender = send_telegram_summary,
    send_telegram: bool = True,
    output_dir: Path | str | None = "reports",
) -> ReportOutcome:
    requested_symbols = list(symbols or split_csv(os.getenv("US_STOCK_UNIVERSE")) or split_csv(DEFAULT_UNIVERSE))
    effective_top_n = clamp_top_n(top_n)
    if not requested_symbols:
        raise ValueError("US stock universe is empty")

    logger.info("Fetching Yahoo Finance daily data for %s symbols", len(requested_symbols))
    history = market_fetcher(requested_symbols, period)
    ranked_all = rank_market_data(history)
    ranked = ranked_all[:effective_top_n]
    if not ranked:
        raise RuntimeError("No US stock candidates could be ranked from Yahoo Finance data")

    gemini_candidates = ranked[:MAX_GEMINI_STOCKS]
    market_context = market_context_fetcher()
    market_news = list(market_news_fetcher())
    news = dict(news_fetcher(gemini_candidates))
    gemini_summary = _call_gemini_summarizer(
        gemini_summarizer,
        gemini_candidates,
        news,
        market_context,
        market_news,
    )
    market_summary = build_market_summary(
        ranked,
        scanned_count=len(requested_symbols),
        market_context=market_context,
        market_news=market_news,
        gemini_summary=gemini_summary,
    )
    report_text = f"# Market Summary\n\n{market_summary}\n\n{build_report(ranked, news=news, gemini_summary=gemini_summary)}"
    telegram_text = build_telegram_report(
        ranked,
        scanned_count=len(requested_symbols),
        news=news,
        market_context=market_context,
        market_news=market_news,
        gemini_summary=gemini_summary,
    )

    report_path: Path | None = None
    if output_dir is not None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        report_path = path / f"us_morning_report_{datetime.now(timezone.utc).strftime('%Y%m%d')}.md"
        report_path.write_text(report_text + "\n", encoding="utf-8")

    telegram_sent = telegram_sender(telegram_text) if send_telegram else False
    return ReportOutcome(
        ranked=list(ranked),
        gemini_candidates=list(gemini_candidates),
        news=news,
        market_context=market_context,
        market_news=list(market_news),
        gemini_summary=gemini_summary,
        report_text=report_text,
        telegram_text=telegram_text,
        report_path=report_path,
        telegram_sent=telegram_sent,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate one concise US morning stock report.")
    parser.add_argument("--symbols", help="Comma-separated US symbols. Defaults to US_STOCK_UNIVERSE or built-in universe.")
    parser.add_argument("--top-n", default=os.getenv("US_REPORT_TOP_N", str(DEFAULT_TOP_N)), help="Top names to report; hard-capped at 10.")
    parser.add_argument("--period", default=os.getenv("US_REPORT_YFINANCE_PERIOD", DEFAULT_PERIOD), help="Yahoo Finance history period.")
    parser.add_argument("--output-dir", default=os.getenv("US_REPORT_OUTPUT_DIR", "reports"), help="Directory for Markdown report artifact.")
    parser.add_argument("--no-telegram", action="store_true", help="Generate the report without sending Telegram.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    symbols = split_csv(args.symbols) if args.symbols else None
    outcome = run_us_morning_report(
        symbols=symbols,
        period=args.period,
        top_n=args.top_n,
        send_telegram=not args.no_telegram,
        output_dir=args.output_dir,
    )
    if outcome.report_path:
        logger.info("Report written to %s", outcome.report_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
