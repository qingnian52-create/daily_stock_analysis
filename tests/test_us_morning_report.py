# -*- coding: utf-8 -*-
from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from src.services.us_morning_report import (
    MAX_GEMINI_STOCKS,
    MarketContext,
    MarketMetric,
    NewsItem,
    StockSignal,
    build_market_summary,
    build_telegram_report,
    build_candidate_reason,
    build_candidate_views,
    calculate_confidence,
    calculate_trade_levels,
    clamp_top_n,
    estimate_holding_period,
    group_candidate_views,
    run_us_morning_report,
    split_telegram_message,
    split_csv,
    split_env_list,
)

ROOT = Path(__file__).resolve().parents[1]


def _history_frame(strength: int) -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=75, freq="B")
    base = 80 + strength
    close = [base + index * (0.12 + strength * 0.01) for index in range(len(dates))]
    volume = [1_000_000 + strength * 10_000 for _ in dates]
    volume[-1] = int(volume[-1] * (1.0 + strength / 20.0))
    return pd.DataFrame(
        {
            "Open": close,
            "High": [value * 1.01 for value in close],
            "Low": [value * 0.99 for value in close],
            "Close": close,
            "Volume": volume,
        },
        index=dates,
    )


def _trade_frame(
    *,
    rows: int = 75,
    trend: str = "bullish",
    high_spike: bool = True,
    step: float = 0.65,
    spread: float = 1.4,
) -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=rows, freq="B")
    if trend == "flat":
        close = [100 + ((index % 3) - 1) * 0.1 for index in range(rows)]
    elif trend == "bearish":
        close = [140 - index * 0.45 for index in range(rows)]
    else:
        close = [100 + index * step for index in range(rows)]
    high = [value + spread for value in close]
    low = [value - spread for value in close]
    if high_spike and rows >= 20:
        high[-8] = close[-1] * 1.04
    volume = [1_000_000 + index * 1000 for index in range(rows)]
    return pd.DataFrame({"Open": close, "High": high, "Low": low, "Close": close, "Volume": volume}, index=dates)


def _sample_signal(symbol: str, index: int, action: str = "Buy") -> StockSignal:
    return StockSignal(
        symbol=symbol,
        price=100 + index,
        daily_return_pct=1.0,
        return_5d_pct=3.0,
        return_20d_pct=8.0,
        return_60d_pct=18.0,
        volume_ratio=1.6,
        ma20=95.0,
        ma50=90.0,
        volatility_20d_pct=25.0,
        rsi_14=58.0,
        score=90 - index,
        reasons=("uptrend above 20/50-day averages",),
        action=action,
        current_price=100 + index,
        entry_zone_low=96 + index,
        entry_zone_high=98 + index,
        entry_zone_status="回调买入区",
        breakout_trigger=104 + index,
        stop_loss=93 + index,
        target_1=108 + index,
        target_2=112 + index,
        risk_reward_ratio=2.0,
        level_basis="偏多趋势且站上 EMA20，买入区由 EMA20 与 ATR14 回撤计算",
        data_timestamp="2026-07-10",
    )


def _sample_market_context() -> MarketContext:
    metrics = {
        "S&P 500": MarketMetric("S&P 500", "^GSPC", 5600.12, 0.42, "2026-07-10"),
        "Nasdaq 100": MarketMetric("Nasdaq 100", "^NDX", 20300.45, 0.81, "2026-07-10"),
        "Dow": MarketMetric("Dow", "^DJI", 41020.33, -0.12, "2026-07-10"),
        "VIX": MarketMetric("VIX", "^VIX", 15.2, -3.1, "2026-07-10"),
        "美国10年期国债收益率": MarketMetric("美国10年期国债收益率", "^TNX", 4.25, -0.5, "2026-07-10", "%"),
        "美元指数": MarketMetric("美元指数", "DX-Y.NYB", 104.2, -0.21, "2026-07-10"),
        "比特币": MarketMetric("比特币", "BTC-USD", 118000.0, 1.8, "2026-07-13"),
    }
    strongest = MarketMetric("科技", "XLK", 250.0, 1.25, "2026-07-10")
    weakest = MarketMetric("能源", "XLE", 90.0, -0.72, "2026-07-10")
    return MarketContext(metrics=metrics, strongest_sector=strongest, weakest_sector=weakest, data_timestamp="2026-07-10")


class TestUSMorningReport(unittest.TestCase):
    def test_us_workflow_cron_is_malaysia_8am_tuesday_to_saturday(self) -> None:
        content = (ROOT / ".github/workflows/us-morning-report.yml").read_text(encoding="utf-8")
        self.assertIn("cron: '0 0 * * 2-6'", content)
        self.assertIn("GitHub Actions cron uses UTC", content)
        self.assertIn("workflow_dispatch:", content)
        self.assertIn("contents: write", content)
        self.assertIn("group: us-morning-report-performance", content)
        self.assertIn("cancel-in-progress: false", content)
        self.assertIn("fetch-depth: 0", content)

    def test_us_workflow_performance_write_modes_and_artifact_fallback(self) -> None:
        content = (ROOT / ".github/workflows/us-morning-report.yml").read_text(encoding="utf-8")
        self.assertIn("performance_write:", content)
        self.assertIn("default: false", content)
        self.assertIn('"${{ github.event_name }}" = "schedule"', content)
        self.assertIn("PERFORMANCE_MANUAL_WRITE_DEFAULT", content)
        self.assertIn("PERFORMANCE_WRITE_EFFECTIVE=true", content)
        self.assertIn("args+=(--performance-write)", content)
        self.assertIn("load-branch --root .", content)
        self.assertIn("persist-branch --root .", content)
        self.assertIn("vars.PERFORMANCE_TRACKING_ENABLED != 'false'", content)
        self.assertIn("continue-on-error: true", content)
        self.assertIn("performance/", content)

    def test_us_workflow_does_not_print_secrets(self) -> None:
        content = (ROOT / ".github/workflows/us-morning-report.yml").read_text(encoding="utf-8")
        self.assertNotIn("printenv", content)
        self.assertNotIn("set -x", content)
        self.assertNotIn("echo ${{ secrets.", content)

    def test_legacy_daily_workflow_has_no_schedule(self) -> None:
        content = (ROOT / ".github/workflows/00-daily-analysis.yml").read_text(encoding="utf-8")
        self.assertIn("workflow_dispatch:", content)
        self.assertNotIn("schedule:", content)

    def test_split_csv_accepts_common_separators_and_dedupes(self) -> None:
        self.assertEqual(split_csv("aapl, msft，nvda;AAPL\namzn、tsla"), ["AAPL", "MSFT", "NVDA", "AMZN", "TSLA"])

    def test_split_env_list_preserves_api_key_case(self) -> None:
        self.assertEqual(split_env_list("AbC123, xyz789,AbC123"), ["AbC123", "xyz789"])

    def test_top_n_is_hard_capped_at_gemini_limit(self) -> None:
        self.assertEqual(clamp_top_n(25), MAX_GEMINI_STOCKS)
        self.assertEqual(clamp_top_n("bad"), MAX_GEMINI_STOCKS)
        self.assertEqual(clamp_top_n(0), 1)

    def test_trade_levels_include_entry_breakout_stop_targets_and_rr(self) -> None:
        frame = _trade_frame(step=0.03, spread=0.3)
        levels = calculate_trade_levels("TEST", frame, score=82.0, volatility_20d_pct=22.0)

        self.assertEqual(levels["action"], "Buy")
        self.assertGreater(levels["entry_zone_low"], 0)
        self.assertLessEqual(levels["entry_zone_low"], levels["entry_zone_high"])
        self.assertGreater(levels["breakout_trigger"], frame["High"].tail(20).max())
        self.assertLess(levels["stop_loss"], levels["entry_zone_low"])
        self.assertGreater(levels["target_1"], levels["entry_zone_high"])
        self.assertGreater(levels["target_2"], levels["target_1"])
        self.assertGreaterEqual(levels["risk_reward_ratio"], 1.5)
        self.assertIn("ATR14", levels["level_basis"])

    def test_near_high_keeps_breakout_but_waits_for_pullback(self) -> None:
        frame = _trade_frame(high_spike=False)
        levels = calculate_trade_levels("TEST", frame, score=82.0, volatility_20d_pct=22.0)

        self.assertEqual(levels["action"], "Watch")
        self.assertEqual(levels["entry_zone_status"], "等待回调")
        self.assertGreater(levels["breakout_trigger"], 0)
        self.assertGreater(levels["target_1"], levels["entry_zone_high"])

    def test_insufficient_data_degrades_without_fake_levels(self) -> None:
        frame = _trade_frame(rows=18)
        levels = calculate_trade_levels("TEST", frame, score=80.0, volatility_20d_pct=22.0)

        self.assertEqual(levels["action"], "Watch")
        self.assertEqual(levels["entry_zone_status"], "暂无可靠价位")
        self.assertIsNone(levels["entry_zone_low"])
        self.assertIsNone(levels["stop_loss"])
        self.assertIn("数据不足", levels["level_basis"])

    def test_unclear_trend_cannot_be_buy(self) -> None:
        frame = _trade_frame(trend="bearish")
        levels = calculate_trade_levels("TEST", frame, score=80.0, volatility_20d_pct=22.0)

        self.assertIn(levels["action"], {"Watch", "Avoid"})
        self.assertNotEqual(levels["action"], "Buy")
        self.assertIsNone(levels["target_1"])
        self.assertIn("趋势不明确", levels["level_basis"])

    def test_invalid_price_data_is_rejected(self) -> None:
        frame = _trade_frame()
        frame.loc[frame.index[-1], "Close"] = -10
        levels = calculate_trade_levels("TEST", frame, score=80.0, volatility_20d_pct=22.0)

        self.assertNotEqual(levels["action"], "Buy")
        self.assertEqual(levels["entry_zone_status"], "暂无可靠价位")
        self.assertIsNone(levels["target_1"])

    def test_pipeline_fetches_yfinance_first_and_only_enriches_top_ten(self) -> None:
        symbols = [f"T{i}" for i in range(12)]
        calls: list[str] = []
        captured_news_symbols: list[str] = []
        captured_gemini_symbols: list[str] = []
        telegram_messages: list[str] = []

        def fake_market_fetcher(requested_symbols, period):
            calls.append("market")
            self.assertEqual(list(requested_symbols), symbols)
            self.assertEqual(period, "6mo")
            return {symbol: _history_frame(index + 1) for index, symbol in enumerate(requested_symbols)}

        def fake_news_fetcher(candidates: list[StockSignal]):
            calls.append("news")
            captured_news_symbols.extend(item.symbol for item in candidates)
            return {
                item.symbol: [NewsItem(title=f"{item.symbol} catalyst", url="https://example.com/story")]
                for item in candidates
            }

        def fake_gemini(candidates, news):
            calls.append("gemini")
            captured_gemini_symbols.extend(item.symbol for item in candidates)
            self.assertEqual(set(news), set(captured_news_symbols))
            return "Momentum is concentrated in the highest-ranked candidates."

        def fake_telegram(message: str) -> bool:
            calls.append("telegram")
            telegram_messages.append(message)
            return True

        with TemporaryDirectory() as tmpdir:
            outcome = run_us_morning_report(
                symbols=symbols,
                period="6mo",
                top_n=25,
                market_fetcher=fake_market_fetcher,
                market_context_fetcher=_sample_market_context,
                news_fetcher=fake_news_fetcher,
                market_news_fetcher=lambda: [
                    NewsItem(title="Fed speakers guide rate expectations", url="https://example.com/fed")
                ],
                gemini_summarizer=fake_gemini,
                telegram_sender=fake_telegram,
                output_dir=Path(tmpdir),
            )

        self.assertEqual(calls, ["market", "news", "gemini", "telegram"])
        self.assertEqual(len(outcome.ranked), MAX_GEMINI_STOCKS)
        self.assertEqual(len(captured_news_symbols), MAX_GEMINI_STOCKS)
        self.assertEqual(captured_gemini_symbols, captured_news_symbols)
        self.assertEqual(len(telegram_messages), 1)
        self.assertIn("US Morning Scanner", telegram_messages[0])
        self.assertTrue(outcome.telegram_sent)

    def test_gemini_cannot_override_structured_levels(self) -> None:
        symbols = ["AAPL"]
        structured_stop = None

        def fake_market_fetcher(requested_symbols, period):
            return {"AAPL": _trade_frame()}

        def fake_news_fetcher(candidates):
            return {"AAPL": [NewsItem(title="AAPL catalyst", url="https://example.com")]}

        def fake_gemini(candidates, news):
            nonlocal structured_stop
            structured_stop = candidates[0].stop_loss
            return "Override idea: current price is $1, stop loss is $1, target is $2."

        with TemporaryDirectory() as tmpdir:
            outcome = run_us_morning_report(
                symbols=symbols,
                market_fetcher=fake_market_fetcher,
                market_context_fetcher=_sample_market_context,
                news_fetcher=fake_news_fetcher,
                market_news_fetcher=lambda: [
                    NewsItem(title="Large-cap tech leads overnight", url="https://example.com/market")
                ],
                gemini_summarizer=fake_gemini,
                telegram_sender=lambda message: True,
                output_dir=Path(tmpdir),
            )

        self.assertIsNotNone(structured_stop)
        self.assertIn(f"止损 ${structured_stop:.2f}", outcome.telegram_text)
        self.assertNotIn("Override idea", outcome.telegram_text)

    def test_market_summary_contains_required_sections(self) -> None:
        ranked = [_sample_signal(f"T{index}", index, "Buy" if index < 4 else "Watch") for index in range(10)]
        market_news = [
            NewsItem(title="Fed minutes shift rate-cut expectations", url="https://example.com/1", source="example", published_date="2026-07-10"),
            NewsItem(title="Large-cap semiconductors lead after earnings", url="https://example.com/2", source="example", published_date="2026-07-10"),
            NewsItem(title="Oil slips as energy stocks lag", url="https://example.com/3", source="example", published_date="2026-07-10"),
        ]
        summary = build_market_summary(
            ranked,
            scanned_count=99,
            market_context=_sample_market_context(),
            market_news=market_news,
            gemini_summary="科技股催化剂更集中，利率回落缓和估值压力。",
        )

        for text in (
            "S&P 500",
            "Nasdaq 100",
            "Dow",
            "VIX",
            "美国10年期国债收益率",
            "美元指数",
            "比特币",
            "最强板块",
            "最弱板块",
            "重要新闻/催化剂",
            "Buy 4、Watch 6、Avoid 0",
            "Top 3 入选原因",
            "今日主要交易策略",
            "今日最大风险",
        ):
            self.assertIn(text, summary)
        self.assertGreaterEqual(len(summary.splitlines()), 8)

    def test_market_summary_degrades_missing_data(self) -> None:
        ranked = [_sample_signal("AAPL", 0, "Watch")]
        summary = build_market_summary(ranked, scanned_count=99, market_context=MarketContext(), market_news=[])

        self.assertIn("S&P 500 暂无数据", summary)
        self.assertIn("VIX 暂无数据", summary)
        self.assertIn("1. 暂无数据", summary)
        self.assertIn("当前定性为中性", summary)

    def test_market_summary_filters_gemini_numeric_overrides(self) -> None:
        ranked = [_sample_signal("AAPL", 0, "Buy")]
        text = build_telegram_report(
            ranked,
            scanned_count=99,
            news={"AAPL": [NewsItem(title="AAPL catalyst", url="https://example.com")]},
            market_context=_sample_market_context(),
            market_news=[NewsItem(title="Fed rate expectations move markets", url="https://example.com")],
            gemini_summary="S&P 500 is 999999 and stop loss should be $1. Real catalyst is earnings revision.",
        )

        self.assertIn("S&P 500：5600.12", text)
        self.assertNotIn("999999", text)
        self.assertNotIn("stop loss should be $1", text)
        self.assertIn("Real catalyst is earnings revision", text)

    def test_telegram_grouped_output_preserves_reason(self) -> None:
        ranked = [_sample_signal(f"T{index}", index) for index in range(10)]
        news = {item.symbol: [NewsItem(title=f"{item.symbol} catalyst headline", url="https://example.com")] for item in ranked}
        text = build_telegram_report(
            ranked,
            scanned_count=99,
            news=news,
            market_context=_sample_market_context(),
            market_news=[NewsItem(title="Fed rate expectations move markets", url="https://example.com")],
            gemini_summary="科技催化剂集中。",
        )

        self.assertIn("扫描：99", text)
        self.assertIn("深入分析：10", text)
        self.assertIn("🔥 Best Trade", text)
        self.assertIn("🥈 Second Choice", text)
        self.assertIn("👀 Watch", text)
        self.assertIn("🚫 Avoid", text)
        self.assertIn("T0｜Score 90.0｜Confidence", text)
        self.assertIn("现价：$100.00", text)
        self.assertIn("目标2：$112.00", text)
        self.assertIn("Reason-技术面：", text)
        self.assertIn("Reason-催化剂：", text)
        self.assertIn("Reason-风险：", text)

    def test_best_trade_empty_when_no_high_quality_trade(self) -> None:
        ranked = [_sample_signal(f"W{index}", index, action="Watch") for index in range(4)]
        text = build_telegram_report(
            ranked,
            scanned_count=99,
            news={},
            market_context=_sample_market_context(),
            market_news=[],
            gemini_summary="",
        )

        self.assertIn("🔥 Best Trade\n今天没有高质量交易机会", text)
        self.assertNotIn("W0｜Score", text.split("🔥 Best Trade", 1)[1].split("🥈 Second Choice", 1)[0])

    def test_reason_has_three_parts(self) -> None:
        signal = _sample_signal("AAPL", 0)
        reason = build_candidate_reason(
            signal,
            [NewsItem(title="AAPL earnings catalyst", url="https://example.com", source="example")],
            _sample_market_context(),
        )

        self.assertTrue(reason.technical)
        self.assertTrue(reason.catalyst)
        self.assertTrue(reason.risk)
        self.assertLessEqual(len(reason.technical.split("。")), 2)

    def test_confidence_range_and_news_renormalization(self) -> None:
        signal = _sample_signal("AAPL", 0)
        with_news = calculate_confidence(
            signal,
            [
                NewsItem(
                    title="AAPL earnings catalyst receives analyst upgrade",
                    url="https://example.com",
                    source="example",
                    published_date="2026-07-10",
                    snippet="Analysts lifted earnings estimates after stronger demand signals and margin commentary.",
                )
            ],
        )
        without_news = calculate_confidence(signal, [])

        self.assertGreaterEqual(with_news, 0)
        self.assertLessEqual(with_news, 100)
        self.assertGreaterEqual(without_news, 0)
        self.assertLessEqual(without_news, 100)
        self.assertGreater(with_news, without_news)

    def test_holding_period_is_deterministic(self) -> None:
        self.assertIn(estimate_holding_period(_sample_signal("AAPL", 0)), {"2-5天", "3-10天", "1-3周", "等待确认"})

    def test_grouping_does_not_force_fill_recommendations(self) -> None:
        views = build_candidate_views([_sample_signal("AAPL", 0, action="Buy")], news={}, market_context=_sample_market_context())
        groups = group_candidate_views(views)

        self.assertLessEqual(len(groups["best"]), 1)
        self.assertEqual(len(groups["second"]), 0)
        self.assertEqual(len(groups["watch"]), 0)

    def test_telegram_long_text_splits_on_blocks(self) -> None:
        content = "市场 Summary\n\n" + "\n\n".join(f"BLOCK{i}\n" + ("x" * 90) for i in range(8))
        chunks = split_telegram_message(content, max_chars=180)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 180 for chunk in chunks))
        self.assertTrue(any("BLOCK3" in chunk for chunk in chunks))
        for chunk in chunks:
            self.assertFalse(chunk.endswith("BLOC"))

    def test_telegram_report_prefers_split_before_watch_section(self) -> None:
        first = "🌍 Market Overview\n" + ("s" * 500) + "\n\n🔥 Best Trade\n" + ("b" * 120)
        second = "👀 Watch\nAMZN｜87｜Watch\n" + ("y" * 300) + "\n\n🚫 Avoid\n- risk"
        chunks = split_telegram_message(f"{first}\n\n{second}", max_chars=900)

        self.assertEqual(len(chunks), 2)
        self.assertIn("🔥 Best Trade", chunks[0])
        self.assertNotIn("AMZN｜87｜Watch", chunks[0])
        self.assertIn("AMZN｜87｜Watch", chunks[1])


if __name__ == "__main__":
    unittest.main()
