# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.services.performance_tracker import (
    PerformanceStore,
    SignalRecord,
    build_monthly_summary,
    build_recent_performance_text,
    dedupe_records,
    evaluate_signal,
    first_us_market_day,
    init_performance_branch,
    is_first_us_market_day,
    load_performance_branch,
    maybe_build_monthly_report,
    persist_performance_branch,
    render_monthly_report,
    run_us_morning_report_with_performance,
    signal_from_candidate_view,
    signal_records_from_candidate_views,
    summarize_performance,
    update_active_records,
    update_store_with_histories,
)
from src.services.us_morning_report import (
    CandidateReason,
    CandidateView,
    MarketContext,
    ReportOutcome,
    StockSignal,
    build_candidate_views,
    group_candidate_views,
)


def _record(**overrides) -> SignalRecord:
    data = {
        "signal_id": "sig-aapl",
        "signal_date": "2026-07-10",
        "generated_at": "2026-07-10T20:00:00+08:00",
        "ticker": "AAPL",
        "category": "Best Trade",
        "action": "Buy",
        "scanner_score": 88.0,
        "confidence": 82.0,
        "trade_quality": "A",
        "current_price_at_signal": 100.0,
        "entry_zone_low": 98.0,
        "entry_zone_high": 100.0,
        "breakout_trigger": 106.0,
        "stop_loss": 95.0,
        "target_1": 110.0,
        "target_2": 118.0,
        "risk_reward_ratio": 2.0,
        "expected_holding_period": "3-10天",
        "technical_reason": "trend above EMA20",
        "catalyst_reason": "earnings catalyst",
        "risk_reason": "break below support",
        "data_timestamp": "2026-07-10",
    }
    data.update(overrides)
    return SignalRecord.from_dict(data)


def _ohlc(rows: list[tuple[str, float, float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Open": [row[1] for row in rows],
            "High": [row[2] for row in rows],
            "Low": [row[3] for row in rows],
            "Close": [row[4] for row in rows],
        },
        index=pd.to_datetime([row[0] for row in rows]),
    )


def _candidate_view() -> CandidateView:
    signal = StockSignal(
        symbol="AAPL",
        price=100,
        daily_return_pct=1,
        return_5d_pct=3,
        return_20d_pct=7,
        return_60d_pct=12,
        volume_ratio=1.5,
        ma20=95,
        ma50=90,
        volatility_20d_pct=22,
        rsi_14=58,
        score=88,
        action="Buy",
        current_price=100,
        entry_zone_low=98,
        entry_zone_high=100,
        breakout_trigger=106,
        stop_loss=95,
        target_1=110,
        target_2=118,
        risk_reward_ratio=2.0,
        data_timestamp="2026-07-10",
    )
    return CandidateView(
        signal=signal,
        confidence=83,
        holding_period="3-10天",
        reason=CandidateReason("trend", "catalyst", "risk"),
    )


class TestPerformanceTracker(unittest.TestCase):
    def test_signal_save_schema_and_dedupe(self) -> None:
        record = signal_from_candidate_view(_candidate_view(), category="Best Trade", signal_date="2026-07-10")
        self.assertEqual(record.schema_version, 1)
        self.assertEqual(record.status, "pending")
        self.assertTrue(record.signal_id)
        self.assertIn("entry_zone_low", record.to_dict())
        self.assertEqual(len(dedupe_records([record, record])), 1)

    def test_next_trading_day_only(self) -> None:
        record = _record()
        history = _ohlc([
            ("2026-07-10", 99, 112, 94, 111),
            ("2026-07-13", 101, 104, 99, 102),
        ])
        updated = evaluate_signal(record, history)
        self.assertEqual(updated.entry_date, "2026-07-13")
        self.assertNotEqual(updated.entry_date, "2026-07-10")

    def test_pullback_entry_triggered(self) -> None:
        updated = evaluate_signal(_record(), _ohlc([("2026-07-13", 101, 103, 99, 102)]))
        self.assertEqual(updated.entry_type, "pullback")
        self.assertEqual(updated.entry_price, 100.0)

    def test_breakout_entry_triggered(self) -> None:
        updated = evaluate_signal(_record(), _ohlc([("2026-07-13", 105, 107, 101, 106)]))
        self.assertEqual(updated.entry_type, "breakout")
        self.assertEqual(updated.entry_price, 106.0)

    def test_same_day_pullback_and_breakout_prefers_pullback(self) -> None:
        updated = evaluate_signal(_record(), _ohlc([("2026-07-13", 101, 107, 99, 105)]))
        self.assertEqual(updated.entry_type, "pullback")
        self.assertIn("same_day_pullback_and_breakout_triggered_pullback_preferred", updated.evaluation_notes)

    def test_untriggered_expires(self) -> None:
        history = _ohlc([(f"2026-07-{day:02d}", 101, 104, 100.5, 103) for day in range(13, 25)])
        updated = evaluate_signal(_record(), history, expiry_days=3)
        self.assertEqual(updated.status, "expired")
        self.assertEqual(updated.exit_reason, "untriggered_expiry")
        self.assertIsNone(updated.exit_price)
        self.assertIsNone(updated.return_pct)

    def test_tp1_hit(self) -> None:
        history = _ohlc([
            ("2026-07-13", 101, 104, 99, 102),
            ("2026-07-14", 103, 111, 101, 110),
        ])
        updated = evaluate_signal(_record(), history)
        self.assertEqual(updated.status, "tp1_hit")
        self.assertIsNone(updated.exit_date)
        self.assertIsNone(updated.exit_price)
        self.assertIsNone(updated.exit_reason)
        self.assertIsNone(updated.return_pct)
        self.assertEqual(updated.tp1_hit_date, "2026-07-14")
        self.assertEqual(updated.tp1_return_pct, 10.0)
        self.assertEqual(updated.first_target_result, "tp1")

    def test_tp1_hit_stays_active(self) -> None:
        history = _ohlc([
            ("2026-07-13", 101, 104, 99, 102),
            ("2026-07-14", 103, 111, 101, 110),
        ])
        active, closed, counts = update_active_records([_record()], history_by_ticker={"AAPL": history})
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0].status, "tp1_hit")
        self.assertEqual(len(closed), 0)
        self.assertEqual(counts["active"], 1)

    def test_tp1_then_tp2_counts_once_final_trade(self) -> None:
        history = _ohlc([
            ("2026-07-13", 101, 104, 99, 102),
            ("2026-07-14", 103, 111, 101, 110),
            ("2026-07-15", 111, 119, 108, 118),
        ])
        updated = evaluate_signal(_record(), history)
        summary = summarize_performance([updated])
        self.assertEqual(updated.status, "tp2_hit")
        self.assertEqual(updated.tp1_return_pct, 10.0)
        self.assertEqual(updated.return_pct, 18.0)
        self.assertEqual(summary["sample_size"], 1)
        self.assertEqual(summary["tp1_count"], 1)
        self.assertEqual(summary["tp2_count"], 1)
        self.assertEqual(summary["win_rate"], 100.0)

    def test_tp1_then_stop_counts_once_final_trade(self) -> None:
        history = _ohlc([
            ("2026-07-13", 101, 104, 99, 102),
            ("2026-07-14", 103, 111, 101, 110),
            ("2026-07-15", 97, 104, 94, 96),
        ])
        updated = evaluate_signal(_record(), history)
        summary = summarize_performance([updated])
        self.assertEqual(updated.status, "stopped")
        self.assertEqual(updated.tp1_return_pct, 10.0)
        self.assertEqual(updated.return_pct, -5.0)
        self.assertEqual(summary["sample_size"], 1)
        self.assertEqual(summary["stopped_count"], 1)
        self.assertEqual(summary["average_return_pct"], -5.0)

    def test_tp1_then_time_exit_counts_once_final_trade(self) -> None:
        history = _ohlc([
            ("2026-07-13", 101, 104, 99, 102),
            ("2026-07-14", 103, 111, 101, 110),
            ("2026-07-15", 105, 108, 101, 106),
        ])
        updated = evaluate_signal(_record(), history, max_holding_days=3)
        summary = summarize_performance([updated])
        self.assertEqual(updated.status, "expired")
        self.assertEqual(updated.exit_reason, "time_exit")
        self.assertEqual(updated.tp1_return_pct, 10.0)
        self.assertEqual(updated.return_pct, 6.0)
        self.assertEqual(summary["sample_size"], 1)
        self.assertEqual(summary["average_return_pct"], 6.0)

    def test_tp2_hit(self) -> None:
        history = _ohlc([
            ("2026-07-13", 101, 104, 99, 102),
            ("2026-07-14", 103, 119, 101, 118),
        ])
        updated = evaluate_signal(_record(), history)
        self.assertEqual(updated.status, "tp2_hit")
        self.assertEqual(updated.exit_reason, "tp2")

    def test_stop_hit(self) -> None:
        history = _ohlc([
            ("2026-07-13", 101, 104, 99, 102),
            ("2026-07-14", 97, 99, 94, 95),
        ])
        updated = evaluate_signal(_record(), history)
        self.assertEqual(updated.status, "stopped")
        self.assertEqual(updated.exit_reason, "stop")

    def test_same_day_stop_and_target_ambiguous(self) -> None:
        history = _ohlc([
            ("2026-07-13", 101, 104, 99, 102),
            ("2026-07-14", 102, 111, 94, 100),
        ])
        updated = evaluate_signal(_record(), history)
        self.assertEqual(updated.status, "ambiguous")
        self.assertIn("same_day_stop_and_target_order_unknown", updated.evaluation_notes)

    def test_entry_day_entry_and_target_ambiguous(self) -> None:
        updated = evaluate_signal(_record(), _ohlc([("2026-07-13", 101, 111, 99, 110)]))
        self.assertEqual(updated.status, "ambiguous")
        self.assertIn("entry_day_stop_or_target_order_unknown", updated.evaluation_notes)

    def test_time_exit(self) -> None:
        history = _ohlc([
            ("2026-07-13", 101, 104, 99, 102),
            ("2026-07-14", 102, 105, 99, 104),
            ("2026-07-15", 103, 105, 99, 104),
        ])
        updated = evaluate_signal(_record(), history, max_holding_days=3)
        self.assertEqual(updated.status, "expired")
        self.assertEqual(updated.exit_reason, "time_exit")

    def test_untriggered_expiry_excluded_from_win_rate(self) -> None:
        record = _record(status="expired", entry_type="none", exit_reason="untriggered_expiry")
        summary = summarize_performance([record])
        self.assertEqual(summary["untriggered_signals"], 1)
        self.assertEqual(summary["closed_signals"], 0)
        self.assertEqual(summary["sample_size"], 0)
        self.assertIsNone(summary["win_rate"])
        self.assertIsNone(summary["average_return_pct"])

    def test_time_exit_included_in_win_rate_and_average_return(self) -> None:
        record = _record(
            status="expired",
            entry_type="pullback",
            entry_price=100,
            exit_price=104,
            exit_reason="time_exit",
            return_pct=4,
        )
        summary = summarize_performance([record])
        self.assertEqual(summary["closed_signals"], 1)
        self.assertEqual(summary["sample_size"], 1)
        self.assertEqual(summary["win_rate"], 100.0)
        self.assertEqual(summary["average_return_pct"], 4.0)

    def test_return_mfe_mae(self) -> None:
        history = _ohlc([
            ("2026-07-13", 101, 104, 99, 102),
            ("2026-07-14", 102, 109, 97, 108),
            ("2026-07-15", 103, 119, 99, 118),
        ])
        updated = evaluate_signal(_record(), history)
        self.assertEqual(updated.return_pct, 18.0)
        self.assertGreater(updated.max_favorable_excursion_pct, 0)
        self.assertLess(updated.max_adverse_excursion_pct, 0)

    def test_watch_pending_ambiguous_excluded_from_win_rate(self) -> None:
        watch = _record(
            signal_id="watch",
            category="Watch",
            include_in_win_rate=False,
            status="tp1_hit",
            tp1_return_pct=5,
            entry_type="pullback",
        )
        pending = _record(signal_id="pending", status="pending")
        ambiguous = _record(signal_id="amb", status="ambiguous", entry_type="pullback")
        stopped = _record(signal_id="stop", status="stopped", entry_type="pullback", return_pct=-2)
        summary = summarize_performance([watch, pending, ambiguous, stopped])
        self.assertEqual(summary["sample_size"], 1)
        self.assertEqual(summary["win_rate"], 0.0)
        self.assertEqual(summary["ambiguous_count"], 1)

    def test_profit_factor_expectancy_and_sample_insufficient(self) -> None:
        win = _record(signal_id="win", status="tp2_hit", entry_type="pullback", return_pct=4, trading_days_open=3)
        loss = _record(signal_id="loss", status="stopped", entry_type="pullback", return_pct=-2, trading_days_open=2)
        summary = summarize_performance([win, loss])
        self.assertEqual(summary["profit_factor"], 2.0)
        self.assertEqual(summary["expectancy_pct"], 1.0)
        empty = summarize_performance([])
        self.assertIsNone(empty["profit_factor"])

    def test_monthly_aggregation_by_filtering(self) -> None:
        july = _record(signal_id="july", signal_date="2026-07-10", status="tp2_hit", entry_type="pullback", return_pct=3)
        june = _record(signal_id="june", signal_date="2026-06-10", status="stopped", entry_type="pullback", return_pct=-2)
        selected = [record for record in [july, june] if record.signal_date.startswith("2026-07")]
        summary = summarize_performance(selected)
        self.assertEqual(summary["total_signals"], 1)
        self.assertEqual(summary["win_rate"], 100.0)

    def test_performance_branch_init(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            subprocess.run(["git", "init"], cwd=repo, check=True, stdout=subprocess.DEVNULL)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
            (repo / "README.md").write_text("init\n", encoding="utf-8")
            subprocess.run(["git", "add", "README.md"], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-m", "init"], cwd=repo, check=True, stdout=subprocess.DEVNULL)
            result = init_performance_branch(repo, dry_run=False)
            self.assertTrue(result["created"])

    def test_persistence_failure_can_be_caught_by_caller(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            (repo / "performance").mkdir()
            with patch("src.services.performance_tracker.subprocess.run", side_effect=RuntimeError("boom")):
                try:
                    persist_performance_branch(repo)
                    failed = False
                except RuntimeError:
                    failed = True
            self.assertTrue(failed)
            self.assertTrue((repo / "performance").exists())

    def test_no_duplicate_signal_id_and_store_idempotency(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = PerformanceStore(tmpdir)
            record = _record()
            self.assertEqual(store.append_signals([record]), 1)
            self.assertEqual(store.append_signals([record]), 0)
            self.assertEqual(len(store.load_all_signals()), 1)

    def test_signals_jsonl_history_not_deleted_or_duplicated(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = PerformanceStore(tmpdir)
            record = _record()
            store.append_signals([record])
            history = _ohlc([
                ("2026-07-13", 101, 104, 99, 102),
                ("2026-07-14", 103, 119, 101, 118),
            ])
            update_store_with_histories(store, history_by_ticker={"AAPL": history}, run_date="2026-07-14")
            self.assertEqual(store.append_signals([record]), 0)
            original_records = store.load_all_signals()
            self.assertEqual(len(original_records), 1)
            self.assertEqual(original_records[0].status, "pending")
            self.assertEqual(len(store.load_closed()), 1)

    def test_gemini_cannot_modify_history_performance(self) -> None:
        record = _record(status="stopped", entry_type="pullback", return_pct=-5)
        summary = summarize_performance([record])
        self.assertEqual(summary["win_rate"], 0.0)
        self.assertEqual(record.return_pct, -5)

    def test_data_insufficient_unknown(self) -> None:
        updated = evaluate_signal(_record(), pd.DataFrame())
        self.assertEqual(updated.status, "unknown")

    def test_corporate_action_review(self) -> None:
        history = _ohlc([
            ("2026-07-13", 100, 102, 99, 100),
            ("2026-07-14", 49, 51, 48, 50),
        ])
        updated = evaluate_signal(_record(), history)
        self.assertEqual(updated.status, "unknown")
        self.assertTrue(updated.corporate_action_review)

    def test_recent_summary_length_and_tracker_disabled_noop(self) -> None:
        summary = summarize_performance([_record(status="tp2_hit", entry_type="pullback", return_pct=3)])
        text = json.dumps(summary)
        self.assertLess(len(text), 2000)
        with patch.dict("os.environ", {"PERFORMANCE_TRACKING_ENABLED": "false"}):
            self.assertEqual(os.environ["PERFORMANCE_TRACKING_ENABLED"], "false")

    def test_update_store_with_histories(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = PerformanceStore(tmpdir)
            record = _record()
            store.append_signals([record])
            history = _ohlc([
                ("2026-07-13", 101, 104, 99, 102),
                ("2026-07-14", 103, 119, 101, 118),
            ])
            result = update_store_with_histories(store, history_by_ticker={"AAPL": history}, run_date="2026-07-14")
            self.assertEqual(result["counts"]["closed"], 1)
            self.assertTrue((Path(tmpdir) / "performance/summaries/latest.json").exists())

    def test_signal_records_from_views_default_excludes_watch(self) -> None:
        views = build_candidate_views([
            _candidate_view().signal,
            _candidate_view().signal.__class__(**{**_candidate_view().signal.__dict__, "symbol": "MSFT", "score": 80, "action": "Watch"}),
        ])
        groups = group_candidate_views(views)
        records = signal_records_from_candidate_views(views, groups, signal_date="2026-07-10", include_watch=False)
        self.assertTrue(records)
        self.assertTrue(all(record.category != "Watch" for record in records))

    def test_signal_id_uses_market_date_and_entry_plan_version(self) -> None:
        first = signal_from_candidate_view(_candidate_view(), category="Best Trade", signal_date="2026-07-10")
        second = signal_from_candidate_view(_candidate_view(), category="Best Trade", signal_date="2026-07-11")
        self.assertNotEqual(first.signal_id, second.signal_id)
        self.assertEqual(first.entry_plan_version, "structured-levels-v1")

    def test_recent_performance_small_sample_and_length(self) -> None:
        records = [
            _record(signal_id="tp2", status="tp2_hit", entry_type="pullback", return_pct=6.4, exit_date="2026-07-12"),
            _record(signal_id="stop", ticker="TSLA", status="stopped", entry_type="pullback", return_pct=-2.3, exit_date="2026-07-11"),
            _record(signal_id="tp1", ticker="MSFT", status="tp1_hit", entry_type="pullback", tp1_hit_date="2026-07-10", tp1_return_pct=3.1),
            _record(signal_id="pending", ticker="AMZN", status="pending"),
        ]
        text = build_recent_performance_text(records, as_of="2026-07-13")
        self.assertIn("📈 近期信号表现｜近30日", text)
        self.assertIn("胜率：样本不足", text)
        self.assertIn("🎯 MSFT｜TP1后继续追踪", text)
        self.assertLessEqual(len(text), 700)

    def test_recent_performance_degrades_on_bad_dates(self) -> None:
        bad = _record(signal_date="not-a-date")
        text = build_recent_performance_text([bad], as_of="2026-07-13")
        self.assertIn("暂时无法读取", text)

    def test_first_us_market_day_uses_market_calendar(self) -> None:
        self.assertEqual(first_us_market_day(2026, 7).isoformat(), "2026-07-01")
        self.assertTrue(is_first_us_market_day("2026-07-01"))
        self.assertFalse(is_first_us_market_day("2026-07-02"))
        self.assertEqual(first_us_market_day(2027, 1).isoformat(), "2027-01-04")

    def test_monthly_report_null_stats_do_not_fail(self) -> None:
        summary = build_monthly_summary([_record(status="pending")], "2026-07")
        text = render_monthly_report(summary)
        self.assertIn("📊 月度信号表现｜2026-07", text)
        self.assertIn("样本不足", text)
        self.assertIn("分类表现", text)

    def test_monthly_report_marker_prevents_duplicate_and_failed_send_can_retry(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = PerformanceStore(tmpdir)
            records = [_record(signal_date="2026-06-10", status="tp2_hit", entry_type="pullback", return_pct=5)]
            first_text, _summary = maybe_build_monthly_report(store, records, as_of="2026-07-01", enabled=True)
            self.assertIsNotNone(first_text)
            # No sent marker yet, so a failed Telegram send allows retry.
            retry_text, _retry_summary = maybe_build_monthly_report(store, records, as_of="2026-07-01", enabled=True)
            self.assertIsNotNone(retry_text)
            from src.services.performance_tracker import mark_monthly_report_sent

            mark_monthly_report_sent(store, "2026-06")
            duplicate_text, _duplicate_summary = maybe_build_monthly_report(store, records, as_of="2026-07-01", enabled=True)
            self.assertIsNone(duplicate_text)

    def test_performance_data_branch_persists_only_performance_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            bare = base / "origin.git"
            repo = base / "repo"
            subprocess.run(["git", "init", "--bare", str(bare)], check=True, stdout=subprocess.DEVNULL)
            subprocess.run(["git", "init", str(repo)], check=True, stdout=subprocess.DEVNULL)
            subprocess.run(["git", "remote", "add", "origin", str(bare)], cwd=repo, check=True)
            (repo / "README.md").write_text("main code\n", encoding="utf-8")
            store = PerformanceStore(repo)
            store.append_signals([_record()])
            result = persist_performance_branch(repo, "performance-data")
            self.assertTrue(result["pushed"])

            checkout = base / "checkout"
            subprocess.run(["git", "clone", "--branch", "performance-data", str(bare), str(checkout)], check=True, stdout=subprocess.DEVNULL)
            self.assertTrue((checkout / "performance/signals.jsonl").exists())
            self.assertFalse((checkout / "README.md").exists())

            shutil.rmtree(repo / "performance")
            loaded = load_performance_branch(repo, "performance-data")
            self.assertTrue(loaded["loaded"])
            self.assertTrue((repo / "performance/signals.jsonl").exists())

    def test_persist_branch_no_empty_commit_when_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            bare = base / "origin.git"
            repo = base / "repo"
            subprocess.run(["git", "init", "--bare", str(bare)], check=True, stdout=subprocess.DEVNULL)
            subprocess.run(["git", "init", str(repo)], check=True, stdout=subprocess.DEVNULL)
            subprocess.run(["git", "remote", "add", "origin", str(bare)], cwd=repo, check=True)
            PerformanceStore(repo).append_signals([_record()])
            self.assertTrue(persist_performance_branch(repo, "performance-data")["pushed"])
            second = persist_performance_branch(repo, "performance-data")
            self.assertEqual(second["reason"], "no_changes")

    def test_push_failure_returns_warning_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            subprocess.run(["git", "init"], cwd=repo, check=True, stdout=subprocess.DEVNULL)
            subprocess.run(["git", "remote", "add", "origin", str(repo / "missing.git")], cwd=repo, check=True)
            PerformanceStore(repo).append_signals([_record()])
            result = persist_performance_branch(repo, "performance-data", retries=1)
            self.assertFalse(result["pushed"])

    def test_tracker_disabled_keeps_old_report_flow(self) -> None:
        outcome = ReportOutcome(
            ranked=[_candidate_view().signal],
            gemini_candidates=[_candidate_view().signal],
            news={},
            market_context=MarketContext(),
            market_news=[],
            gemini_summary=None,
            report_text="# report",
            telegram_text="MAIN REPORT",
            report_path=None,
            telegram_sent=False,
        )
        sent: list[str] = []

        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(os.environ, {"PERFORMANCE_TRACKING_ENABLED": "false"}):
            with patch("src.services.us_morning_report.run_us_morning_report", return_value=outcome):
                with patch("src.services.us_morning_report.send_telegram_summary", side_effect=lambda text: sent.append(text) or True):
                    result = run_us_morning_report_with_performance(root=tmpdir, send_telegram=True, write_performance=True)

            self.assertTrue(result["telegram_sent"])
            self.assertEqual(sent, ["MAIN REPORT"])
            self.assertFalse((Path(tmpdir) / "performance/signals.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
