# -*- coding: utf-8 -*-
"""Deterministic performance tracking for US Morning Scanner signals."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

import pandas as pd

from src.services.us_morning_report import CandidateView

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
ENTRY_PLAN_VERSION = "structured-levels-v1"
VALID_STATUS = {"pending", "triggered", "tp1_hit", "tp2_hit", "stopped", "expired", "ambiguous", "unknown"}
VALID_ENTRY_TYPE = {"pullback", "breakout", "none"}
FORMAL_CATEGORIES = {"Best Trade", "Second Choice", "Buy"}


@dataclass
class SignalRecord:
    signal_id: str
    signal_date: str
    generated_at: str
    ticker: str
    category: str
    action: str
    scanner_score: float
    confidence: float
    trade_quality: str
    current_price_at_signal: float
    entry_zone_low: float | None
    entry_zone_high: float | None
    breakout_trigger: float | None
    stop_loss: float | None
    target_1: float | None
    target_2: float | None
    risk_reward_ratio: float | None
    expected_holding_period: str
    technical_reason: str
    catalyst_reason: str
    risk_reason: str
    data_timestamp: str
    status: str = "pending"
    entry_type: str = "none"
    entry_date: str | None = None
    entry_price: float | None = None
    exit_date: str | None = None
    exit_price: float | None = None
    exit_reason: str | None = None
    highest_price_after_entry: float | None = None
    lowest_price_after_entry: float | None = None
    max_favorable_excursion_pct: float | None = None
    max_adverse_excursion_pct: float | None = None
    return_pct: float | None = None
    trading_days_open: int = 0
    tp1_hit_date: str | None = None
    tp1_return_pct: float | None = None
    tp2_hit_date: str | None = None
    stop_hit_date: str | None = None
    last_evaluated_date: str | None = None
    evaluation_notes: list[str] = field(default_factory=list)
    source_report_path: str | None = None
    schema_version: int = SCHEMA_VERSION
    entry_plan_version: str = ENTRY_PLAN_VERSION
    first_target_result: str | None = None
    final_result: str | None = None
    time_exit_price: float | None = None
    corporate_action_review: bool = False
    include_in_win_rate: bool = True

    def __post_init__(self) -> None:
        if self.status not in VALID_STATUS:
            raise ValueError(f"invalid status: {self.status}")
        if self.entry_type not in VALID_ENTRY_TYPE:
            raise ValueError(f"invalid entry_type: {self.entry_type}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SignalRecord":
        values = dict(data)
        values.setdefault("evaluation_notes", [])
        values.setdefault("schema_version", SCHEMA_VERSION)
        values.setdefault("entry_plan_version", ENTRY_PLAN_VERSION)
        values.setdefault("first_target_result", None)
        values.setdefault("final_result", None)
        values.setdefault("time_exit_price", None)
        values.setdefault("tp1_return_pct", None)
        values.setdefault("corporate_action_review", False)
        values.setdefault("include_in_win_rate", values.get("category") in FORMAL_CATEGORIES)
        return cls(**values)


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_signal_id(signal_date: str, ticker: str, category: str, action: str) -> str:
    raw = f"{signal_date}|{ticker.upper()}|{category}|{action}|{ENTRY_PLAN_VERSION}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _trade_quality(confidence: float) -> str:
    if confidence >= 90:
        return "A+"
    if confidence >= 80:
        return "A"
    if confidence >= 65:
        return "B"
    return "C"


def signal_from_candidate_view(
    view: CandidateView,
    *,
    category: str,
    signal_date: str,
    generated_at: str | None = None,
    source_report_path: str | None = None,
    include_watch: bool = False,
) -> SignalRecord:
    signal = view.signal
    include = category in FORMAL_CATEGORIES or (category == "Watch" and include_watch)
    signal_id = _stable_signal_id(signal_date, signal.symbol, category, signal.action)
    return SignalRecord(
        signal_id=signal_id,
        signal_date=signal_date,
        generated_at=generated_at or _iso_now(),
        ticker=signal.symbol,
        category=category,
        action=signal.action,
        scanner_score=round(signal.score, 2),
        confidence=round(view.confidence, 2),
        trade_quality=_trade_quality(view.confidence),
        current_price_at_signal=round(signal.current_price, 4),
        entry_zone_low=signal.entry_zone_low,
        entry_zone_high=signal.entry_zone_high,
        breakout_trigger=signal.breakout_trigger,
        stop_loss=signal.stop_loss,
        target_1=signal.target_1,
        target_2=signal.target_2,
        risk_reward_ratio=signal.risk_reward_ratio,
        expected_holding_period=view.holding_period,
        technical_reason=view.reason.technical,
        catalyst_reason=view.reason.catalyst,
        risk_reason=view.reason.risk,
        data_timestamp=signal.data_timestamp,
        source_report_path=source_report_path,
        include_in_win_rate=include,
    )


def signal_records_from_groups(
    groups: Mapping[str, Sequence[CandidateView]],
    *,
    signal_date: str,
    generated_at: str | None = None,
    source_report_path: str | None = None,
    include_watch: bool = False,
) -> list[SignalRecord]:
    records: list[SignalRecord] = []
    for category, key in (("Best Trade", "best"), ("Second Choice", "second"), ("Watch", "watch")):
        if category == "Watch" and not include_watch:
            continue
        for view in groups.get(key, []):
            records.append(
                signal_from_candidate_view(
                    view,
                    category=category,
                    signal_date=signal_date,
                    generated_at=generated_at,
                    source_report_path=source_report_path,
                    include_watch=include_watch,
                )
            )
    return dedupe_records(records)


def signal_records_from_candidate_views(
    views: Sequence[CandidateView],
    groups: Mapping[str, Sequence[CandidateView]],
    *,
    signal_date: str,
    generated_at: str | None = None,
    source_report_path: str | None = None,
    include_watch: bool = False,
) -> list[SignalRecord]:
    """Create records after the morning report is finalized."""
    records: list[SignalRecord] = []
    used_symbols: set[str] = set()
    for category, key in (("Best Trade", "best"), ("Second Choice", "second")):
        for view in groups.get(key, []):
            records.append(
                signal_from_candidate_view(
                    view,
                    category=category,
                    signal_date=signal_date,
                    generated_at=generated_at,
                    source_report_path=source_report_path,
                    include_watch=include_watch,
                )
            )
            used_symbols.add(view.signal.symbol)

    for view in views:
        if view.signal.symbol in used_symbols or view.signal.action != "Buy":
            continue
        records.append(
            signal_from_candidate_view(
                view,
                category="Buy",
                signal_date=signal_date,
                generated_at=generated_at,
                source_report_path=source_report_path,
                include_watch=include_watch,
            )
        )
        used_symbols.add(view.signal.symbol)

    if include_watch:
        for view in groups.get("watch", []):
            if view.signal.symbol in used_symbols:
                continue
            records.append(
                signal_from_candidate_view(
                    view,
                    category="Watch",
                    signal_date=signal_date,
                    generated_at=generated_at,
                    source_report_path=source_report_path,
                    include_watch=True,
                )
            )
            used_symbols.add(view.signal.symbol)
    return dedupe_records(records)


def dedupe_records(records: Sequence[SignalRecord]) -> list[SignalRecord]:
    seen: set[str] = set()
    unique: list[SignalRecord] = []
    for record in records:
        if record.signal_id in seen:
            continue
        seen.add(record.signal_id)
        unique.append(record)
    return unique


def _parse_date(value: str) -> date:
    return date.fromisoformat(value[:10])


def _market_date(index_value: Any) -> str:
    return pd.Timestamp(index_value).date().isoformat()


def _round_pct(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 4)


def _return_pct(entry_price: float | None, exit_price: float | None) -> float | None:
    if not entry_price or not exit_price or entry_price <= 0:
        return None
    return round((exit_price / entry_price - 1.0) * 100.0, 4)


def _has_required_levels(record: SignalRecord) -> bool:
    values = [
        record.entry_zone_low,
        record.entry_zone_high,
        record.breakout_trigger,
        record.stop_loss,
        record.target_1,
        record.target_2,
    ]
    return all(value is not None and value > 0 for value in values)


def _detect_corporate_action(history: pd.DataFrame) -> bool:
    if history is None or history.empty or "Close" not in history:
        return False
    close = pd.to_numeric(history["Close"], errors="coerce").dropna()
    if len(close) < 2:
        return False
    ratios = close / close.shift(1)
    ratios = ratios.dropna()
    return bool(((ratios > 1.8) | (ratios < 0.55)).any())


def _prepare_history_after_signal(record: SignalRecord, history: pd.DataFrame) -> pd.DataFrame:
    if history is None or history.empty:
        return pd.DataFrame()
    required = {"Open", "High", "Low", "Close"}
    if not required.issubset(history.columns):
        return pd.DataFrame()
    rows = history.copy()
    rows = rows.sort_index()
    signal_day = _parse_date(record.signal_date)
    mask = [pd.Timestamp(index).date() > signal_day for index in rows.index]
    return rows.loc[mask]


def evaluate_signal(
    record: SignalRecord,
    history: pd.DataFrame,
    *,
    expiry_days: int = 10,
    max_holding_days: int = 10,
    conservative_ambiguous: bool = True,
) -> SignalRecord:
    updated = SignalRecord.from_dict(record.to_dict())
    notes = list(updated.evaluation_notes)

    if not _has_required_levels(updated):
        updated.status = "unknown"
        updated.last_evaluated_date = updated.last_evaluated_date or updated.signal_date
        notes.append("missing_structured_levels")
        updated.evaluation_notes = _dedupe_notes(notes)
        return updated

    if _detect_corporate_action(history):
        updated.status = "unknown"
        updated.corporate_action_review = True
        notes.append("corporate_action_review")
        updated.evaluation_notes = _dedupe_notes(notes)
        return updated

    rows = _prepare_history_after_signal(updated, history)
    if rows.empty:
        updated.status = "unknown"
        notes.append("insufficient_ohlc_data")
        updated.evaluation_notes = _dedupe_notes(notes)
        return updated

    for day_index, (idx, row) in enumerate(rows.iterrows(), 1):
        market_day = _market_date(idx)
        day_open = float(row["Open"])
        day_high = float(row["High"])
        day_low = float(row["Low"])
        day_close = float(row["Close"])
        updated.last_evaluated_date = market_day

        if updated.entry_type == "none":
            if day_index > expiry_days:
                updated.status = "expired"
                updated.exit_date = market_day
                updated.exit_price = None
                updated.time_exit_price = None
                updated.exit_reason = "untriggered_expiry"
                notes.append("entry_not_triggered_before_expiry")
                break

            pullback = day_low <= updated.entry_zone_high and day_high >= updated.entry_zone_low
            breakout = day_high >= updated.breakout_trigger
            if not pullback and not breakout:
                continue

            if pullback and breakout:
                notes.append("same_day_pullback_and_breakout_triggered_pullback_preferred")
            if pullback:
                updated.entry_type = "pullback"
                updated.entry_price = (
                    max(day_open, updated.entry_zone_low)
                    if day_open < updated.entry_zone_high
                    else updated.entry_zone_high
                )
            else:
                updated.entry_type = "breakout"
                updated.entry_price = updated.breakout_trigger
            updated.entry_date = market_day
            updated.status = "triggered"
            updated.highest_price_after_entry = day_high
            updated.lowest_price_after_entry = day_low
            updated.trading_days_open = 1

            entry_day_stop = day_low <= updated.stop_loss
            entry_day_target = day_high >= updated.target_1 or day_high >= updated.target_2
            if conservative_ambiguous and (entry_day_stop or entry_day_target):
                updated.status = "ambiguous"
                updated.exit_date = market_day
                updated.exit_reason = "entry_day_order_unknown"
                if entry_day_stop:
                    updated.stop_hit_date = market_day
                if day_high >= updated.target_1:
                    updated.tp1_hit_date = market_day
                    updated.tp1_return_pct = _return_pct(updated.entry_price, updated.target_1)
                    updated.first_target_result = "tp1"
                if day_high >= updated.target_2:
                    updated.tp2_hit_date = market_day
                    updated.first_target_result = updated.first_target_result or "tp2"
                notes.append("entry_day_stop_or_target_order_unknown")
                break
            continue

        updated.trading_days_open += 1
        updated.highest_price_after_entry = max(updated.highest_price_after_entry or day_high, day_high)
        updated.lowest_price_after_entry = min(updated.lowest_price_after_entry or day_low, day_low)
        stop_hit = day_low <= updated.stop_loss
        tp1_hit = day_high >= updated.target_1
        tp2_hit = day_high >= updated.target_2

        if stop_hit and (tp1_hit or tp2_hit):
            updated.status = "ambiguous"
            updated.exit_date = market_day
            updated.exit_reason = "same_day_stop_and_target"
            updated.stop_hit_date = market_day
            if tp1_hit:
                updated.tp1_hit_date = updated.tp1_hit_date or market_day
                updated.tp1_return_pct = updated.tp1_return_pct or _return_pct(updated.entry_price, updated.target_1)
                updated.first_target_result = updated.first_target_result or "tp1"
            if tp2_hit:
                updated.tp2_hit_date = updated.tp2_hit_date or market_day
            notes.append("same_day_stop_and_target_order_unknown")
            break

        if tp2_hit:
            updated.status = "tp2_hit"
            updated.exit_date = market_day
            updated.exit_price = updated.target_2
            updated.exit_reason = "tp2"
            updated.tp2_hit_date = market_day
            updated.tp1_hit_date = updated.tp1_hit_date or market_day
            updated.tp1_return_pct = updated.tp1_return_pct or _return_pct(updated.entry_price, updated.target_1)
            updated.first_target_result = updated.first_target_result or "tp1"
            updated.final_result = "tp2"
            break
        if tp1_hit and updated.status != "tp1_hit":
            updated.status = "tp1_hit"
            updated.exit_date = None
            updated.exit_price = None
            updated.exit_reason = None
            updated.tp1_hit_date = market_day
            updated.tp1_return_pct = _return_pct(updated.entry_price, updated.target_1)
            updated.first_target_result = "tp1"
            # Continue tracking for TP2 until holding limit.
        if stop_hit:
            updated.status = "stopped"
            updated.exit_date = market_day
            updated.exit_price = updated.stop_loss
            updated.exit_reason = "stop"
            updated.stop_hit_date = market_day
            updated.final_result = "stop"
            break
        if updated.trading_days_open >= max_holding_days:
            updated.status = "expired"
            updated.exit_date = market_day
            updated.exit_price = day_close
            updated.time_exit_price = day_close
            updated.exit_reason = "time_exit"
            updated.final_result = "time_exit"
            notes.append("max_holding_days_time_exit")
            break

    if updated.entry_type == "none" and updated.status == "pending" and len(rows) >= expiry_days:
        last = rows.iloc[min(expiry_days, len(rows)) - 1]
        updated.status = "expired"
        updated.exit_date = updated.last_evaluated_date
        updated.exit_price = None
        updated.time_exit_price = None
        updated.exit_reason = "untriggered_expiry"
        notes.append("entry_not_triggered_before_expiry")

    _finalize_excursions(updated)
    updated.return_pct = _return_pct(updated.entry_price, updated.exit_price)
    updated.evaluation_notes = _dedupe_notes(notes)
    return updated


def _dedupe_notes(notes: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for note in notes:
        if note and note not in seen:
            seen.add(note)
            result.append(note)
    return result


def _finalize_excursions(record: SignalRecord) -> None:
    if not record.entry_price or record.entry_price <= 0:
        return
    if record.highest_price_after_entry is not None:
        record.max_favorable_excursion_pct = _round_pct(
            (record.highest_price_after_entry / record.entry_price - 1.0) * 100.0
        )
    if record.lowest_price_after_entry is not None:
        record.max_adverse_excursion_pct = _round_pct(
            (record.lowest_price_after_entry / record.entry_price - 1.0) * 100.0
        )


def load_json_records(path: Path) -> list[SignalRecord]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        return []
    return [SignalRecord.from_dict(item) for item in data]


def load_jsonl_records(path: Path) -> list[SignalRecord]:
    if not path.exists():
        return []
    records: list[SignalRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(SignalRecord.from_dict(json.loads(line)))
    return records


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(content)
        tmp_name = handle.name
    Path(tmp_name).replace(path)


def atomic_write_json(path: Path, data: Any) -> None:
    _atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


class PerformanceStore:
    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.performance_dir = self.root / "performance"
        self.signals_jsonl = self.performance_dir / "signals.jsonl"
        self.active_json = self.performance_dir / "active_signals.json"
        self.closed_json = self.performance_dir / "closed_signals.json"
        self.daily_dir = self.performance_dir / "daily"
        self.summaries_dir = self.performance_dir / "summaries"

    def ensure(self) -> None:
        self.daily_dir.mkdir(parents=True, exist_ok=True)
        self.summaries_dir.mkdir(parents=True, exist_ok=True)
        if not self.active_json.exists():
            atomic_write_json(self.active_json, [])
        if not self.closed_json.exists():
            atomic_write_json(self.closed_json, [])
        if not self.signals_jsonl.exists():
            self.signals_jsonl.parent.mkdir(parents=True, exist_ok=True)
            self.signals_jsonl.touch()

    def load_active(self) -> list[SignalRecord]:
        self.ensure()
        return load_json_records(self.active_json)

    def load_closed(self) -> list[SignalRecord]:
        self.ensure()
        return load_json_records(self.closed_json)

    def load_all_signals(self) -> list[SignalRecord]:
        self.ensure()
        return load_jsonl_records(self.signals_jsonl)

    def append_signals(self, records: Sequence[SignalRecord]) -> int:
        self.ensure()
        existing = {record.signal_id for record in self.load_all_signals()}
        new_records = [record for record in dedupe_records(records) if record.signal_id not in existing]
        if not new_records:
            return 0
        with self.signals_jsonl.open("a", encoding="utf-8") as handle:
            for record in new_records:
                handle.write(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
        active = {record.signal_id: record for record in self.load_active()}
        for record in new_records:
            if record.status in {"pending", "triggered", "tp1_hit"}:
                active[record.signal_id] = record
        atomic_write_json(self.active_json, [record.to_dict() for record in active.values()])
        return len(new_records)

    def save_state(self, active: Sequence[SignalRecord], closed: Sequence[SignalRecord], *, run_date: str | None = None) -> None:
        self.ensure()
        active_unique = dedupe_records(active)
        closed_unique = dedupe_records(closed)
        atomic_write_json(self.active_json, [record.to_dict() for record in active_unique])
        atomic_write_json(self.closed_json, [record.to_dict() for record in closed_unique])
        if run_date:
            atomic_write_json(
                self.daily_dir / f"{run_date}.json",
                {
                    "run_date": run_date,
                    "active_count": len(active_unique),
                    "closed_count": len(closed_unique),
                    "updated_at": _iso_now(),
                },
            )

    def save_summary(self, summary: Mapping[str, Any]) -> None:
        self.ensure()
        atomic_write_json(self.summaries_dir / "latest.json", dict(summary))
        _atomic_write_text(self.summaries_dir / "latest.md", render_summary_markdown(summary))


def fetch_yfinance_history(ticker: str, start: str | None = None, period: str = "3mo") -> pd.DataFrame:
    import yfinance as yf

    kwargs: dict[str, Any] = {"interval": "1d", "auto_adjust": False, "progress": False}
    if start:
        kwargs["start"] = start
    else:
        kwargs["period"] = period
    data = yf.download(ticker, **kwargs)
    return data.dropna(how="all") if isinstance(data, pd.DataFrame) else pd.DataFrame()


def update_active_records(
    records: Sequence[SignalRecord],
    *,
    history_by_ticker: Mapping[str, pd.DataFrame],
    expiry_days: int = 10,
    max_holding_days: int = 10,
) -> tuple[list[SignalRecord], list[SignalRecord], dict[str, int]]:
    active: list[SignalRecord] = []
    closed: list[SignalRecord] = []
    counts = {"processed": 0, "closed": 0, "active": 0, "errors": 0, "skipped": 0}
    for record in records:
        counts["processed"] += 1
        history = history_by_ticker.get(record.ticker)
        if history is None:
            counts["skipped"] += 1
            skipped = SignalRecord.from_dict(record.to_dict())
            skipped.evaluation_notes = _dedupe_notes([*skipped.evaluation_notes, "history_fetch_unavailable"])
            active.append(skipped)
            continue
        try:
            updated = evaluate_signal(
                record,
                history,
                expiry_days=expiry_days,
                max_holding_days=max_holding_days,
            )
        except Exception as exc:  # noqa: BLE001 - tracker must not break scanner.
            logger.warning("performance tracker failed for %s: %s", record.ticker, type(exc).__name__)
            failed = SignalRecord.from_dict(record.to_dict())
            failed.status = "unknown"
            failed.evaluation_notes = _dedupe_notes([*failed.evaluation_notes, f"evaluation_error:{type(exc).__name__}"])
            updated = failed
            counts["errors"] += 1
        if updated.status in {"pending", "triggered", "tp1_hit"}:
            active.append(updated)
            counts["active"] += 1
        else:
            closed.append(updated)
            counts["closed"] += 1
    return active, closed, counts


def _is_active_record(record: SignalRecord) -> bool:
    return record.status in {"pending", "triggered", "tp1_hit"}


def _is_officially_closed(record: SignalRecord) -> bool:
    if record.status in {"tp2_hit", "stopped", "ambiguous"}:
        return True
    return record.status == "expired" and record.exit_reason == "time_exit"


def _formal_closed(records: Sequence[SignalRecord]) -> list[SignalRecord]:
    return [
        record
        for record in records
        if record.include_in_win_rate
        and (
            record.status in {"tp2_hit", "stopped"}
            or (record.status == "expired" and record.exit_reason == "time_exit")
        )
        and record.entry_type != "none"
        and record.return_pct is not None
    ]


def summarize_performance(
    records: Sequence[SignalRecord],
    *,
    recent_window_days: int | None = None,
    as_of: str | None = None,
) -> dict[str, Any]:
    selected = list(records)
    if recent_window_days is not None and as_of:
        cutoff = _parse_date(as_of) - pd.Timedelta(days=recent_window_days)
        cutoff_date = pd.Timestamp(cutoff).date()
        selected = [record for record in selected if _parse_date(record.signal_date) >= cutoff_date]

    triggered = [record for record in selected if record.entry_type != "none"]
    untriggered = [record for record in selected if record.exit_reason == "untriggered_expiry"]
    open_records = [record for record in selected if _is_active_record(record)]
    closed = [record for record in selected if _is_officially_closed(record)]
    formal = _formal_closed(selected)
    wins = [record for record in formal if record.return_pct is not None and record.return_pct > 0]
    losses = [record for record in formal if record.return_pct is not None and record.return_pct < 0]
    returns = [record.return_pct for record in formal if record.return_pct is not None]
    planned_rr = [record.risk_reward_ratio for record in selected if record.risk_reward_ratio is not None]
    realized_rr = [
        abs(record.return_pct) / abs((record.stop_loss / record.entry_price - 1.0) * 100.0)
        for record in formal
        if record.return_pct is not None and record.entry_price and record.stop_loss and record.entry_price != record.stop_loss
    ]
    gross_profit = sum(record.return_pct for record in wins if record.return_pct is not None)
    gross_loss = abs(sum(record.return_pct for record in losses if record.return_pct is not None))
    win_rate = len(wins) / len(formal) * 100.0 if formal else None
    avg_win = sum(record.return_pct for record in wins) / len(wins) if wins else None
    avg_loss = sum(record.return_pct for record in losses) / len(losses) if losses else None
    loss_rate = 100.0 - win_rate if win_rate is not None else None
    expectancy = None
    if win_rate is not None and avg_win is not None and avg_loss is not None:
        expectancy = (win_rate / 100.0) * avg_win + (loss_rate / 100.0) * avg_loss

    return {
        "generated_at": _iso_now(),
        "total_signals": len(selected),
        "triggered_signals": len(triggered),
        "untriggered_signals": len(untriggered),
        "open_signals": len(open_records),
        "closed_signals": len(closed),
        "tp1_count": sum(1 for record in selected if record.tp1_hit_date is not None),
        "tp2_count": sum(1 for record in selected if record.status == "tp2_hit"),
        "stopped_count": sum(1 for record in selected if record.status == "stopped"),
        "expired_count": sum(1 for record in selected if record.status == "expired"),
        "time_exit_count": sum(1 for record in selected if record.status == "expired" and record.exit_reason == "time_exit"),
        "ambiguous_count": sum(1 for record in selected if record.status == "ambiguous"),
        "unknown_count": sum(1 for record in selected if record.status == "unknown"),
        "corporate_action_review_count": sum(1 for record in selected if record.corporate_action_review),
        "win_rate": round(win_rate, 2) if win_rate is not None else None,
        "average_return_pct": round(sum(returns) / len(returns), 4) if returns else None,
        "median_return_pct": round(median(returns), 4) if returns else None,
        "average_win_pct": round(avg_win, 4) if avg_win is not None else None,
        "average_loss_pct": round(avg_loss, 4) if avg_loss is not None else None,
        "profit_factor": round(gross_profit / gross_loss, 4) if gross_loss > 0 else None,
        "expectancy_pct": round(expectancy, 4) if expectancy is not None else None,
        "average_holding_days": round(
            sum(record.trading_days_open for record in formal) / len(formal), 2
        ) if formal else None,
        "average_planned_rr": round(sum(planned_rr) / len(planned_rr), 4) if planned_rr else None,
        "average_realized_rr": round(sum(realized_rr) / len(realized_rr), 4) if realized_rr else None,
        "max_drawdown_of_closed_signal_sequence": _max_drawdown(returns),
        "best_trade": _trade_summary(max(formal, key=lambda r: r.return_pct) if returns else None),
        "worst_trade": _trade_summary(min(formal, key=lambda r: r.return_pct) if returns else None),
        "sample_size": len(formal),
    }


def _trade_summary(record: SignalRecord | None) -> dict[str, Any] | None:
    if record is None:
        return None
    return {
        "ticker": record.ticker,
        "status": record.status,
        "return_pct": record.return_pct,
        "signal_date": record.signal_date,
        "exit_date": record.exit_date,
    }


def _fmt_pct(value: float | None, *, signed: bool = False) -> str:
    if value is None:
        return "暂无"
    sign = "+" if signed and value > 0 else ""
    return f"{sign}{value:.2f}%"


def _fmt_number(value: float | int | None, digits: int = 2) -> str:
    if value is None:
        return "暂无"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def _recent_record_line(record: SignalRecord) -> str:
    if record.status == "tp2_hit":
        return f"✅ {record.ticker}｜TP2｜{_fmt_pct(record.return_pct, signed=True)}"
    if record.status == "stopped":
        return f"❌ {record.ticker}｜止损｜{_fmt_pct(record.return_pct, signed=True)}"
    if record.status == "expired" and record.exit_reason == "time_exit":
        return f"➖ {record.ticker}｜时间退出｜{_fmt_pct(record.return_pct, signed=True)}"
    if record.status == "tp1_hit":
        return f"🎯 {record.ticker}｜TP1后继续追踪"
    if record.status in {"pending", "triggered"}:
        return f"⏳ {record.ticker}｜等待入场"
    if record.exit_reason == "untriggered_expiry":
        return f"⏳ {record.ticker}｜未触发到期"
    if record.status == "ambiguous":
        return f"❔ {record.ticker}｜模糊结果"
    return f"⏳ {record.ticker}｜{record.status}"


def build_recent_performance_text(
    records: Sequence[SignalRecord],
    *,
    recent_window_days: int = 30,
    as_of: str | None = None,
    max_chars: int = 700,
) -> str:
    try:
        summary = summarize_performance(records, recent_window_days=recent_window_days, as_of=as_of)
    except Exception as exc:  # noqa: BLE001 - Telegram report should survive tracker issues.
        logger.warning("recent performance summary failed: %s", type(exc).__name__)
        return "📈 近期表现：暂时无法读取，不影响今日扫描结果。"

    selected = list(records)
    if recent_window_days is not None and as_of:
        cutoff_date = pd.Timestamp(_parse_date(as_of) - pd.Timedelta(days=recent_window_days)).date()
        selected = [record for record in selected if _parse_date(record.signal_date) >= cutoff_date]
    closed_priority = [
        record
        for record in selected
        if record.status in {"tp2_hit", "stopped", "ambiguous"}
        or (record.status == "expired" and record.exit_reason in {"time_exit", "untriggered_expiry"})
    ]
    active_priority = [record for record in selected if record.status in {"tp1_hit", "triggered", "pending"}]
    recent = sorted(
        [*closed_priority, *active_priority],
        key=lambda record: record.exit_date or record.last_evaluated_date or record.entry_date or record.signal_date,
        reverse=True,
    )[:5]
    sample_size = int(summary.get("sample_size") or 0)
    sample_text = "样本不足" if sample_size < 5 else _fmt_pct(summary.get("win_rate"))
    avg_text = _fmt_pct(summary.get("average_return_pct"), signed=True) if sample_size >= 1 else "暂无"
    lines = [
        f"📈 近期信号表现｜近{recent_window_days}日",
        "",
        f"已生成：{summary.get('total_signals', 0)}",
        f"正式入场：{summary.get('triggered_signals', 0)}",
        f"进行中：{summary.get('open_signals', 0)}",
        f"TP1已触发仍追踪：{sum(1 for record in selected if record.status == 'tp1_hit')}",
        f"TP2：{summary.get('tp2_count', 0)}",
        f"止损：{summary.get('stopped_count', 0)}",
        f"时间退出：{sum(1 for record in selected if record.status == 'expired' and record.exit_reason == 'time_exit')}",
        f"未触发到期：{summary.get('untriggered_signals', 0)}",
        f"模糊结果：{summary.get('ambiguous_count', 0)}",
        "",
        f"胜率：{sample_text}",
        f"平均收益：{avg_text}",
        f"Profit Factor：{_fmt_number(summary.get('profit_factor'))}",
        f"Expectancy：{_fmt_pct(summary.get('expectancy_pct'), signed=True)}",
        "",
        "最近5笔：",
    ]
    lines.extend([_recent_record_line(record) for record in recent] or ["样本不足"])
    text = "\n".join(lines)
    if len(text) <= max_chars:
        return text
    keep = lines[:18]
    keep.append("最近5笔：")
    keep.extend([_recent_record_line(record) for record in recent[:3]] or ["样本不足"])
    return "\n".join(keep)[:max_chars].rstrip()


def _max_drawdown(returns: Sequence[float]) -> float | None:
    if not returns:
        return None
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for value in returns:
        equity += value
        peak = max(peak, equity)
        max_dd = min(max_dd, equity - peak)
    return round(max_dd, 4)


def render_summary_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Performance Summary",
        "",
        f"- Total signals: {summary.get('total_signals')}",
        f"- Triggered: {summary.get('triggered_signals')}",
        f"- Closed: {summary.get('closed_signals')}",
        f"- Win rate: {summary.get('win_rate')}",
        f"- Average return %: {summary.get('average_return_pct')}",
        f"- Profit factor: {summary.get('profit_factor')}",
        f"- Expectancy %: {summary.get('expectancy_pct')}",
    ]
    return "\n".join(lines) + "\n"


def _observed_date(year: int, month: int, day: int) -> date:
    actual = date(year, month, day)
    if actual.weekday() == 5:
        return actual - timedelta(days=1)
    if actual.weekday() == 6:
        return actual + timedelta(days=1)
    return actual


def _easter_date(year: int) -> date:
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def _nth_weekday(year: int, month: int, weekday: int, nth: int) -> date:
    current = date(year, month, 1)
    while current.weekday() != weekday:
        current += timedelta(days=1)
    return current + timedelta(days=7 * (nth - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    if month == 12:
        current = date(year, 12, 31)
    else:
        current = date(year, month + 1, 1) - timedelta(days=1)
    while current.weekday() != weekday:
        current -= timedelta(days=1)
    return current


def nyse_holidays(year: int) -> set[date]:
    holidays = {
        _observed_date(year, 1, 1),
        _nth_weekday(year, 1, 0, 3),
        _nth_weekday(year, 2, 0, 3),
        _easter_date(year) - timedelta(days=2),
        _last_weekday(year, 5, 0),
        _observed_date(year, 6, 19),
        _observed_date(year, 7, 4),
        _nth_weekday(year, 9, 0, 1),
        _nth_weekday(year, 11, 3, 4),
        _observed_date(year, 12, 25),
    }
    return holidays


def is_us_market_day(value: str | date) -> bool:
    day = _parse_date(value) if isinstance(value, str) else value
    return day.weekday() < 5 and day not in nyse_holidays(day.year)


def first_us_market_day(year: int, month: int) -> date:
    current = date(year, month, 1)
    while not is_us_market_day(current):
        current += timedelta(days=1)
    return current


def is_first_us_market_day(value: str | date) -> bool:
    day = _parse_date(value) if isinstance(value, str) else value
    return is_us_market_day(day) and day == first_us_market_day(day.year, day.month)


def previous_month(value: str | date) -> str:
    day = _parse_date(value) if isinstance(value, str) else value
    first = date(day.year, day.month, 1)
    previous = first - timedelta(days=1)
    return previous.strftime("%Y-%m")


def _summary_for_group(records: Sequence[SignalRecord]) -> dict[str, Any]:
    summary = summarize_performance(records)
    return {
        "signals": summary["total_signals"],
        "sample_size": summary["sample_size"],
        "win_rate": summary["win_rate"],
        "average_return_pct": summary["average_return_pct"],
    }


def _grouped_summary(records: Sequence[SignalRecord], field_name: str, values: Sequence[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for value in values:
        result[value] = _summary_for_group([record for record in records if getattr(record, field_name) == value])
    return result


def _common_text(records: Sequence[SignalRecord], field_name: str) -> str:
    counts: dict[str, int] = {}
    for record in records:
        text = str(getattr(record, field_name) or "").strip()
        if not text:
            continue
        key = text.split("；", 1)[0].split("。", 1)[0][:36]
        counts[key] = counts.get(key, 0) + 1
    if not counts:
        return "样本不足"
    return max(counts.items(), key=lambda item: item[1])[0]


def build_monthly_summary(records: Sequence[SignalRecord], month: str) -> dict[str, Any]:
    selected = [record for record in records if record.signal_date.startswith(month)]
    summary = summarize_performance(selected)
    winners = [record for record in _formal_closed(selected) if record.return_pct is not None and record.return_pct > 0]
    losers = [record for record in _formal_closed(selected) if record.return_pct is not None and record.return_pct <= 0]
    return {
        "month": month,
        **summary,
        "winning_trades": len(winners),
        "losing_trades": len(losers),
        "by_category": _grouped_summary(selected, "category", ["Best Trade", "Second Choice", "Watch"]),
        "by_trade_quality": _grouped_summary(selected, "trade_quality", ["A+", "A", "B", "C"]),
        "by_holding_period": _grouped_summary(selected, "expected_holding_period", ["2-5天", "3-10天", "1-3周"]),
        "common_success_structure": _common_text(winners, "technical_reason"),
        "common_failure_reason": _common_text(losers, "risk_reason"),
    }


def _line_group_stats(stats: Mapping[str, Any]) -> list[str]:
    lines = []
    for key, value in stats.items():
        lines.append(
            f"- {key}：信号 {value.get('signals', 0)}，样本 {value.get('sample_size', 0)}，胜率 {_fmt_pct(value.get('win_rate'))}"
        )
    return lines


def render_monthly_report(summary: Mapping[str, Any]) -> str:
    month = summary.get("month", "未知月份")
    sample_size = int(summary.get("sample_size") or 0)
    win_rate = "样本不足" if sample_size < 5 else _fmt_pct(summary.get("win_rate"))
    lines = [
        f"📊 月度信号表现｜{month}",
        "",
        f"信号总数：{summary.get('total_signals', 0)}",
        f"正式入场：{summary.get('triggered_signals', 0)}",
        f"仍未触发：{summary.get('untriggered_signals', 0)}",
        f"完成交易：{summary.get('sample_size', 0)}",
        f"TP2成功：{summary.get('tp2_count', 0)}",
        f"止损：{summary.get('stopped_count', 0)}",
        f"时间退出：{summary.get('time_exit_count', 0)}",
        f"模糊结果：{summary.get('ambiguous_count', 0)}",
        "",
        f"胜率：{win_rate}",
        f"平均收益：{_fmt_pct(summary.get('average_return_pct'), signed=True)}",
        f"中位数收益：{_fmt_pct(summary.get('median_return_pct'), signed=True)}",
        f"平均盈利：{_fmt_pct(summary.get('average_win_pct'), signed=True)}",
        f"平均亏损：{_fmt_pct(summary.get('average_loss_pct'), signed=True)}",
        f"Profit Factor：{_fmt_number(summary.get('profit_factor'))}",
        f"Expectancy：{_fmt_pct(summary.get('expectancy_pct'), signed=True)}",
        f"平均持有天数：{_fmt_number(summary.get('average_holding_days'))}",
        f"平均计划RR：{_fmt_number(summary.get('average_planned_rr'))}",
        f"平均实际RR：{_fmt_number(summary.get('average_realized_rr'))}",
        f"最大连续回撤：{_fmt_pct(summary.get('max_drawdown_of_closed_signal_sequence'), signed=True)}",
        "",
        f"🏆 最佳交易：{_trade_line(summary.get('best_trade'))}",
        f"📉 最差交易：{_trade_line(summary.get('worst_trade'))}",
        "",
        "分类表现:",
        *_line_group_stats(summary.get("by_category", {})),
        "",
        "Trade Quality:",
        *_line_group_stats(summary.get("by_trade_quality", {})),
        "",
        "持有周期:",
        *_line_group_stats(summary.get("by_holding_period", {})),
        "",
        f"常见成功结构：{summary.get('common_success_structure') or '样本不足'}",
        f"常见失败原因：{summary.get('common_failure_reason') or '样本不足'}",
    ]
    return "\n".join(lines)


def _trade_line(value: Any) -> str:
    if not isinstance(value, Mapping):
        return "样本不足"
    return f"{value.get('ticker')}｜{value.get('status')}｜{_fmt_pct(value.get('return_pct'), signed=True)}"


def _monthly_marker_path(store: PerformanceStore) -> Path:
    return store.summaries_dir / "monthly_sent.json"


def load_monthly_sent_markers(store: PerformanceStore) -> dict[str, Any]:
    store.ensure()
    path = _monthly_marker_path(store)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def mark_monthly_report_sent(store: PerformanceStore, month: str, *, sent_at: str | None = None) -> None:
    markers = load_monthly_sent_markers(store)
    markers[month] = sent_at or _iso_now()
    atomic_write_json(_monthly_marker_path(store), markers)


def save_monthly_report(store: PerformanceStore, summary: Mapping[str, Any], report_text: str) -> None:
    store.ensure()
    month = str(summary.get("month"))
    atomic_write_json(store.summaries_dir / f"monthly-{month}.json", dict(summary))
    _atomic_write_text(store.summaries_dir / f"monthly-{month}.md", report_text + "\n")


def maybe_build_monthly_report(
    store: PerformanceStore,
    records: Sequence[SignalRecord],
    *,
    as_of: str,
    enabled: bool = True,
) -> tuple[str | None, dict[str, Any] | None]:
    if not enabled or not is_first_us_market_day(as_of):
        return None, None
    month = previous_month(as_of)
    if month in load_monthly_sent_markers(store):
        return None, None
    summary = build_monthly_summary(records, month)
    report_text = render_monthly_report(summary)
    save_monthly_report(store, summary, report_text)
    return report_text, summary


def _run_git(args: Sequence[str], *, cwd: Path, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=cwd, check=check, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _git_remote_url(repo: Path) -> str | None:
    result = _run_git(["remote", "get-url", "origin"], cwd=repo, check=False)
    return result.stdout.strip() if result.returncode == 0 and result.stdout.strip() else None


def _copy_performance_dir(source: Path, destination: Path) -> None:
    src = source / "performance"
    dst = destination / "performance"
    if dst.exists():
        shutil.rmtree(dst)
    if src.exists():
        shutil.copytree(src, dst)
    else:
        dst.mkdir(parents=True, exist_ok=True)


def _clear_non_git_files(path: Path) -> None:
    for item in path.iterdir():
        if item.name == ".git":
            continue
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


def load_performance_branch(repo: Path | str, branch: str = "performance-data", *, dry_run: bool = False) -> dict[str, Any]:
    repo = Path(repo)
    if dry_run:
        return {"loaded": False, "dry_run": True}
    remote_url = _git_remote_url(repo)
    if not remote_url:
        return {"loaded": False, "skipped": True, "reason": "no_origin_remote"}

    fetch = _run_git(["fetch", "origin", branch, "--depth=1"], cwd=repo, check=False)
    if fetch.returncode != 0:
        logger.warning("performance-data branch is unavailable; continuing without historical performance data")
        return {"loaded": False, "missing": True, "reason": "branch_unavailable"}

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        _run_git(["init"], cwd=tmp)
        _run_git(["remote", "add", "origin", remote_url], cwd=tmp)
        _run_git(["fetch", "origin", branch, "--depth=1"], cwd=tmp)
        _run_git(["checkout", "FETCH_HEAD"], cwd=tmp)
        _copy_performance_dir(tmp, repo)
    return {"loaded": True, "branch": branch}


def init_performance_branch(repo: Path | str, branch: str = "performance-data", *, dry_run: bool = False) -> dict[str, Any]:
    repo = Path(repo)
    git_dir = repo / ".git"
    if not git_dir.exists():
        return {"created": False, "skipped": True, "reason": "not_git_repo"}
    existing = subprocess.run(
        ["git", "show-ref", "--verify", "--quiet", f"refs/heads/{branch}"],
        cwd=repo,
        check=False,
    )
    if existing.returncode == 0:
        return {"created": False, "skipped": False, "reason": "branch_exists"}
    if dry_run:
        return {"created": False, "skipped": False, "dry_run": True}
    subprocess.run(["git", "branch", branch], cwd=repo, check=True)
    return {"created": True, "skipped": False}


def persist_performance_branch(
    repo: Path | str,
    branch: str = "performance-data",
    *,
    message: str = "Update performance tracker data",
    retries: int = 3,
    dry_run: bool = False,
) -> dict[str, Any]:
    repo = Path(repo)
    if dry_run:
        return {"committed": False, "pushed": False, "dry_run": True}
    remote_url = _git_remote_url(repo)
    if not remote_url:
        return {"committed": False, "pushed": False, "reason": "no_origin_remote"}

    for attempt in range(1, retries + 1):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            _run_git(["init"], cwd=tmp)
            _run_git(["config", "user.email", "actions@github.com"], cwd=tmp)
            _run_git(["config", "user.name", "github-actions"], cwd=tmp)
            _run_git(["remote", "add", "origin", remote_url], cwd=tmp)
            fetch = _run_git(["fetch", "origin", branch, "--depth=1"], cwd=tmp, check=False)
            if fetch.returncode == 0:
                _run_git(["checkout", "-B", branch, "FETCH_HEAD"], cwd=tmp)
                _clear_non_git_files(tmp)
            else:
                _run_git(["checkout", "--orphan", branch], cwd=tmp)
                _clear_non_git_files(tmp)

            _copy_performance_dir(repo, tmp)
            _run_git(["add", "performance"], cwd=tmp)
            diff = _run_git(["diff", "--cached", "--quiet"], cwd=tmp, check=False)
            if diff.returncode == 0:
                return {"committed": False, "pushed": False, "reason": "no_changes"}
            _run_git(["commit", "-m", message], cwd=tmp)
            push = _run_git(["push", "origin", f"HEAD:{branch}"], cwd=tmp, check=False)
            if push.returncode == 0:
                return {"committed": True, "pushed": True, "attempt": attempt}
        logger.warning("performance-data push failed on attempt %s/%s", attempt, retries)
    return {"committed": True, "pushed": False, "reason": "push_failed"}


def update_store_with_histories(
    store: PerformanceStore,
    *,
    history_by_ticker: Mapping[str, pd.DataFrame],
    expiry_days: int = 10,
    max_holding_days: int = 10,
    run_date: str | None = None,
) -> dict[str, Any]:
    active = store.load_active()
    existing_closed = store.load_closed()
    new_active, newly_closed, counts = update_active_records(
        active,
        history_by_ticker=history_by_ticker,
        expiry_days=expiry_days,
        max_holding_days=max_holding_days,
    )
    closed_by_id = {record.signal_id: record for record in existing_closed}
    for record in newly_closed:
        closed_by_id[record.signal_id] = record
    store.save_state(new_active, list(closed_by_id.values()), run_date=run_date)
    summary = summarize_performance([*new_active, *closed_by_id.values()], as_of=run_date)
    store.save_summary(summary)
    return {"counts": counts, "summary": summary}


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def infer_signal_market_date(records: Sequence[Any], *, now: datetime | None = None) -> str:
    dates: list[date] = []
    for item in records:
        value = getattr(item, "data_timestamp", "") or ""
        try:
            dates.append(_parse_date(str(value)))
        except ValueError:
            continue
    if dates:
        return max(dates).isoformat()
    current = (now or datetime.now(ZoneInfo("America/New_York"))).date()
    while not is_us_market_day(current):
        current -= timedelta(days=1)
    return current.isoformat()


def _history_for_active(records: Sequence[SignalRecord]) -> dict[str, pd.DataFrame]:
    histories: dict[str, pd.DataFrame] = {}
    by_ticker: dict[str, list[SignalRecord]] = {}
    for record in records:
        by_ticker.setdefault(record.ticker, []).append(record)
    for ticker, ticker_records in by_ticker.items():
        start = min(record.signal_date for record in ticker_records)
        try:
            histories[ticker] = fetch_yfinance_history(ticker, start=start)
        except Exception as exc:  # noqa: BLE001 - tracker is fail-open.
            logger.warning("history fetch failed for %s: %s", ticker, type(exc).__name__)
    return histories


def _safe_tracker_records(store: PerformanceStore) -> tuple[list[SignalRecord], list[SignalRecord]]:
    return store.load_active(), store.load_closed()


def _append_text_to_file(path: Path | None, heading: str, text: str) -> None:
    if path is None:
        return
    try:
        existing = path.read_text(encoding="utf-8")
        path.write_text(f"{existing.rstrip()}\n\n## {heading}\n\n{text}\n", encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        logger.warning("failed to append %s to report artifact: %s", heading, type(exc).__name__)


def run_us_morning_report_with_performance(
    *,
    root: Path | str = ".",
    symbols: Sequence[str] | None = None,
    period: str | None = None,
    top_n: int | str | None = None,
    output_dir: Path | str | None = "reports",
    send_telegram: bool = True,
    write_performance: bool = True,
    telegram_summary_enabled: bool = True,
    monthly_report_enabled: bool = True,
) -> dict[str, Any]:
    from src.services.us_morning_report import (
        DEFAULT_PERIOD,
        build_candidate_views,
        group_candidate_views,
        run_us_morning_report,
        send_telegram_summary,
    )

    tracking_enabled = env_flag("PERFORMANCE_TRACKING_ENABLED", True)
    store = PerformanceStore(root)
    tracker_available = False
    tracker_warning = ""
    active_before: list[SignalRecord] = []
    closed_before: list[SignalRecord] = []

    if tracking_enabled:
        try:
            active_before, closed_before = _safe_tracker_records(store)
            tracker_available = True
        except Exception as exc:  # noqa: BLE001
            tracker_warning = f"tracker_read_failed:{type(exc).__name__}"
            logger.warning("performance tracker read failed: %s", type(exc).__name__)

    if tracking_enabled and tracker_available and write_performance:
        try:
            histories = _history_for_active(active_before)
            update_store_with_histories(
                store,
                history_by_ticker=histories,
                expiry_days=int(os.getenv("PERFORMANCE_SIGNAL_EXPIRY_DAYS", "10")),
                max_holding_days=int(os.getenv("PERFORMANCE_MAX_HOLDING_DAYS", "10")),
            )
            active_before, closed_before = _safe_tracker_records(store)
        except Exception as exc:  # noqa: BLE001
            tracker_warning = f"tracker_update_failed:{type(exc).__name__}"
            logger.warning("performance tracker update failed: %s", type(exc).__name__)
            tracker_available = False

    outcome = run_us_morning_report(
        symbols=symbols,
        period=period or os.getenv("US_REPORT_YFINANCE_PERIOD", DEFAULT_PERIOD),
        top_n=top_n or os.getenv("US_REPORT_TOP_N", "10"),
        send_telegram=False,
        output_dir=output_dir,
    )

    signal_date = infer_signal_market_date(outcome.ranked)
    new_signal_count = 0
    monthly_text: str | None = None
    monthly_sent = False
    if tracking_enabled and tracker_available and write_performance:
        try:
            include_watch = env_flag("PERFORMANCE_INCLUDE_WATCH", False)
            views = build_candidate_views(outcome.ranked, news=outcome.news, market_context=outcome.market_context)
            groups = group_candidate_views(views)
            new_records = signal_records_from_candidate_views(
                views,
                groups,
                signal_date=signal_date,
                generated_at=_iso_now(),
                source_report_path=str(outcome.report_path) if outcome.report_path else None,
                include_watch=include_watch,
            )
            new_signal_count = store.append_signals(new_records)
            active_now, closed_now = _safe_tracker_records(store)
            summary = summarize_performance(
                [*active_now, *closed_now],
                recent_window_days=int(os.getenv("PERFORMANCE_RECENT_WINDOW_DAYS", "30")),
                as_of=signal_date,
            )
            store.save_summary(summary)
            monthly_text, _monthly_summary = maybe_build_monthly_report(
                store,
                [*active_now, *closed_now],
                as_of=signal_date,
                enabled=monthly_report_enabled and env_flag("PERFORMANCE_MONTHLY_REPORT_ENABLED", True),
            )
        except Exception as exc:  # noqa: BLE001
            tracker_warning = f"tracker_save_failed:{type(exc).__name__}"
            logger.warning("performance tracker save failed: %s", type(exc).__name__)
            tracker_available = False

    recent_text = ""
    if tracking_enabled and telegram_summary_enabled and env_flag("PERFORMANCE_TELEGRAM_SUMMARY_ENABLED", True):
        if tracker_available:
            try:
                active_now, closed_now = _safe_tracker_records(store)
                recent_text = build_recent_performance_text(
                    [*active_now, *closed_now],
                    recent_window_days=int(os.getenv("PERFORMANCE_RECENT_WINDOW_DAYS", "30")),
                    as_of=signal_date,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("recent performance read failed: %s", type(exc).__name__)
                recent_text = "📈 近期表现：暂时无法读取，不影响今日扫描结果。"
        else:
            recent_text = "📈 近期表现：暂时无法读取，不影响今日扫描结果。"

    final_telegram_text = outcome.telegram_text if not recent_text else f"{outcome.telegram_text}\n\n{recent_text}"
    _append_text_to_file(outcome.report_path, "Performance Tracker", recent_text)

    telegram_sent = send_telegram_summary(final_telegram_text) if send_telegram else False
    if send_telegram and monthly_text:
        try:
            monthly_sent = send_telegram_summary(monthly_text)
            if monthly_sent:
                mark_monthly_report_sent(store, previous_month(signal_date))
        except Exception as exc:  # noqa: BLE001
            logger.warning("monthly performance Telegram failed: %s", type(exc).__name__)

    return {
        "report_path": str(outcome.report_path) if outcome.report_path else None,
        "telegram_sent": telegram_sent,
        "telegram_text": final_telegram_text,
        "tracker_available": tracker_available,
        "tracker_warning": tracker_warning,
        "new_signal_count": new_signal_count,
        "signal_date": signal_date,
        "monthly_report_sent": monthly_sent,
        "monthly_report_text": monthly_text,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="US Morning Scanner performance tracker")
    parser.add_argument("command", choices=("update", "summary", "monthly", "rebuild", "load-branch", "persist-branch", "morning-report"))
    parser.add_argument("--root", default=".", help="Repository or performance-data working directory")
    parser.add_argument("--branch", default=os.getenv("PERFORMANCE_DATA_BRANCH", "performance-data"))
    parser.add_argument("--month", help="YYYY-MM for monthly command")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm", action="store_true", help="Required for rebuild writes")
    parser.add_argument("--symbols", help="Comma-separated US symbols for morning-report")
    parser.add_argument("--top-n", default=os.getenv("US_REPORT_TOP_N", "10"))
    parser.add_argument("--period", default=os.getenv("US_REPORT_YFINANCE_PERIOD", "6mo"))
    parser.add_argument("--output-dir", default=os.getenv("US_REPORT_OUTPUT_DIR", "reports"))
    parser.add_argument("--no-telegram", action="store_true")
    parser.add_argument("--performance-write", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    store = PerformanceStore(Path(args.root))
    store.ensure()
    processed = skipped = errors = 0
    if args.command == "summary":
        records = [*store.load_active(), *store.load_closed()]
        summary = summarize_performance(records)
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        processed = len(records)
    elif args.command == "update":
        records = store.load_active()
        histories = {}
        for record in records:
            try:
                histories[record.ticker] = fetch_yfinance_history(record.ticker, start=record.signal_date)
                processed += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning("history fetch failed for %s: %s", record.ticker, type(exc).__name__)
                errors += 1
        if not args.dry_run:
            result = update_store_with_histories(store, history_by_ticker=histories, run_date=date.today().isoformat())
            processed = result["counts"]["processed"]
            skipped = result["counts"]["skipped"]
            errors += result["counts"]["errors"]
        else:
            skipped = len(records)
    elif args.command == "monthly":
        records = [*store.load_active(), *store.load_closed()]
        month = args.month or date.today().strftime("%Y-%m")
        selected = [record for record in records if record.signal_date.startswith(month)]
        summary = summarize_performance(selected)
        print(json.dumps({"month": month, "summary": summary}, ensure_ascii=False, indent=2, sort_keys=True))
        processed = len(selected)
    elif args.command == "rebuild":
        if not args.dry_run and not args.confirm:
            parser.error("rebuild writes require --confirm; use --dry-run to inspect")
        records = store.load_all_signals()
        processed = len(records)
        if args.dry_run:
            skipped = processed
        else:
            store.save_state(records, [], run_date=date.today().isoformat())
    elif args.command == "load-branch":
        result = load_performance_branch(args.root, args.branch, dry_run=args.dry_run)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    elif args.command == "persist-branch":
        result = persist_performance_branch(args.root, args.branch, dry_run=args.dry_run)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        if result.get("pushed") is False and result.get("reason") == "push_failed":
            logger.warning("performance tracker data was not pushed; artifact backup should be used")
    elif args.command == "morning-report":
        result = run_us_morning_report_with_performance(
            root=args.root,
            symbols=[item.strip().upper() for item in args.symbols.split(",") if item.strip()] if args.symbols else None,
            period=args.period,
            top_n=args.top_n,
            output_dir=args.output_dir,
            send_telegram=not args.no_telegram,
            write_performance=args.performance_write,
            telegram_summary_enabled=env_flag("PERFORMANCE_TELEGRAM_SUMMARY_ENABLED", True),
            monthly_report_enabled=env_flag("PERFORMANCE_MONTHLY_REPORT_ENABLED", True),
        )
        print(json.dumps({k: v for k, v in result.items() if k not in {"telegram_text", "monthly_report_text"}}, ensure_ascii=False, sort_keys=True))
    print(f"processed={processed} skipped={skipped} errors={errors}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
