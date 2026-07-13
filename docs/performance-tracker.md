# AI Performance Tracker

This tracker records US Morning Scanner signals after the morning report is generated, then evaluates later trading days with Yahoo Finance OHLC data. It is research telemetry only, not a broker fill record and not a performance guarantee.

## Storage

Long-term data is stored on the independent `performance-data` branch under:

```text
performance/
  signals.jsonl
  active_signals.json
  closed_signals.json
  daily/
  summaries/
    latest.json
    latest.md
    monthly-YYYY-MM.json
    monthly-YYYY-MM.md
    monthly_sent.json
```

`signals.jsonl` is append-only. Historical signal content is not rewritten, deleted, or edited to improve statistics. `active_signals.json` and `closed_signals.json` are current state snapshots written atomically.

## Workflow

The US Morning workflow loads `performance/` from `performance-data`, updates active signals, runs the existing 99-stock scanner, saves new signals from Best Trade and Second Choice, optionally records Watch when `PERFORMANCE_INCLUDE_WATCH=true`, adds a short recent-performance block to Telegram, then pushes only `performance/` back to `performance-data`.

Scheduled runs write performance data by default. Manual runs default to no write because test or historical re-runs should not accidentally contaminate live statistics. Use the `performance_write` workflow input only when you intentionally want a manual run to persist tracker data.

## Signal Rules

Signals are generated only after the Morning Scanner report is complete. `signal_date` uses the latest US market data date from Yahoo Finance, not the Malaysia calendar date.

Entry is evaluated only from trading days after `signal_date`; same-day OHLC is never used to create the signal and judge its outcome.

- Pullback entry triggers when the day range overlaps the entry zone.
- Breakout entry triggers when high reaches `breakout_trigger`.
- If both happen on the same day, pullback is recorded and the note explains both were possible.
- Untriggered signals expire with `exit_reason=untriggered_expiry` and do not enter win rate.
- Entered trades that reach max holding period exit at that day's close with `exit_reason=time_exit` and do enter formal statistics.

## TP1 Is Not A Final Win

`tp1_hit` means the first target was touched, but the trade remains active. It keeps:

- `tp1_hit_date`
- `tp1_return_pct`
- `first_target_result`

It must not have `exit_date`, `exit_price`, `exit_reason`, or final `return_pct`. The signal continues tracking until TP2, stop, time exit, ambiguous, unknown, or corporate-action review.

## Statistics

Formal win rate uses only closed, non-ambiguous, entered trades:

- `tp2_hit`
- `stopped`
- `expired` with `exit_reason=time_exit`

Pending, triggered, TP1-active, untriggered expiry, Watch by default, ambiguous, unknown, and corporate-action review records are excluded from formal win rate. Ambiguous results are counted separately because daily OHLC cannot prove whether stop or target happened first.

Profit Factor is gross profit divided by absolute gross loss. Expectancy is win rate times average win plus loss rate times average loss. If the denominator is zero or the sample is insufficient, the value is `null`/`暂无`.

## Monthly Report

The monthly report is generated on the first actual US market day of each month for the previous month. A sent marker is saved in `performance/summaries/monthly_sent.json`; if Telegram sending fails, the marker is not written so the next run can retry.

Monthly category, trade-quality, holding-period, success-structure, and failure-reason summaries use structured fields already stored in `SignalRecord`. Gemini does not classify historical outcomes.

## Operations

Disable the tracker without affecting the Morning Scanner:

```bash
PERFORMANCE_TRACKING_ENABLED=false
```

Run locally without writing Telegram:

```bash
python scripts/performance_tracker.py morning-report --no-telegram
python scripts/performance_tracker.py summary
python scripts/performance_tracker.py monthly --month 2026-07
```

Load and persist the data branch:

```bash
python scripts/performance_tracker.py load-branch --branch performance-data
python scripts/performance_tracker.py persist-branch --branch performance-data
```

Back up the branch by cloning or fetching `performance-data`. To restore, copy the desired `performance/` directory into the repo and run `persist-branch`. Artifacts are backups only; Git branch storage is the durable source.

## Limitations

Yahoo Finance daily OHLC cannot determine intraday order when entry, stop, and target occur on the same day. Those cases are marked `ambiguous` and excluded from formal win rate.

Corporate actions, splits, or bad adjusted data can distort fixed price levels. Obvious price-ratio breaks are marked for review and excluded from formal statistics unless manually verified.

This tracker does not know actual account orders, fills, slippage, partial exits, commissions, or position sizing. Treat it as scanner-quality research, not actual trading P&L.
