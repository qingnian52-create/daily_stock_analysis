# US Morning Trading Report

The dedicated US morning report is a lightweight scanner for GitHub Actions. It ranks the default 99-symbol US stock universe with Yahoo Finance data first, enriches only the final top 10 with Tavily and Gemini, then sends one grouped Telegram summary. Long summaries may be split into two Telegram messages at complete report blocks; it never sends per-stock messages.

## Workflow

The workflow is `.github/workflows/us-morning-report.yml`.

- Schedule: Tuesday-Saturday at `00:00 UTC`, which is Tuesday-Saturday `08:00` in Malaysia time.
- Manual run: GitHub Actions -> `US Morning Trading Report` -> `Run workflow`.
- Artifacts: Markdown report files are uploaded from `reports/`; performance backup files are uploaded from `performance/`.
- Performance data: scheduled runs load and write the `performance-data` branch. Manual runs do not write performance data unless the `performance_write` input is explicitly enabled.

## Required Secrets

- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`
- `GEMINI_API_KEY`

## Optional Secrets

- `TAVILY_API_KEY`
- `GEMINI_API_KEYS` as a fallback if `GEMINI_API_KEY` is not set.
- `TAVILY_API_KEYS` as a fallback if `TAVILY_API_KEY` is not set.
- `TELEGRAM_MESSAGE_THREAD_ID`

## Optional Variables

- `US_STOCK_UNIVERSE`: comma-separated US symbols. If omitted, the script uses a built-in liquid large-cap universe.
- `US_REPORT_TOP_N`: report size, hard-capped at `10`.
- `US_REPORT_YFINANCE_PERIOD`: Yahoo Finance history period, default `6mo`.
- `GEMINI_MODEL`: default `gemini-2.5-flash`.

The scanner computes market context, current price, entry zone, breakout trigger, stop loss, targets, and risk/reward from Yahoo Finance technical data. Tavily supplies market and stock news. Gemini is only used for catalyst/risk compression and must not overwrite program-calculated market values or trading levels.

## Performance Tracker

The workflow can append a short `📈 近期信号表现` block to the existing Telegram report. This does not change the scanner universe, ranking, trade levels, reasons, or main Telegram layout.

Recommended variables:

- `PERFORMANCE_TRACKING_ENABLED=true`
- `PERFORMANCE_DATA_BRANCH=performance-data`
- `PERFORMANCE_SIGNAL_EXPIRY_DAYS=10`
- `PERFORMANCE_MAX_HOLDING_DAYS=10`
- `PERFORMANCE_INCLUDE_WATCH=false`
- `PERFORMANCE_RECENT_WINDOW_DAYS=30`
- `PERFORMANCE_MONTHLY_REPORT_ENABLED=true`
- `PERFORMANCE_CONSERVATIVE_AMBIGUOUS=true`
- `PERFORMANCE_TELEGRAM_SUMMARY_ENABLED=true`
- `PERFORMANCE_MANUAL_WRITE_DEFAULT=false`

Full tracker behavior, storage, backup, restore, and statistical definitions are documented in `docs/performance-tracker.md`.

## Local Run

```bash
python scripts/us_morning_report.py --no-telegram
python scripts/us_morning_report.py --symbols AAPL,MSFT,NVDA,AMZN,META
```

The report is informational only. Confirm liquidity, events, and your own risk limits before trading.
