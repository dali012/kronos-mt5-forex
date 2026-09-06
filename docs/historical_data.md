# Historical data and fixed-strategy baseline

Phase 3 provides a public Binance USD-M perpetual data pipeline and a reproducible
NautilusTrader replay of the deployed trend strategy. It does not tune parameters
or introduce another backtest engine. Results describe an explicitly approximate
historical replay, not evidence that the strategy has an unseen profitable edge.

## Installation and commands

Run from a checkout of the source commit recorded in the report. Python 3.12 is
recommended; NautilusTrader is pinned to 1.228.0 by `requirements-live.txt`.

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements-research.txt
export PYTHONPATH=src

# Default: eight deployed symbols, 1d candles, 2023-01-01 through yesterday UTC.
# Historical funding is included; no trading credentials are read or accepted.
python -m kronos_mt5.marketdata download --destination research/data

# Small real-data validation smoke, independent of strategy warm-up.
python -m kronos_mt5.marketdata download \
  --symbols BTCUSDT --intervals 1d --start 2024-01-01 --end 2024-01-07 \
  --destination research/real-smoke

# Copy the exact manifest path printed by download; validation never uses network.
python -m kronos_mt5.marketdata validate \
  --manifest research/data/manifests/dataset-<ID>.json

# Public exchange filter snapshot. Save and reuse it for reproducibility.
python -m kronos_mt5.marketdata exchange-info --output research/exchange-info.json

# Freeze the deployed signal/risk parameters, with documented replay costs.
python -m kronos_mt5.baseline config --output research/baseline-config.json

# Development + chronological windows; final holdout is not evaluated by default.
python -m kronos_mt5.baseline run \
  --manifest research/data/manifests/dataset-<ID>.json \
  --config research/baseline-config.json --exchange-info research/exchange-info.json \
  --output research/baseline

# One fixed-baseline evaluation of the final holdout, shown separately.
# Same config, filters and code must match baseline-lock.json.
python -m kronos_mt5.baseline run \
  --manifest research/data/manifests/dataset-<ID>.json \
  --config research/baseline-config.json --exchange-info research/exchange-info.json \
  --include-holdout --output research/baseline

# Offline reproduction: verifies source/dependency hashes and dataset identity,
# reruns the original config, and fails if any reported metrics differ.
python -m kronos_mt5.baseline reproduce \
  --report research/baseline/report.json \
  --manifest research/data/manifests/dataset-<ID>.json \
  --output research/reproduced

# Bundled deterministic synthetic fixtures: no network, database, or credentials.
python -m kronos_mt5.baseline smoke --output research/synthetic-smoke
```

All commands are noninteractive. A corrupt archive/Parquet/manifest, invalid data,
missing required funding, inadequate warm-up, changed frozen configuration, or
failed reproduction returns a nonzero exit code and a JSON error on stderr.
The downloader prints the immutable dataset manifest path and validation result
as JSON on success. Repeating it resumes validated partitions. Multiple simultaneous
writers to the same destination are not supported; use a separate research root.

Symbols, supported fixed UTC intervals, inclusive start/end dates and destination
are CLI arguments. Supported intervals are `1m 3m 5m 15m 30m 1h 2h 4h 6h 8h 12h 1d`;
weekly/calendar-month aggregations are deliberately excluded. The strategy consumes
**1d**, derived from the live runner's `1-DAY-LAST-EXTERNAL` bar spec and guarded by
a regression test. The baseline only loads daily data; a multi-year eight-symbol
daily universe is small. Downloader and validator process one partition at a time.

A representative portfolio run can use `--start 2024-01-01 --end 2024-12-31` for all
eight symbols. The first 253 days are warm-up, leaving 113 evaluation days. For a
single-symbol smoke, also generate the configuration with `config --symbols BTCUSDT`;
reports label such runs `subset_smoke_only`, not the deployed portfolio baseline.
`download --no-funding` requires `config --omit-funding` for replay and produces
`INCOMPLETE_FUNDING_OMITTED`; it must not be presented as a cost-complete baseline.

For a complete run through the latest completed UTC day, this block captures the
printed manifest path automatically (all artifacts remain gitignored):

```bash
mkdir -p research
python -m kronos_mt5.marketdata download --destination research/full-data > research/full-download.json
KRONOS_DATASET_MANIFEST=$(python -c 'import json; print(json.load(open("research/full-download.json"))["manifest"])')
python -m kronos_mt5.marketdata exchange-info --output research/full-exchange-info.json
python -m kronos_mt5.baseline config --output research/full-config.json
python -m kronos_mt5.baseline run --manifest "$KRONOS_DATASET_MANIFEST" \
  --config research/full-config.json --exchange-info research/full-exchange-info.json \
  --include-holdout --output research/full-baseline
```

## Sources, usage, and format

Primary sources:

- [Binance public data documentation](https://github.com/binance/binance-public-data):
  monthly/daily ZIP archives and SHA-256 `.CHECKSUM` files.
- [USD-M archive index](https://data.binance.vision/?prefix=data/futures/um/):
  `monthly/klines/SYMBOL/INTERVAL/`, `daily/klines/SYMBOL/INTERVAL/`, and
  `monthly/fundingRate/SYMBOL/`.
- [Binance USD-M market-data API](https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data):
  `/fapi/v1/klines`, `/fapi/v1/fundingRate`, `/fapi/v1/exchangeInfo`.
- [Nautilus backtesting](https://nautilustrader.io/docs/latest/concepts/backtesting/):
  event sequencing, synthetic market-data assumptions and simulated exchange.

Public availability does not itself establish permission to redistribute datasets.
Check Binance's applicable terms, geographic restrictions and usage permissions
before redistributing or commercially using data. This PR stores neither market
data nor generated results in Git; it does not claim a new dataset license.
Public archives may later be corrected. Research snapshots pin downloaded content;
ordinary resume does not refresh already-valid archives. To compare revisions,
download to a separate root and retain both manifests.

Candles prefer monthly archives, falling back on daily archives and then the public
API only when archives return 404. Funding uses monthly archives, with public
funding-history pagination for missing months. Checksum mismatch or corrupt ZIP
never silently falls back to another source. Missing checksum files are recorded
as `checksum_verified: false`. Requests retry temporary errors at most five times
with bounded exponential backoff; API pages are paced one second apart. Short pages
are not assumed to be EOF. HTTP 429 `Retry-After` is honored; waits exceeding 30 seconds
return a resumable error rather than hammering the endpoint.

```text
research/                         # gitignored
  data/
    klines/venue=binance-um/symbol=BTCUSDT/interval=1d/part-YYYY-MM-<content>.parquet
    funding/venue=binance-um/symbol=BTCUSDT/part-YYYY-MM-<content>.parquet
    manifests/binance-um__BTCUSDT__1d.json
    manifests/binance-um__BTCUSDT__funding.json
    manifests/dataset-<ID>.json
    manifests/failure-<symbol>-<interval>-<partition>.json
  exchange-info.json
  baseline-config.json
  baseline/
    baseline-lock.json
    configuration.json
    report.json
    report.md
    development-replay.json
    holdout-replay.json            # only if explicitly evaluated
```

Parquet uses Snappy compression. `open_time` is the candle's unique UTC Unix epoch
**millisecond** key; `close_time = open_time + interval - 1ms`. USD-M timestamps are
not silently rescaled from spot microsecond data. OHLC must be finite and positive,
`low <= open,close <= high`, and volume must be finite and nonnegative. Duplicate,
out-of-order, overlapping, off-grid and missing candles fail validation, including
missing first/last observations. The downloader only accepts fully completed UTC
days and drops incomplete candles before writing. It never sorts/deduplicates bad
source rows into an apparently valid partition.

Funding stores `funding_time` (original UTC milliseconds), signed `funding_rate`,
and nullable `mark_price`. Interval changes up to eight hours are allowed. Funding
gaps exceeding eight hours plus 60 seconds of settlement-timestamp jitter fail
validation. This detects large gaps but cannot prove an individual event is absent
when funding cadence changes below eight hours. The tolerance is explicit in the
manifest; original timestamps are preserved, including Binance's millisecond jitter.
Archived rates have no historical mark price; API rows sometimes do.

Files are written to a unique temporary file beside their destination, then renamed
atomically. Manifests are likewise written atomically, after validated partitions.
Resume verifies stored-file checksums and data before skipping. Daily fallbacks are
checkpointed individually. Content-addressed partition filenames preserve old
snapshots when requested ranges expand. Orphan temporary/unreferenced files are
ignored; they are never treated as complete dataset partitions.

## Manifest schema

`dataset-<ID>.json` has `schema_version: 1`, venue, symbols, intervals, `start_ms`,
exclusive `end_ms`, funding inclusion, download time, and exact partition entries.
It is a portable snapshot: paths are relative to the dataset root, two levels above
the manifest. Preserve that layout when copying a dataset. Validation rejects path
escapes, missing groups and cross-partition gaps/overlaps. No unlisted files are read.

Each partition records:

- `symbol`, `interval`, `kind`, logical `partition`, `start_ms`, exclusive `end_ms`,
  `rows`, `downloaded_at`, `file`, and stored Parquet `file_sha256`.
- `sources`: exact archive or API request URL, archive SHA-256, verification status,
  and byte size where available. Archive checksums and stored-file checksums are
  distinct; neither is substituted for the other.
- `validation`: row count, duplicates, out-of-order rows, missing intervals,
  overlaps, invalid OHLC/volume, first/last UTC opens, expected rows and `ok`.
  Funding has its own rate, coverage and missing-mark checks.

Series manifests (`manifest_version: 1.0.0`) are mutable resume indexes. Dataset
snapshots carry copies of their partition metadata, so later index updates do not
change previous reports. The report embeds the exact snapshot and its SHA-256,
configuration and fingerprint, exchange filter snapshot, source commit, dirty-tree
flag, source-file hashes, and dependency versions. Keep these artifacts privately;
none belong in a commit.

## Production reconstruction and approximations

Read-only inspection on 2026-09-06 identified deployed commit
`dc8a74c2a9afda9f28a0d8b2f7a1bf6044df35b2`; the deployed strategy file matched that
commit's hash. Systemd runs `kronos_mt5.live.run_trend_binance` from the server repo,
with production state in `logs/companion.db`. Services, configuration, that database,
orders and positions were not modified. Research never imports the live runner,
loads `.env`, opens the companion database or connects a trading execution client.

The non-secret snapshot is in `baseline/config.py`: BTC, ETH, BNB, XRP, ADA, SOL,
LTC and LINK; daily sign-momentum lookbacks 21/63/126/252; 33-return volatility;
15% per-leg vol target; allocator target 10%; volatility stops; cost-aware
rebalancing; adverse-funding veto; and 20% portfolio drawdown kill-switch. These
signal/risk parameters cannot be tuned through the baseline configuration reader.
Nautilus dispatches symbols in the deployed order. Its existing allocator's
one-cycle lag is preserved. Shadow challengers are not evaluated.

The Phase 2 sanitized audit confirms this universe and deployment. Its summary
SHA-256 is recorded in configuration provenance. It has 460 fills, 29 missing
commission records, and no reliable liquidity/execution telemetry. It cannot
identify a realistic live maker/taker mix or a fee tier. The baseline therefore
uses an **explicit conservative approximation of 5bps taker commission per side**,
1bp half-spread and 2bps adverse slippage. These are stated assumptions, not a claim
about an account's current Binance rate. They can be varied as execution sensitivity
inputs before the holdout lock is established.

The actual `TrendStrategy` and `RiskState` supply signals, volatility sizing,
allocator, stops and rebalance decisions. A research-only subclass changes timing
and execution transport; production files are unchanged. The repository strategy
has newer patient-order safety code than the server. Its signal/risk calculations
are reused; exact old patient-limit lifecycle behavior is not reconstructed.

Important limits, repeated in every report:

- Patient entries are approximated as market orders at the **next executable daily
  open**, with full taker costs. There are no assumed maker fills, queue priority,
  partial-fill liquidity, cancel confirmations, or 300-second fallback prices.
- Intraday quotes assume O→H→L→C at open, 08:00+1ns, 16:00+1ns and end−2ns.
  `bar_path: OLHC` is available for a pre-holdout path sensitivity check. Real
  intraday timing is unknown. Stop orders execute against these quotes, including
  gap-through losses; the engine never fills magically at an unavailable trigger.
- Entry quantity/price increments, minimum quantity/notional and maximum quantity
  use the pinned public perpetual exchange filter snapshot. Linear USDT-settled
  `CryptoPerpetual` instruments replace the old spot-like generic test instruments.
  New exposure above two times gross equity is rejected. Historical filter changes,
  maintenance tiers, liquidation/ADL and exact account leverage are unknown.
- Funding cash flows enter the simulated account at original funding timestamps
  before subsequent sizing. Long positive-rate positions pay; shorts receive.
  Published marks are used when available; otherwise the last causal synthetic
  quote is an explicitly counted approximation. The funding veto sees the latest
  *settled* rate, whereas production polls a predicted/current rate.
- Mark-price watchdog incidents and five-second intraday portfolio risk monitoring
  cannot be reproduced from daily OHLC. The replay checks risk at synthetic quotes.
- Today's survivor universe and previous strategy research contaminate a claim of
  pristine out-of-sample selection. Chronological splits alone cannot undo prior
  experimentation on 2023+ data. No historical profitability claim is made here.

## Leakage, windows, and reporting

Closed bars enter indicators only at `open + 1 day − 1ns`. An order intent generated
there waits until the next day's open quote. Earlier bars fill only the warm-up
buffer. The last warm-up close may queue a first evaluation-day intent, but no
warm-up fills/equity changes appear in reported returns. Future price/rate perturbation
tests assert earlier indicators, decisions and fills are unchanged.

The default dataset starts 2023-01-01, so 253 bars of warm-up place the earliest
executable development date on **2023-09-11**. Download earlier history if a longer
warm-up prefix is required; baseline evaluation always excludes its first 253 days.
Development ends at 2026-01-01 exclusive. The final holdout starts exactly then and
ends after the latest complete dataset day. Insufficient development or holdout
warm-up fails explicitly. `backtest.walk_forward` and this baseline share the same
chronological window generator; no calibration, bootstrap or random shuffle is
used for this fixed-parameter replay. Windows are adjacent 90-day periods with a
possibly shorter tail. All windows reset to initial capital, use preceding warm-up
only, start flat, and liquidate at their predetermined final close with costs.

A `baseline-lock.json` created before evaluation freezes configuration, source hashes,
exchange filters and the fixed holdout boundary. `--include-holdout` is explicit.
It does not pretend to prevent someone from copying data or creating a new output
directory; maintaining an untouched future holdout is also a research discipline.
Once its results are inspected, the 2026 holdout must not guide later strategy
selection, execution calibration or Reddit/academic idea screening. Those require
new untouched future observations.

JSON and Markdown separate development, walk-forward windows and final holdout.
They include equity/return/CAGR, drawdown and duration, Sharpe/Sortino/Calmar,
flat-to-flat trade count and statistics, holding time, turnover, gross/net PnL,
commission/spread/slippage/funding, symbol/direction/month/year attribution,
time-weighted exposure/concurrency, and the existing portfolio volatility regime.
Regime attribution uses the previously observable classifier state.

Ratios use daily returns and 365 periods/year; risk-free rate is zero. Undefined
ratios/trade metrics are `null` and suppressed explicitly. Annualization below 365
days is low confidence. Drawdown uses the observed synthetic intraday equity points
and duration includes recovery (or ends censored at the replay boundary). Calendar
PnL is marked to market, not assigned wholly to a trade's closing month. Costs are
positive when paid; funding can be negative when received. Spread/slippage are
embedded in execution prices and added back only to derive gross PnL; they are
never subtracted twice. Currency rounding residuals remain visible and are checked.

Cash/no-trade is a zero-return reference over each matching window. BTC buy-and-hold
is explicitly informational: an unlevered futures-candle price reference with no
commission/funding, not an equivalent executable spot/perpetual portfolio.

## Verification

```bash
python -m pytest -q -m "not slow"
python -m compileall -q src tests backtest
ruff check src/kronos_mt5/marketdata src/kronos_mt5/baseline \
  src/kronos_mt5/walk_forward.py backtest/walk_forward.py \
  tests/test_marketdata.py tests/test_baseline.py
git diff --check
```

Tests use mocked HTTP, generated deterministic OHLCV/funding, known equity/trade
sequences and small Nautilus replays. They cover resume/pagination, checksums,
corrupt files, UTC/current candles, gaps/duplicates, atomic writes, precision and
notional filters, next-bar execution, future-data perturbations, funding/accounting,
warm-up, chronological boundaries, frozen holdout, and exact report reproduction.
