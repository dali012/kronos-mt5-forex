# Performance audit (offline forensics)

An offline, read-only tool that recomputes performance and data quality from a
sanitized companion export. It exists so that Phase 3 strategy work starts from
measured facts rather than from the numbers the bot happened to record.

It is **forensics, not a backtest, and not an endorsement**.

## What it can conclude

- What the recorded equity curve did: return, drawdown, volatility-based ratios,
  exposure and daily hit rate, computed from a normalized daily series.
- Whether the accounting closes: whether equity change is explained by realized
  PnL, commissions, funding and the change in unrealized PnL, and how large the
  unexplained residual is.
- How the account traded: fill counts, turnover, commissions and recorded
  slippage by symbol, side and execution kind.
- How the recorded data is deficient, with specific coded findings.
- How shadow models differ from the live model **in positioning**.

## What it cannot conclude

- **Whether the strategy is profitable.** The sample is Binance TESTNET with
  `DEMO_ONLY=true`. Testnet fills, liquidity, funding and latency do not
  reproduce live queue position or market impact. A profitable testnet curve is
  evidence the system runs, not that the edge is real.
- **Whether any shadow model is better.** Shadow rows record target portfolios,
  not returns. Without forward prices, PnL is reported as `NOT_EVALUABLE`.
- **Why anything happened.** Equity moves shown next to incidents are temporal
  association. The tool has no causal mechanism and says so wherever it reports.
- **Trade statistics, when position lifecycle coverage is incomplete.** Win rate,
  profit factor and expectancy over a handful of recorded positions describe a
  biased sub-sample.
- **Anything about a different period.** 81 days is one regime, not a
  distribution. Do not tune parameters against it.

## Running it

Against the sanitized export archive:

```bash
python -m kronos_mt5.performance_audit \
  --input /path/to/kronos-performance-export-YYYYMMDD-HHMMSSZ.tar.gz \
  --output-dir /tmp/kronos-performance-report
```

Against a sanitized database directly:

```bash
python -m kronos_mt5.performance_audit \
  --input /path/to/companion_sanitized.db \
  --output-dir /tmp/kronos-performance-report
```

Outputs, all written into `--output-dir`:

| File | Contents |
|---|---|
| `report.md` | Human-readable performance and data-quality report |
| `summary.json` | Versioned machine-readable metrics and findings |
| `daily_equity.csv` | Normalized daily equity series |
| `fill_attribution.csv` | Execution metrics grouped by symbol/side/kind |

### Safety properties

- The input is opened through a SQLite **read-only URI** with `query_only=1`; a
  test asserts the input file's hash and mtime are unchanged after a full run.
- Archives are extracted only into a temporary directory that is deleted
  afterwards, and every member is validated first: absolute paths, `..`
  traversal, symlinks, hardlinks and device nodes are rejected, with member-count
  and total-size ceilings against decompression bombs.
- Nothing is written back to the export, and no free-text database field is
  echoed beyond what a finding needs as evidence.
- Generated reports, exports and databases are git-ignored — never commit real
  trading data.

## Metric definitions

**Daily series.** One point per UTC day, the **last valid observation of that
day**. No interpolation; missing days are reported, never filled. Every ratio
below is computed on this series, because the raw table holds a snapshot roughly
every 60 seconds and those observations are not independent.

The report also prints the **raw observation series** (first, last, intraday
minimum and maximum). These legitimately differ from the daily figures — a raw
`SELECT MIN(equity)` sees intraday lows the daily series never keeps. Both are
shown so the two can be reconciled.

### Day counting

Four counts, deliberately distinct and never interchangeable:

| Field | Meaning | Example (16 Jun .. 05 Sep) |
|---|---:|---|
| `calendar_days_covered` | inclusive span of dates | 82 |
| `elapsed_days` | time actually elapsed — **the CAGR exponent** | 81 |
| `days_observed` | dates that carry an observation | 82 |
| `return_periods` | day-over-day returns produced | 81 |

`ratio_return_periods` is the subset of `return_periods` that spans exactly one
calendar day; only those feed the volatility-based ratios.

| Metric | Definition |
|---|---|
| Total return | `last_daily / first_daily - 1` |
| Period return | `last_observation / first_observation - 1` (raw endpoints) |
| Daily return | `E_t / E_{t-1} - 1` between consecutive observed days |
| Annualized return (CAGR) | `(1 + total_return) ** (365 / elapsed_days) - 1` |
| Annualized volatility | `stdev(consecutive daily returns) * sqrt(365)`, sample stdev |
| Sharpe | `mean / stdev * sqrt(365)` over consecutive returns, risk-free rate **0** |
| Sortino | `mean / sqrt(mean(min(r,0)^2)) * sqrt(365)` over consecutive returns |
| Calmar | `annualized_return / abs(max_drawdown)` |
| Max drawdown | Worst `value / running_peak - 1` on the daily series |
| Recovery | First later day whose equity reaches the pre-drawdown peak |
| Exposure | Derived from `n_open` per day |

`periods_per_year = 365` because the venue trades every calendar day.

**CAGR compounds over elapsed time, not over the inclusive date count.** Using
82 instead of 81 for this export would report −0.3095 % instead of the correct
−0.3134 %. When `elapsed_days <= 0` (a single observation, or several
observations on one day) CAGR is `null`, never zero.

**Gap-spanning returns are excluded from the ratios.** A return bridging missing
calendar days carries multi-day variance; pooling it with true daily returns
deflates volatility and inflates Sharpe. Such returns stay in the return series,
in `total_return`, and in the positive/negative day counts, but are excluded from
volatility, Sharpe and Sortino. `excluded_gap_returns` reports how many, and
`ratios_note` explains it in the report. Below `MIN_RATIO_RETURNS` (2) usable
returns the ratios are reported as **unavailable** rather than computed.

Nothing is ever interpolated into a gap.

**Low confidence.** Any window shorter than 365 elapsed days sets
`annualized_low_confidence`, and the report prints the caveat next to the ratios.

## Sign conventions

These differ between sources, which is itself a source of error:

| Field | Convention |
|---|---|
| `income.amount` | Binance ledger. `REALIZED_PNL` signed; `COMMISSION` **negative** (a cost); `FUNDING_FEE` signed (received +, paid −) |
| `equity.commissions` | **Positive** = cost (it is `-COMMISSION` income) |
| `fills.commission` | **Positive** = cost, observed fills only |
| `fills.slippage` | Implementation shortfall in **quote currency (USDT)**, **positive = worse** than the reference price |
| `impl_shortfall_bps` | Same sign convention in basis points (newer exports only) |

`fills.slippage` is **not** basis points and **not** per unit. Verified against
its producer (`companion/recorder.py` →
`execution.implementation_shortfall_quote`) and empirically against exported
rows. It is a *measurement* already embedded in the execution price, so the audit
reports it but never adds it to the cash accounting.

**Accounting identity checked:**

```
equity_change = realized - commissions + funding + other_income
                + unrealized_change + residual
```

The residual is reported with its tolerance and likely explanations. Values are
never forced to agree.

### Reconciliation window

Ledgers are scoped to exactly the interval the equity curve covers:

```
first_equity_ts < record_ts <= last_equity_ts
```

The start is **exclusive** because the first equity observation is the opening
balance — anything stamped at or before it is already inside that balance and
would be double counted. The end is **inclusive** because the last observation is
the closing balance and must contain everything up to it.

The same window is applied to Binance income totals, fill commissions, fill
slippage and the fill-versus-income commission comparison, so those figures will
**not** match a naive `SUM` over the whole table. The report shows, for both
ledgers: rows included, before, after, and with an unusable timestamp, plus the
excluded income amounts by type and the excluded fill commission and slippage.

### Duplicate records

Exact duplicates of an `income_id` or `fill_id` are counted once. The surviving
row is chosen by the lexicographically smallest field signature, so the result is
identical regardless of database row order.

If the same id appears with **conflicting content**, no row is silently chosen:
`reconciliation_reliable` becomes `false`, an `ERROR` finding `PA-ACC-005` is
raised, and the affected totals are marked untrustworthy.

## Incident attribution

Each incident is measured across `start - 6h` .. `end + 6h`. For an open
incident the end anchor is its start, and `end_anchor_source` says so explicitly.

Endpoints are resolved by deterministic nearest-observation lookup (`bisect`,
ties resolving to the earlier observation) and must land within a **10 minute
freshness tolerance** of the requested boundary — roughly ten missed snapshots at
the ~60 second equity cadence. Reaching further would produce a number that looks
like measurement but describes a different moment.

Each incident reports `status` (`EVALUABLE` / `NOT_EVALUABLE`), the target and
actual timestamps for both endpoints, the distance from each target, the equity
either side, the change, and a `reason` when it cannot be evaluated. No change is
computed when an endpoint is missing or stale, when both endpoints resolve to the
same observation, or when the selected observations do not straddle the incident.
The report prints **not evaluable** rather than `0`.

## Findings

Each finding carries a stable code, scope, evidence, interpretation impact and
remediation, at severity `ERROR` / `WARNING` / `INFO`.

There is deliberately **no single health score**: a scalar would average away the
specific weaknesses that limit interpretation.

Codes are stable across releases: `PA-ENV-*` (environment), `PA-STAT-*`
(statistical validity), `PA-DATA-*` (data completeness), `PA-ACC-*` (accounting),
`PA-OPS-*` (operational), `PA-SHADOW-*` (shadow evaluation), `PA-SCHEMA-*`
(schema).

### Provenance is proven, never assumed

An export archive carries the sanitized environment file and the source commit. A
bare `.db` usually does not. When the audit cannot establish the Binance
environment, `DEMO_ONLY`, and the source commit, it raises a prominent
`PA-ENV-002 — Trading environment provenance is unknown` and the report states
that it cannot tell whether the database represents testnet, live trading,
several runs mixed together, or a copy of another account.

**The environment is never inferred from table contents.** Without proof, no
testnet claim is made and nothing is presented as live. Prefer auditing the full
archive so provenance travels with the data.

## Why testnet is not evidence of live profitability

Testnet has its own matching engine, its own liquidity and its own funding. There
is no competition for queue position and no market impact, so maker fills that
would miss in production fill freely. Funding and mark prices track production
only loosely. A testnet result validates plumbing — order lifecycle, risk
controls, accounting, telemetry — and nothing about edge.

## Why OHLCV is required before comparing strategies

Every shadow model records the portfolio it *would* have held. Turning that into
a return requires the prices it would have been marked and filled at. Without
them, the honest answer is `NOT_EVALUABLE`, which is what the tool reports.

To evaluate any counterfactual strategy, the next dataset must provide:

1. **Timestamped OHLCV** for every traded symbol, covering the full window at or
   finer than the rebalance interval.
2. **The execution/rebalance timing convention** — which bar close forms a
   target, and at which subsequent price it would have filled.
3. **A transaction-cost model** — maker/taker fees, expected spread and slippage
   per symbol and order size.
4. **Funding timestamps and realized rates** per symbol.
5. **A delisting / halted-symbol / missing-candle policy.**

Without all five, a model comparison measures the assumptions, not the strategies.

## How the next phase consumes `summary.json`

`summary.json` is versioned by `schema_version` and split into:

- `run` — everything that varies per run (wall-clock time, input path). Never
  read this for analysis.
- `analysis` — fully deterministic for identical input bytes. Same input, same
  bytes out.

Phase 3 should:

1. Assert `schema_version` is compatible before reading anything.
2. Read `analysis.findings` **first** and refuse to proceed on any unresolved
   `ERROR`, or on `PA-DATA-002` (incomplete lifecycle) if it intends to use trade
   statistics.
3. Use `analysis.equity` as the baseline curve to beat, `analysis.regimes` to
   split the sample at restart boundaries, and `analysis.context.symbols` as the
   symbol universe for the OHLCV fetch.
4. Treat `analysis.shadow.pnl_evaluation.missing_data` as the literal
   requirements list for the dataset it must build.
