# Cross-Sectional Forward-Return Model on US Equities

A random forest ranks US equities by predicted **40-day** forward return, and the top 1% of that
ranking is traded as a concentrated long book with an optional short hedge. Built on Sharadar data
covering 1998-2026, evaluated walk-forward with a 63-session embargo, and tested on a 2023-2026
holdout.

Every feature is derived from daily OHLCV, plus VIX and two S&P-derived series. No fundamentals, no
analyst data, no alternative data.

**Headline result: the ranking works; the portfolio has not clearly beaten the index risk-adjusted.**
Rank-IC is **+0.0487** in development (Newey–West t = 5.58, corrected for the 40-day label overlap)
and **+0.044** out of sample (t = 2.30). The book returned +15.7%/yr against SPY's +8.8% in
development and +37.4% against +22.6% in the holdout — but out of sample its excess Sharpe of 1.17
sits between SPY's 1.18 and the MTUM momentum ETF's 1.12, a three-way tie.

It does comfortably beat a naive momentum sort. Ranking on 12-1 momentum alone, with identical
universe, costs and schedule, returned **+0.25%/yr** against the model's +15.9%.

**Full write-up: [`report/RESEARCH_SUMMARY.md`](report/RESEARCH_SUMMARY.md).**

---

## Layout

```
sh_scan.py       builds/loads the feature panel, universe eligibility, benchmarks, market context
sh_regress.py    the model — RF/XGBoost/LightGBM, walk-forward with embargo, plus --holdout
sh_gates.py      pre-registered acceptance checks (rank-IC, decile ladder, calibration, vol-tilt)
sh_book.py       turns OOS predictions into portfolios: continuous, monthly batches, hedges
sh_analysis.py   performance metrics and return-distribution tables

notebooks/       six analysis notebooks (see below)
report/          RESEARCH_SUMMARY.md — the write-up — plus master_results.csv
report/figures/  publication figures at 200 dpi, written by the notebooks
docs/            statistics reference and terminology glossary
data_sharadar/   ~9 GB panel and derived artifacts (gitignored, regenerable)
_archive/        superseded FMP-era pipeline, prior findings (gitignored)
```

## Notebooks

Read in this order.

| notebook | what it establishes |
|---|---|
| `sharadar_results_v11.ipynb` | the model works in development — gates, books, bootstrap CIs, selection bar |
| `comparing_models.ipynb` | Random Forest beats XGBoost and LightGBM, most decisively in the traded tail |
| `momentum_control.ipynb` | it is not simply momentum — vs a naive 12-1 sort and vs MTUM |
| `monte_carlo_v11.ipynb` | ruin risk by horizon, drawdown pain, and how noisy a short measurement window is |
| `holdout_v11.ipynb` | 2023-2026, out of sample |
| `report_summary_v11.ipynb` | master results table and the curated figure set |

Notebooks resolve the repo root themselves, so they run from either `notebooks/` or the root.

## Reproducing

```bash
python sh_scan.py --build                                     # ~5 min, builds the panel
python sh_regress.py --smoke --h 40 --adv-floor 25e6          # cheap sanity check first
python sh_regress.py --h 40 --adv-floor 25e6                  # ~21 min -> OOS predictions
python sh_book.py --signal sh_oos_H40_cz_adv25.parquet --h 40 --adv-floor 25e6 \
    --thr 0.99 --tag h40_top1
```

Model comparison: `--model xgb` or `--model lgbm` (~4 min each). Training-row cap: `--train-subsmp`,
default 500,000 — a full six-fold run at 1M came out marginally worse and was rejected.

The holdout:

```bash
python sh_regress.py --h 40 --adv-floor 25e6 --holdout
python sh_book.py --signal sh_oos_H40_cz_adv25_holdout.parquet --h 40 --adv-floor 25e6 \
    --thr 0.99 --tag holdout_h40_top1
```

Run the smoke test after touching `sh_regress.py` or `sh_scan.py`. It has caught real bugs cheaply
before a 20-minute run hit them.

## Conventions that matter

**Train gross, evaluate net.** The label is the gross forward return. Cost enters only at selection
and evaluation. A cost-adjusted label would inject an anti-volatility tilt, because cost correlates
with volatility.

**No volume level anywhere.** Split-adjusted volume leaks forward returns: the adjustment inflates
the historical volume of companies that split later, and companies split after their price has run
up. Measured on this panel, within-date rank correlation against forward 5-year return is **+0.17**
for adjusted volume against **+0.06** for split-invariant `dollar_volume`, so roughly **+0.11** is
artifact — about twice the size of the model's real edge. Only scale-invariant transforms are used
(ratios and z-scores, where the adjustment cancels).

**Sharpe and Sortino are excess of Rf = 3.75%.** Rf is subtracted from the numerator but each book is
divided by its own volatility, so the penalty is `Rf / vol` — low-volatility books are penalised about
twice as hard, and it reorders results relative to raw Sharpe.

**Sortino divides downside deviation by the full sample size**, not by the count of negative days.
Dividing by the negative count is the common error and inflates the ratio.

**Significance is corrected for overlapping labels.** Consecutive days share 39 of their 40 forward
days, so a naive t is badly inflated. Newey–West at 40 lags takes the development rank-IC t from a
naive 24.0 to **5.58**, an effective sample of 228 independent observations rather than 4,228.

**Benchmarks are built with the same machinery as the books.** `sp500_ew` in particular is *not* a
naive daily cross-sectional mean, which implicitly rebalances 500 names daily at zero cost and
harvests the bid-ask bounce. That artifact was worth roughly 10pp/yr on an earlier build.

## Environment

Python 3.11.8, pandas 2.1.4, scikit-learn 1.2.2, xgboost 2.1.4, lightgbm 4.5.0, plus `bidask` for the
cost model's spread calibration.

⚠️ Two pandas-version traps were hit here and are guarded in the code: `include_groups` (needs ≥ 2.2)
and the `"M"` → `"ME"` offset rename. `Series.unique()` also returns `Timestamp` on 2.1.x and
`datetime64` on newer versions, and the two hash differently — `sh_book._dt64_unique` normalises it.

## Memory

The feature panel is ~4.5 GB and this was developed on a 16 GB machine. `sh_regress.py` runs a
preflight check and refuses to start below `MIN_FREE_GB = 3.5`, which is deliberate: an earlier run
exhausted physical memory and took the machine down rather than being OOM-killed. Peak resident is
~4 GB. Do not run two panel-loading processes at once.
