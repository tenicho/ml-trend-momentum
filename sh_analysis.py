"""
sh_analysis.py — performance metrics and return-distribution analysis for the Sharadar run.

Port of sc_analysis_v11.py's metrics block, unchanged — these formulas were already reviewed and
are data-source agnostic (they take a daily net-return series, nothing about FMP or Sharadar).
Copied rather than imported so the Sharadar directory has no dependency on the FMP-era modules.

⚠️ TRAIN GROSS, EVALUATE NET. The model's label is the GROSS forward return; cost is never in the
label. Cost is applied here, at evaluation and selection time only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ── performance metrics ───────────────────────────────────────────────────────────────────────
def metrics(net: pd.Series, target: float = 0.0, periods: int = TRADING_DAYS) -> dict:
    """Standard performance ratios from a daily net-return series.

    Definitions, written out because the common implementations differ:

      CAGR     geometric, from the compounded equity curve over the actual elapsed calendar years
      vol      std of daily returns x sqrt(252)
      Sharpe   mean / std x sqrt(252), excess over `target` (0 by default, so this is a raw
               Sharpe, not excess-of-cash — state that when quoting it)
      Sortino  mean / DOWNSIDE DEVIATION x sqrt(252), where

                   downside_deviation = sqrt( mean( min(r - target, 0)^2 ) )

               ⚠️ The mean divides by the FULL sample size, not by the count of negative days.
               Dividing by the count of negatives is the common error and inflates Sortino,
               badly for strategies that are down infrequently.
      maxDD    the worst peak-to-trough decline of the compounded curve
      Calmar   CAGR / |maxDD|
    """
    r = pd.Series(net).dropna()
    if len(r) < 2:
        return {k: np.nan for k in
                ("CAGR", "vol", "Sharpe", "Sortino", "maxDD", "Calmar", "days")}

    eq = (1 + r).cumprod()
    years = (r.index[-1] - r.index[0]).days / 365.25 if isinstance(r.index, pd.DatetimeIndex) \
        else len(r) / periods
    cagr = eq.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan

    sd = r.std()
    excess = r - target
    downside = np.sqrt(np.mean(np.minimum(excess, 0.0) ** 2))      # full-N denominator

    dd = (eq / eq.cummax() - 1).min()

    return {
        "CAGR": cagr,
        "vol": sd * np.sqrt(periods),
        "Sharpe": excess.mean() / sd * np.sqrt(periods) if sd > 0 else np.nan,
        "Sortino": excess.mean() / downside * np.sqrt(periods) if downside > 0 else np.nan,
        "maxDD": dd,
        "Calmar": cagr / abs(dd) if dd < 0 else np.nan,
        "days": len(r),
    }


def metrics_table(curves: dict[str, pd.Series], target: float = 0.0) -> pd.DataFrame:
    """One row per book. `curves` maps name -> daily NET return series (not equity)."""
    return pd.DataFrame([{"book": k, **metrics(v, target)} for k, v in curves.items()])


def drawdown_series(net: pd.Series) -> pd.Series:
    eq = (1 + pd.Series(net).dropna()).cumprod()
    return eq / eq.cummax() - 1


# ── the distribution analysis ─────────────────────────────────────────────────────────────────
def decile_distributions(oos: pd.DataFrame, n_buckets: int = 10,
                         ret_col: str = "fwd", cost_col: str = "cost") -> pd.DataFrame:
    """Forward-return distribution by predicted-score bucket.

    THE QUESTION THIS ANSWERS. A model can produce a positive average and still be untradeable, if
    its confidence does not track its outcomes. What we want is a distribution that shifts steadily
    right as the predicted score rises — not just a higher mean in the top bucket, but the whole
    distribution moving, including the median and the downside.

    ⚠️ Buckets are formed PER DATE, never pooled. Pooling lets a high-volatility era dominate the
    top bucket and manufactures a spread that is really a time effect.
    """
    o = oos.copy()
    o["bucket"] = o.groupby("date")["pred"].transform(
        lambda s: pd.qcut(s.rank(method="first"), n_buckets, labels=False, duplicates="drop"))
    o["net_"] = o[ret_col] - o[cost_col]

    rows = []
    for b, g in o.groupby("bucket"):
        gross, net = g[ret_col], g["net_"]
        rows.append({
            "bucket": int(b) + 1,
            "n": len(g),
            "gross_mean": gross.mean(), "gross_median": gross.median(),
            "net_mean": net.mean(), "net_median": net.median(),
            "net_p10": net.quantile(.10), "net_p25": net.quantile(.25),
            "net_p75": net.quantile(.75), "net_p90": net.quantile(.90),
            "hit_rate": (net > 0).mean(),
            "std": net.std(),
        })
    return pd.DataFrame(rows)


def top_slice_distributions(oos: pd.DataFrame, quantiles=(0.90, 0.95, 0.99, 0.995),
                            ret_col: str = "fwd", cost_col: str = "cost") -> pd.DataFrame:
    """Same idea at the sharp end, where the book actually trades.

    The decile view stops at the top 10%. A book trading the top 1% needs to know the distribution
    keeps improving past that point — this is where v8 died, with a top-1% worse than its top-10%.
    """
    o = oos.copy()
    o["pp"] = o.groupby("date")["pred"].rank(pct=True)
    o["net_"] = o[ret_col] - o[cost_col]

    rows = []
    for q in quantiles:
        s = o[o["pp"] >= q]
        if len(s) < 200:
            continue
        rows.append({
            "select": f"top {(1-q)*100:g}%",
            "n": len(s),
            "gross_mean": s[ret_col].mean(),
            "net_mean": s["net_"].mean(),
            "net_median": s["net_"].median(),
            "net_p10": s["net_"].quantile(.10),
            "net_p90": s["net_"].quantile(.90),
            "hit_rate": (s["net_"] > 0).mean(),
            "cost_bp": s[cost_col].mean() * 1e4,
        })
    return pd.DataFrame(rows)
