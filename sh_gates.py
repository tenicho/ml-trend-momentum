"""
sh_gates.py — the pre-registered acceptance checks, printed after every run.

Port of the v10/v11 gate block. These are written down BEFORE the run that tests them, and the
wording is deliberately unflattering: each one states what a failure would mean, so a bad result
is legible rather than something you have to notice.

  GATE 1  rank-IC        does the model's ORDERING of stocks match the ordering they delivered?
  GATE 2  decile ladder  does the whole return distribution shift right with the score?
  GATE 3  calibration    is the score scaled like the thing it predicts?
  GATE 4  vol tilt       is it just buying volatility? (the v7 trap)
  SWEEP   selectivity    does demanding a higher score keep paying, out to where the book trades?
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def report(oos: pd.DataFrame, H: int, label_kind: str, verbose: bool = True) -> dict:
    o = oos.copy()
    o["vol_q"] = o.groupby("date")["sig"].rank(pct=True)

    # GATE 1 — per-date rank-IC, then averaged, so a few huge days cannot carry it.
    # Columns are selected BEFORE apply rather than passing include_groups=False. The two are
    # equivalent -- both keep the grouping key out of the lambda's frame -- but include_groups only
    # exists in pandas >= 2.2, and this env is on 2.1.4. Selecting columns works on both.
    per_date = o.groupby("date")[["pred", "fwd"]].apply(
        lambda d: d["pred"].corr(d["fwd"], method="spearman")).dropna()
    ic, ic_sd = per_date.mean(), per_date.std()
    t_ic = ic / (ic_sd / np.sqrt(len(per_date))) if ic_sd > 0 else np.nan
    ic_net = o.groupby("date")[["pred", "net"]].apply(
        lambda d: d["pred"].corr(d["net"], method="spearman")).dropna().mean()
    by_fold = o.groupby("fold")[["pred", "fwd"]].apply(
        lambda d: d["pred"].corr(d["fwd"], method="spearman"))

    # GATE 2 — deciles formed PER DATE, never pooled: pooling lets a volatile era dominate the top
    # bucket and manufactures a spread that is really a time effect.
    o["dec"] = o.groupby("date")["pred"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 10, labels=False, duplicates="drop"))
    dec = o.groupby("dec").agg(gross=("fwd", "mean"), net=("net", "mean"),
                               hit=("net", lambda s: (s > 0).mean()),
                               volq=("vol_q", "mean"), advr=("advr", "mean"), n=("net", "size"))
    dec[["gross", "net", "hit"]] *= 100
    d1, d10 = dec["gross"].iloc[0], dec["gross"].iloc[-1]
    mono = pd.Series(dec["gross"].to_numpy()).corr(
        pd.Series(np.arange(len(dec))), method="spearman")
    u_shape = d1 > dec["gross"].iloc[1:-1].min()

    m = np.isfinite(o["pred"]) & np.isfinite(o["y"])
    slope = np.polyfit(o.loc[m, "pred"], o.loc[m, "y"], 1)[0] if m.sum() > 10 else np.nan
    top_volq = dec["volq"].iloc[-1]

    if verbose:
        print(f"\n  ── gates · H={H} label={label_kind} ({len(o):,} OOS rows, "
              f"{pd.Timestamp(o.date.min()).date()}..{pd.Timestamp(o.date.max()).date()}) ──")
        print(f"  GATE 1  rank-IC vs GROSS: {ic:+.4f}  (day-to-day sd {ic_sd:.4f} over "
              f"{len(per_date):,} dates, t={t_ic:.1f})  [vs net {ic_net:+.4f}]")
        print(f"          by fold: [{', '.join(f'{v:+.3f}' for v in by_fold)}]")
        neg = int((by_fold < 0).sum())
        print(f"          {'every fold positive' if neg == 0 else f'⚠️  {neg} FOLD(S) NEGATIVE'} "
              f"— all-positive means it is not one lucky era")
        print(f"  GATE 2  {H}-day return by predicted decile, D1 (worst) → D10 (best):")
        print("    gross% " + "  ".join(f"{v:+.2f}" for v in dec["gross"]))
        print("      net% " + "  ".join(f"{v:+.2f}" for v in dec["net"]))
        print(f"          D1 {d1:+.2f}  D10 {d10:+.2f}  spread {d10-d1:+.2f}pp | "
              f"rank corr {mono:+.3f} | monotone-UP {'YES ✅' if d10 > d1 else 'NO ❌'} | "
              f"U-shape {'YES ⚠️' if u_shape else 'no ✅'}")
        print(f"  GATE 3  calibration slope (realized on predicted): {slope:.3f}  (1.0 = well scaled)")
        print(f"  GATE 4  vol-tilt: mean vol-quantile of D10 = {top_volq:.3f}  "
              f"({'LOW-vol tilt ⚠️' if top_volq < 0.45 else 'ok ✅'})")
        print(f"          D10: ADV pctile {dec['advr'].iloc[-1]:.3f} | net hit rate "
              f"{dec['hit'].iloc[-1]:.1f}%  ({int(dec['n'].iloc[-1]):,} rows)")
        sweep(o, H)

    return dict(H=H, label=label_kind, ic=ic, t_ic=t_ic, ic_min_fold=by_fold.min(),
                neg_folds=int((by_fold < 0).sum()), d1=d1, d10=d10, spread=d10 - d1,
                mono_corr=mono, u_shape=u_shape, slope=slope, top_volq=top_volq, rows=len(o))


def sweep(o: pd.DataFrame, H: int,
          quantiles=(0.0, 0.50, 0.75, 0.90, 0.95, 0.98, 0.99, 0.995, 0.999)) -> pd.DataFrame:
    """Does the payoff keep rising as we demand a higher score?

    A model can rank well on average and still be untradeable, if its very best picks are no better
    than its merely-good ones. That is where v8 died — its top 1% was worse than its top 10%, so no
    threshold was worth trading. Every row here should beat the one above it.

    ⚠️ `t_gross` is OPTIMISTIC and shown for within-table comparison only. With H-day holds the
    observations overlap, so the true standard error is larger by roughly sqrt(H).
    """
    nd = o["date"].nunique()
    pp = o.groupby("date")["pred"].rank(pct=True)
    cyc = 252.0 / H
    rows = []
    for q in quantiles:
        sel = o[pp >= q] if q > 0 else o
        if len(sel) < 200:
            continue
        g, n = sel["fwd"].to_numpy(), sel["net"].to_numpy()
        rows.append({
            "select": "ALL" if q == 0 else f"top {(1-q)*100:g}%",
            "picks/day": len(sel) / nd,
            "gross%": g.mean() * 100, "net%": n.mean() * 100,
            "net_ann%": ((1 + n.mean()) ** cyc - 1) * 100,
            "cost bp": sel["cost"].mean() * 1e4,
            "hit%": (n > 0).mean() * 100,
            "med%": np.median(n) * 100,
            "t_gross": g.mean() / (g.std(ddof=1) / np.sqrt(len(g))),
            "ADVpct": sel["advr"].mean(),
        })
    t = pd.DataFrame(rows)
    print(f"\n  ── SELECTIVITY SWEEP · H={H} ({cyc:.1f} holds/yr) ──")
    print("  " + t.round(2).to_string(index=False).replace("\n", "\n  "))
    inc = t["gross%"].is_monotonic_increasing
    print(f"  gross rises monotonically with selectivity: "
          f"{'YES ✅' if inc else 'NO ⚠️  — the model does not know its own best picks'}")
    print("  ⚠️  net_ann% is a per-SIGNAL return compounded, NOT a portfolio return. Dedup and")
    print("      equal-weight compounding both cut it. The tradeable number comes from the backtest.")
    return t
