"""
sh_book.py — build the equity curves and performance table for the Sharadar run.

Reuses the OOS predictions written by sh_regress.py. Cost, execution mechanics and metrics each
come from one place so every number in the write-up is built the same way.

Books produced (per the v11 plan — no weekly, no tranche-3; keep it to what was asked for)
    continuous          enter every day, fractional daily quota so the average book size is exact
    monthly_tranche2     one rebalance a month, half the book, each half held two months.
                         This IS the "pure long" curve — the unhedged long leg of the scheme used
                         for hedging below.
    + hedged variants    monthly_tranche2 long, with an 80/20 and 60/40 short leg

Benchmarks
    SPY                  the standard reference
    VTI                  total US market
    sp500_ew             equal-weight S&P 500, built the SAME way as the strategy books: enter at
                         the open of t+1, hold h days, cost charged on entry. NOT a naive daily
                         rebalance — that harvests the bid-ask bounce (buy the day's losers, sell
                         the day's winners) and is not a portfolio anyone could actually hold. This
                         cost the FMP build ~10pp/yr of phantom benchmark return (FINDINGS_v11 §-).

⚠️ TRAIN GROSS, EVALUATE NET. The label is the gross forward return; cost enters only here, at
selection and evaluation time.

Run:  python sh_book.py --signal sh_oos_H20_cz_adv25.parquet --tag adv25
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

import sh_analysis as an
import sh_scan

_COST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cost-model")
if os.path.isdir(_COST) and _COST not in sys.path:
    sys.path.insert(0, _COST)

BORROW_DEFAULT = 0.08          # annual, the frozen central case
HEDGE_WEIGHTS = {"80/20": (0.80, 0.20), "60/40": (0.60, 0.40)}
COST_COLS = ["ticker", "date", "open", "high", "low", "close", "close_raw", "dollar_volume"]


def _month_end() -> str:
    """Month-end offset alias. pandas renamed "M" -> "ME" in 2.2 and this env runs 2.1.4, so probe
    rather than hardcode. Both aliases mean month-END, so the rebalance dates are identical."""
    try:
        pd.tseries.frequencies.to_offset("ME")
        return "ME"
    except ValueError:
        return "M"


def _semi_month() -> str:
    """Semi-month-end alias: "SM" on pandas < 2.2, "SME" from 2.2. Probe rather than hardcode."""
    for alias in ("SME", "SM"):
        try:
            pd.tseries.frequencies.to_offset(alias)
            return alias
        except ValueError:
            continue
    raise SystemExit("pandas exposes no semi-month-end offset alias")


MONTH_END = _month_end()
SEMI_MONTH = _semi_month()


def _dt64_unique(s) -> np.ndarray:
    """Sorted unique dates as datetime64[ns].

    ⚠️ Series.unique() returns pandas Timestamps on some pandas versions and numpy datetime64 on
    others, and the two HASH DIFFERENTLY. Mixing them makes dict lookups miss silently-then-loudly
    (a KeyError on a date that is plainly present). Normalise the dtype in one place instead."""
    return np.unique(pd.Series(s).to_numpy("datetime64[ns]"))


# ── context: panel, cost, universe ───────────────────────────────────────────────────────────
def load_context(adv_floor: float = 25e6, verbose: bool = True):
    """Panel, arrays, cost per row, and the eligible mask — everything the books need."""
    import cost_model as cm

    need = ["open", "high", "low", "close", "close_raw", "dollar_volume", "med_dv", "bars"]
    p = sh_scan.load_features(columns=need, verbose=verbose)
    p = p.sort_values(["ticker", "date"]).reset_index(drop=True)

    # sliced to what cost_model touches — see sh_regress.build_context for why this matters (it's
    # the difference between fitting in 16GB and getting OOM-killed on the full feature frame).
    cf = cm.prepare_features(p[COST_COLS], price_col="close_raw")
    model = cm.fit_cost_model(cf, verbose=verbose)
    cost = model.cost_bps(cf["dollar_adv"], cf["sigma_20"], cf["sigma_ref"], cf["price"]) / 1e4

    close, openp, tend = sh_scan._arrays(p)
    dates = p["date"].to_numpy()
    elig = sh_scan.eligible_mask(p, adv_floor=adv_floor, verbose=verbose)
    if verbose:
        print(f"cost model: median {np.nanmedian(cost[elig])*1e4:.1f} bp round-trip "
              f"on the eligible universe")
    return dict(panel=p, close=close, openp=openp, tend=tend, dates=dates,
                cost=cost, elig=elig, model=model)


# ── trade generation (execution mechanics — data-source agnostic) ───────────────────────────────
def trades_continuous(w, tend, dmap, book, H):
    """Enter every day. picks/day is set so the average open book is `book` names.

    ⚠️ The daily quota is usually FRACTIONAL (book=30 over H=20 wants 1.5 names/day) and rounding
    it to an integer would silently overshoot the target book by up to 33%. A carried budget spends
    1 name on some days and 2 on others so the long-run average is exact.
    """
    quota, budget = book / H, 0.0
    op, out = {}, []
    for d, g in w.sort_values(["date", "score"], ascending=[True, False]).groupby("date"):
        op = {t: x for t, x in op.items() if x > d}
        budget += quota
        n, take = 0, int(budget)
        budget -= take
        for rr in g.itertuples():
            if n >= take:
                break
            if rr.ticker in op:
                continue
            gp = int(rr.gpos)
            if gp + 1 > tend[gp]:
                continue
            xi = min(gp + H, tend[gp])
            op[rr.ticker] = dmap[xi]
            out.append((gp, xi))
            n += 1
    return np.array(out, dtype=np.int64).reshape(-1, 2)


def trades_periodic(w, tend, dmap, dates_idx, book, freq=None, tranches=1):
    """Rebalance on a calendar schedule. Each rebalance replaces 1/`tranches` of the book, and
    each tranche is held `tranches` rebalance periods — so the book always contains `tranches`
    vintages while still trading only once per period."""
    all_dates = _dt64_unique(w["date"])
    marks = pd.Series(all_dates, index=pd.DatetimeIndex(all_dates))
    rebal = marks.groupby(pd.Grouper(freq=freq or MONTH_END)).first().dropna().to_numpy()
    per_tranche = max(1, int(round(book / tranches)))
    pos_of = {d: i for i, d in enumerate(dates_idx)}

    op, out = {}, []
    for ri, d in enumerate(rebal):
        op = {t: x for t, x in op.items() if x > d}
        j = min(ri + tranches, len(rebal) - 1)
        exit_date = rebal[j]
        if exit_date <= d:
            continue
        span = pos_of[exit_date] - pos_of[d]
        g = w[w.date == d].sort_values("score", ascending=False)
        n = 0
        for rr in g.itertuples():
            if n >= per_tranche:
                break
            if rr.ticker in op:
                continue
            gp = int(rr.gpos)
            if gp + 1 > tend[gp]:
                continue
            xi = min(gp + span, tend[gp])
            op[rr.ticker] = dmap[xi]
            out.append((gp, xi))
            n += 1
    return np.array(out, dtype=np.int64).reshape(-1, 2)


def _net(tr, cost, close, openp, dmap):
    """Equal-weight daily net return of the open book; round-trip charged on the entry day."""
    if len(tr) == 0:
        return pd.Series(dtype=float)
    gp, xi = tr[:, 0], tr[:, 1]
    D, R, EC = [], [], []
    for g, x, c in zip(gp, xi, cost[gp]):
        ei = g + 1
        if x < ei:
            continue
        ks = np.arange(ei, x + 1)
        rt = close[ks] / np.concatenate(([openp[ei]], close[ks[:-1]])) - 1.0
        e = np.zeros(len(ks)); e[0] = c
        D.append(dmap[ks]); R.append(rt); EC.append(e)
    d = pd.DataFrame({"date": np.concatenate(D), "ret": np.concatenate(R),
                      "ec": np.concatenate(EC)})
    g_ = d.groupby("date")
    n = g_.size()
    return (g_["ret"].mean() - g_["ec"].sum() / n).sort_index()


# ── books ─────────────────────────────────────────────────────────────────────────────────────
def _legs(w, thr):
    """Top-`thr` long candidates and bottom-(1-thr) short candidates, scored for the builders."""
    w = w.copy()
    w["pp"] = w.groupby("date")["pred"].rank(pct=True)
    lo = w[w.pp >= thr].copy();  lo["score"] = lo["pp"]
    sh = w[w.pp <= 1 - thr].copy(); sh["score"] = -sh["pp"]      # most negative first
    return lo, sh


def build_books(signal: str, h: int = 20, book: int = 30, thr: float = 0.99,
                ctx=None, verbose: bool = True):
    """Returns (curves, legs, ctx). `curves['monthly_tranche2']` IS the pure-long book."""
    ctx = ctx or load_context(verbose=verbose)
    close, openp, tend = ctx["close"], ctx["openp"], ctx["tend"]
    dates, cost = ctx["dates"], ctx["cost"]

    w = pd.read_parquet(os.path.join(sh_scan.DATA, signal),
                        columns=["date", "ticker", "gpos", "pred"])
    tk = ctx["panel"]["ticker"].to_numpy()
    nchk = min(200_000, len(w))
    chk = tk[w.gpos.to_numpy()[:nchk]] == w.ticker.to_numpy()[:nchk]
    if not chk.all():
        raise SystemExit(f"gpos MISALIGNED — {(~chk).sum():,}/{nchk:,} ticker mismatches. "
                         "The panel sort differs from the training script; results are void.")
    if verbose:
        print(f"✓ gpos alignment verified on {nchk:,} rows   |   {signal} ({len(w):,} rows)")

    dates_idx = _dt64_unique(w["date"])
    dmap = dates
    lo, sh = _legs(w, thr)
    if verbose:
        print(f"\n{signal}: {len(lo):,} long candidates, {len(sh):,} short candidates "
              f"(top {(1-thr)*100:g}%)")

    schemes = {
        "continuous":       lambda s: trades_continuous(s, tend, dmap, book, h),
        "monthly_tranche2": lambda s: trades_periodic(s, tend, dmap, dates_idx, book,
                                                      MONTH_END, 2),
        # Same two-overlapping-tranche structure, but rebalanced twice a month so each tranche is
        # held ~21 sessions instead of ~42. The point is to hold for the horizon the model actually
        # forecasts (H=20) while keeping the book size unchanged -- monthly_tranche2 spends its
        # second month outside the forecast window. Turnover roughly doubles as a result.
        "semi_tranche2":    lambda s: trades_periodic(s, tend, dmap, dates_idx, book,
                                                      SEMI_MONTH, 2),
    }

    curves, legs = {}, {}
    for name, fn in schemes.items():
        tr = fn(lo)
        net = _net(tr, cost, close, openp, dmap)
        curves[name] = net
        legs[name] = {"long": net}
        if verbose:
            print(f"  {name:<18}{len(tr):>7,} trades")

    for name, fn in schemes.items():
        tr = fn(sh)
        if len(tr):
            legs[name]["short"] = _net(tr, cost, close, openp, dmap)

    curves["pure_long"] = curves["monthly_tranche2"]
    return curves, legs, ctx


def hedged(legs: dict, scheme: str, borrow: float = BORROW_DEFAULT) -> dict:
    """Long/short combinations of one scheme. Borrow is charged daily on the short weight."""
    out = {}
    L = legs[scheme].get("long")
    S = legs[scheme].get("short")
    if L is None or S is None:
        return out
    idx = L.index.intersection(S.index)
    L, S = L.reindex(idx), S.reindex(idx)
    for lab, (wl, ws) in HEDGE_WEIGHTS.items():
        out[f"{scheme} {lab} hedge"] = wl * L - ws * (S + borrow / 252.0)
    return out


# ── benchmarks ────────────────────────────────────────────────────────────────────────────────
def benchmarks(ctx, index, h: int = 20) -> dict:
    """SPY, VTI (total US market), and equal-weight S&P 500 built the SAME way as the books.

    ⚠️ THE OBVIOUS EQUAL-WEIGHT BENCHMARK IS WRONG. Taking the daily cross-sectional mean return of
    every S&P 500 member looks like the natural comparison. It is not: it implicitly rebalances 500
    names EVERY DAY at zero cost, which harvests the bid-ask bounce (buy the day's losers, sell the
    day's winners) — an artifact worth ~10pp/yr on the FMP build (own-the-pond: 19.93%/yr @ Sharpe
    0.98 built that way vs 9.61% built properly). So this is built with the same machinery as the
    strategy books: enter at the open of t+1, hold `h` days, equal weight, same cost model on entry.
    """
    px = sh_scan.benchmarks(tickers=("SPY", "VTI"))
    out = {}
    for t in ("SPY", "VTI"):
        if t in px.columns:
            out[t] = px[t].pct_change(fill_method=None).reindex(index).dropna()

    mem = sh_scan.sp500_membership()
    panel = ctx["panel"][["date", "ticker"]].reset_index().rename(columns={"index": "gpos"})
    mm = mem.merge(panel, on=["date", "ticker"], how="inner")
    gp = mm["gpos"].to_numpy()
    gp = gp[np.isfinite(ctx["cost"][gp])]
    gp = np.sort(gp)
    gp = gp[::h]                                  # ~one entry per name per holding period
    xi = np.minimum(gp + h, ctx["tend"][gp])
    tr = np.column_stack([gp, xi])
    net = _net(tr, ctx["cost"], ctx["close"], ctx["openp"], ctx["dates"])
    out["sp500_ew"] = net.reindex(index).dropna()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--signal", required=True)
    ap.add_argument("--h", type=int, default=20)
    ap.add_argument("--book", type=int, default=30)
    ap.add_argument("--thr", type=float, default=0.99)
    ap.add_argument("--adv-floor", type=float, default=25e6)
    ap.add_argument("--borrow", type=float, default=BORROW_DEFAULT)
    ap.add_argument("--hedge-on", default="monthly_tranche2")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    ctx = load_context(args.adv_floor)
    curves, legs, ctx = build_books(args.signal, args.h, args.book, args.thr, ctx=ctx)
    curves.update(hedged(legs, args.hedge_on, args.borrow))
    idx = curves["continuous"].index
    curves.update(benchmarks(ctx, idx, args.h))

    t = an.metrics_table(curves)
    print("\n" + "=" * 104)
    print(f"BOOKS · H={args.h} · target book {args.book} · top {(1-args.thr)*100:g}% · "
          f"borrow {args.borrow:.0%}")
    print("=" * 104)
    show = t.assign(CAGR=t.CAGR.map("{:+.2%}".format), vol=t.vol.map("{:.1%}".format),
                    Sharpe=t.Sharpe.round(2), Sortino=t.Sortino.round(2),
                    maxDD=t.maxDD.map("{:.1%}".format), Calmar=t.Calmar.round(2))
    print(show.to_string(index=False))

    out = os.path.join(sh_scan.DATA, f"sh_books_{args.tag}.parquet")
    pd.DataFrame({k: (1 + v).cumprod() for k, v in curves.items()}).to_parquet(out)
    t.to_parquet(os.path.join(sh_scan.DATA, f"sh_books_{args.tag}_metrics.parquet"), index=False)
    print(f"\n💾 {out} + metrics")


if __name__ == "__main__":
    main()
