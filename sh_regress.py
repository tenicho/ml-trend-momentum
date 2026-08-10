"""
sh_regress.py — cross-sectional forward-return regression on the Sharadar panel.

Port of sc_regress_v11 (FMP). The model is unchanged: a random forest on per-date cross-sectionally
ranked features, walk-forward with an embargo, trained on the GROSS executable return. What changes
is the data underneath and three things that follow from it.

 1. NO PRICE RECONSTRUCTION. Sharadar ships `close_raw`, so the point-in-time price is a column
    rather than a rebuild. Price LEVELS use it; returns and indicators use adjusted `close`.

 2. NO VOLUME LEVEL, ANYWHERE. Split-adjusted volume leaks forward returns: companies that split
    later are companies that went up, and the adjustment inflates their earlier volume. Measured on
    this panel (within-date Spearman vs forward 5y return): adjusted volume +0.17, split-invariant
    dollar_volume +0.06, so ~+0.11 is the split artifact. Adjusted volume also correlates −0.27 with
    the inverse forward-5y split factor, i.e. it predicts splits that have not happened yet. Only
    ratio transforms survive (vol_ratio, vol_z, updn_vol, cmf20), where the split cancels.
    ⚠️ An earlier version of this note claimed −0.37 to −0.46. That sign was wrong; the relationship
    is positive, as the mechanism implies. Corrected 2026-08-07 by direct measurement.

 3. SIGNED DOLLAR FLOW REPLACES OBV. `flow_10` / `flow_63` are (up-day dollars − down-day dollars)
    ÷ all dollars over the window, built on dollar_volume so they are split-invariant and bounded
    in [−1, +1]. The OBV they replace was unusable for exactly the reason in (2).

 ⚠️ THE BUCKET-REGIME BLOCK IS NOT PORTED. It existed to test whether the model could rotate across
 liquidity buckets; that thesis was refuted on the FMP data (gate 2 failed on every execution
 scheme, and the premise test showed no persistence in bucket leadership). Re-adding it is cheap if
 wanted, but it would be carrying a dead hypothesis forward.

Discipline unchanged: walk-forward, 63-session embargo ≥ H, the forest only ever trains on the past.
DEV ends 2022-12-31; 2023-07 onward is held back.

Run:  python sh_regress.py --smoke
      python sh_regress.py --h 20 --adv-floor 25e6
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

import sh_scan


# 16GB machine, ~4.5GB feature panel. Every full-frame copy used to be a 4.5GB spike, and two
# overlapping spikes exhausted physical memory -- which took the whole machine down rather than
# OOM-killing the process. Report RSS per stage so peaks stay visible, and refuse to start with no
# headroom.
#
# The gate started at 5.0 GB, set before the copy-elimination fixes below were in place. Measured
# peak RSS across full runs since then is ~2.3 GB (see the [mem] lines in data_sharadar/run_*.log),
# so 3.5 GB leaves ~50% headroom over the observed worst case. Raise it again if the panel grows.
MIN_FREE_GB = 3.5


def _mem(label="", verbose=True):
    if not verbose:
        return
    try:
        import psutil
        print(f"  [mem] {label:<26s} rss {psutil.Process().memory_info().rss/(1<<30):5.2f} GB"
              f" | available {psutil.virtual_memory().available/(1<<30):5.2f} GB", flush=True)
    except Exception:
        pass


def preflight(min_free_gb=MIN_FREE_GB):
    # A hard gate. Better to refuse the run than to exhaust physical memory and lock the machine.
    try:
        import psutil
    except Exception:
        print("[preflight] psutil unavailable -- memory gate skipped")
        return
    avail = psutil.virtual_memory().available / (1 << 30)
    if avail < min_free_gb:
        raise SystemExit(
            f"\n! ABORTED: {avail:.1f} GB available, need >= {min_free_gb:.1f} GB.\n"
            f"  The panel alone is ~4.5 GB. Continuing risks exhausting physical memory and\n"
            f"  hard-locking the machine. Close other apps and retry.\n")
    print(f"[preflight] {avail:.1f} GB available -- ok")

_COST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cost-model")
if os.path.isdir(_COST) and _COST not in sys.path:
    sys.path.insert(0, _COST)

DEV_END = pd.Timestamp("2022-12-31")
EMBARGO = 63
N_FOLDS = 6
TRAIN_SUBSMP = 500_000
WINSOR_Z = 4.0
PRIMARY_H = 20
RF_KW = dict(max_features=0.33, n_jobs=-1, random_state=0)
# colsample_bytree=0.33 mirrors RF's max_features=0.33; depth/lr are plain defaults, not tuned —
# this is a fair-ish architecture comparison (bagging vs boosting), not a hyperparameter bake-off.
XGB_KW = dict(max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.33,
              n_jobs=-1, random_state=0, tree_method="hist")
# LightGBM, matched to XGB_KW knob for knob. num_leaves=31 is the largest value consistent with
# max_depth=5 (2^5 - 1), so the two boosters get the same capacity ceiling.
# ⚠️ subsample is SILENTLY IGNORED by LightGBM unless subsample_freq >= 1. Omitting subsample_freq
# is the standard way this comparison gets rigged in boosting's favour without anyone noticing.
LGBM_KW = dict(num_leaves=31, max_depth=5, learning_rate=0.05, subsample=0.8, subsample_freq=1,
               colsample_bytree=0.33, n_jobs=-1, random_state=0, verbosity=-1)
LEAF_FRAC, LEAF_MIN = 0.0004, 25

# ── features ──────────────────────────────────────────────────────────────────────────────────
TREND = ["dist_sma200", "dist_sma50", "dist_sma20", "adx14", "macd_dist",
         "donch_pos", "linreg30", "kumo_dist", "ma_ribbon"]
OSC = ["rsi14", "roc10", "stoch_k14", "mfi14", "aroon_osc25", "cmo20", "rvm20"]
MOM = ["ret_20", "ret_12_1", "mom_3_1", "mom_6_1", "pct_52wk_high", "fip", "mom_consist"]
VOL_BLOCK = ["ewm_sig", "vol20", "vol60", "gk_vol", "vol_regime", "atr14"]
VOLUME_RATIOS = ["vol_ratio", "vol_z", "updn_vol", "cmf20"]      # ratios only — never a level
FLOW = ["flow_10", "flow_63"]                                     # the OBV replacement
LIQUIDITY = ["med_dv", "dollar_volume"]
LEVEL = ["close_raw"]                                             # the real traded price
OTHER = ["gap"]

XSEC_ALL = TREND + OSC + MOM + VOL_BLOCK + VOLUME_RATIOS + FLOW + LIQUIDITY + LEVEL + OTHER
XSEC_NO_VOL = [c for c in XSEC_ALL if c not in VOL_BLOCK]
RAW_FEATS = ["vix", "spy_dist200", "spy_5dvol"]   # market-wide → constant per date → NEVER ranked

FEATURE_SETS = {"all": XSEC_ALL, "no_vol": XSEC_NO_VOL}


def leaf_size(n, override=None):
    return int(override) if override else max(LEAF_MIN, int(round(LEAF_FRAC * n)))


def make_label(kind, fwd, dates):
    """Target from the GROSS forward return. Cost is applied at selection, never in the label —
    cost correlates with volatility, so a cost-adjusted label injects an anti-volatility tilt."""
    s = pd.Series(fwd)
    if kind == "raw":
        return fwd.astype(np.float64)
    g = s.groupby(dates)
    if kind == "rank":
        return g.rank(pct=True).to_numpy() * 2.0 - 1.0
    if kind == "cz":
        mu, sd = g.transform("mean").to_numpy(), g.transform("std").to_numpy()
        with np.errstate(invalid="ignore", divide="ignore"):
            z = (fwd - mu) / sd
        return np.clip(z, -WINSOR_Z, WINSOR_Z)
    raise ValueError(kind)


def assemble_features(panel, mask, xsec, raw, winsor=(0.005, 0.995), verbose=False,
                      fit_end=None):
    """Per-date cross-sectional percentile rank for per-ticker columns; raw+winsorized for
    market-wide ones.

    ⚠️ Market-wide series are CONSTANT within a date. Ranking them ties every row to 0.5 and
    silently deletes them — the one place the rank-everything rule must not apply."""
    xs = [c for c in xsec if c in panel.columns]
    rw = [c for c in raw if c in panel.columns]
    missing = [c for c in list(xsec) + list(raw) if c not in panel.columns]
    if missing:
        print(f"  ⚠️  absent from panel: {missing}")

    sub = panel.loc[mask, ["date"] + xs + rw]
    X = pd.DataFrame(index=sub.index)
    g = sub.groupby("date", sort=False)
    # Cast per column. rank(pct=True) returns float64, so building all 36 columns first and casting
    # at the end held a ~2GB float64 frame alongside its float32 copy. Bit-identical, half the peak.
    for c in xs:
        X[c + "_pct"] = g[c].rank(pct=True).astype(np.float32)
    # Winsorisation bounds are a FULL-SAMPLE statistic unless restricted. Computing them over all
    # dates lets the holdout period influence the clip levels applied to training features -- a
    # small leak, but a leak. With fit_end set, the bounds come from the fitting window only.
    wmask = slice(None) if fit_end is None else (sub["date"] <= np.datetime64(fit_end))
    if verbose and fit_end is not None:
        n_fit = int(np.sum(wmask)) if fit_end is not None else len(sub)
        print(f"  winsor bounds from {n_fit:,}/{len(sub):,} rows (<= {pd.Timestamp(fit_end).date()})")
    for c in rw:
        lo, hi = sub.loc[wmask, c].quantile(winsor)
        X[c] = sub[c].clip(lo, hi).astype(np.float32)
    del sub, g
    gc.collect()
    X = X.astype("float32")   # block consolidation now, not a dtype conversion
    if verbose:
        nan = X.isna().mean().sort_values(ascending=False)
        print(f"  features: {X.shape[1]} ({len(xs)} ranked + {len(rw)} market-wide) × "
              f"{X.shape[0]:,} rows | top-NaN: {dict(nan.head(3).round(3))}")
    return X, list(X.columns)


def forest_mean_std(rf, X, n_jobs=-1):
    """Per-row mean and std across trees. Threaded — a plain loop over estimators does not inherit
    the forest's n_jobs and runs single-threaded."""
    from joblib import Parallel, delayed
    parts = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(e.predict)(X) for e in rf.estimators_)
    k = len(parts)
    s = np.zeros(len(X)); ss = np.zeros(len(X))
    for p in parts:
        s += p; ss += p * p
    m = s / k
    return m, np.sqrt(np.maximum(ss / k - m * m, 0.0))


def fit_predict(model_kind, trees, Xtr, ytr, Xte):
    """Dispatch on model architecture. Returns (pred_mean, pred_sd).

    RF gives a real per-row sd across its (bagged, independent) trees. Boosted trees (XGBoost,
    LightGBM) are sequential residual fits, not independent draws, so there is no analogous
    per-tree spread — `pred_sd` comes back as zeros for both. It is an unused diagnostic column
    downstream (never read by sh_gates or the backtest), so this is a deliberate no-op, not a
    missing feature.
    """
    if model_kind == "rf":
        ml = leaf_size(len(Xtr))
        rf = RandomForestRegressor(n_estimators=trees, min_samples_leaf=ml, **RF_KW)
        rf.fit(Xtr, ytr)
        return forest_mean_std(rf, Xte)
    if model_kind == "xgb":
        from xgboost import XGBRegressor
        xgb = XGBRegressor(n_estimators=trees, **XGB_KW)
        xgb.fit(Xtr, ytr)
        return xgb.predict(Xte).astype(np.float64), np.zeros(len(Xte))
    if model_kind == "lgbm":
        from lightgbm import LGBMRegressor
        gbm = LGBMRegressor(n_estimators=trees, **LGBM_KW)
        gbm.fit(Xtr, ytr)
        return gbm.predict(Xte).astype(np.float64), np.zeros(len(Xte))
    raise ValueError(model_kind)


def run_holdout(H, label_kind, ctx, trees=200, model_kind="rf", verbose=True):
    """THE SINGLE HOLDOUT LOOK. One fit, no folds.

    Train on everything up to DEV_END minus the embargo, then predict every eligible row AFTER
    DEV_END. That embargo is not optional: a training row dated 2022-12-15 carries a 20-day forward
    label measured into January 2023, which is inside the holdout. Cutting 63 sessions before the
    holdout's first date removes any label that reaches across the boundary, at the cost of giving
    up the last quarter of DEV as training data.

    Everything else is identical to the frozen walk-forward config, deliberately. Nothing here is
    tuned, selected, or re-run.
    """
    X, gpos, close, openp, tend, dates, tick, cost, sig, mdv, advr = ctx

    tgt = np.minimum(gpos + H, tend[gpos])
    ent = gpos + 1
    entry = np.where(ent <= tend[gpos], openp[np.minimum(ent, len(openp) - 1)], np.nan)
    fwd = close[tgt] / entry - 1.0
    net = fwd - cost
    y = make_label(label_kind, fwd, dates)

    finite = np.isfinite(fwd) & np.isfinite(y)
    dev = finite & (dates <= DEV_END.to_datetime64())
    hold = finite & (dates > DEV_END.to_datetime64())
    if hold.sum() == 0:
        raise SystemExit("no eligible rows after DEV_END — nothing to hold out")

    uniqd = np.array(sorted(pd.unique(dates[dev])))
    h_start = dates[hold].min()
    cut = uniqd[-EMBARGO]                      # 63 eligible sessions before the DEV boundary
    Xv = X.to_numpy(np.float32, copy=False)

    tr = np.flatnonzero(dev & (dates < cut))
    te = np.flatnonzero(hold)
    n_avail = len(tr)
    rng = np.random.default_rng(0)
    if len(tr) > TRAIN_SUBSMP:
        tr = rng.choice(tr, TRAIN_SUBSMP, replace=False)
    tr = tr[np.isfinite(Xv[tr]).all(1)]
    te = te[np.isfinite(Xv[te]).all(1)]

    if verbose:
        print(f"  TRAIN  {pd.Timestamp(uniqd[0]).date()} .. {pd.Timestamp(cut).date()} (exclusive)"
              f"  |  {len(tr):,} of {n_avail:,} eligible rows")
        print(f"  EMBARGO {EMBARGO} sessions: {pd.Timestamp(cut).date()} .. "
              f"{pd.Timestamp(h_start).date()} discarded")
        print(f"  TEST   {pd.Timestamp(h_start).date()} .. "
              f"{pd.Timestamp(dates[hold].max()).date()}  |  {len(te):,} rows, "
              f"{len(pd.unique(dates[te])):,} sessions", flush=True)

    pm, ps = fit_predict(model_kind, trees, Xv[tr], y[tr], Xv[te])
    ic = pd.Series(pm).corr(pd.Series(y[te]), method="spearman")
    if verbose:
        print(f"  pooled rank-IC {ic:+.4f}", flush=True)

    return pd.DataFrame({
        "date": dates[te], "ticker": tick[te], "gpos": gpos[te],
        "fwd": fwd[te].astype(np.float32), "cost": cost[te].astype(np.float32),
        "net": net[te].astype(np.float32), "sig": sig[te].astype(np.float32),
        "med_dv": mdv[te].astype(np.float32), "advr": advr[te].astype(np.float32),
        "y": y[te].astype(np.float32), "pred": pm.astype(np.float32),
        "pred_sd": ps.astype(np.float32), "fold": np.int8(0),
    })


def run(H, label_kind, ctx, n_folds=N_FOLDS, trees=200, train_years=0, model_kind="rf",
        verbose=True):
    X, gpos, close, openp, tend, dates, tick, cost, sig, mdv, advr = ctx

    tgt = np.minimum(gpos + H, tend[gpos])
    ent = gpos + 1
    entry = np.where(ent <= tend[gpos], openp[np.minimum(ent, len(openp) - 1)], np.nan)
    fwd = close[tgt] / entry - 1.0          # EXECUTABLE: buy the open of t+1, exit close of t+H
    net = fwd - cost
    y = make_label(label_kind, fwd, dates)

    finite = np.isfinite(fwd) & np.isfinite(y)
    dev = finite & (dates <= DEV_END.to_datetime64())
    uniqd = np.array(sorted(pd.unique(dates[dev])))
    start = int(0.30 * len(uniqd))
    folds = np.array_split(uniqd[start:], n_folds)
    rng = np.random.default_rng(0)
    Xv = X.to_numpy(np.float32, copy=False)

    preds = []
    for fi, fd in enumerate(folds):
        t_start, t_end = fd[0], fd[-1]
        ci = np.searchsorted(uniqd, t_start) - EMBARGO
        if ci <= 10:
            continue
        cut = uniqd[ci]
        ok = dates < cut
        if train_years > 0:
            ok &= dates >= cut - np.timedelta64(int(round(train_years * 365.25)), "D")
        tr = np.flatnonzero(dev & ok)
        te = np.flatnonzero(dev & (dates >= t_start) & (dates <= t_end))
        n_avail = len(tr)
        if len(tr) > TRAIN_SUBSMP:
            tr = rng.choice(tr, TRAIN_SUBSMP, replace=False)
        tr = tr[np.isfinite(Xv[tr]).all(1)]
        te = te[np.isfinite(Xv[te]).all(1)]
        if len(tr) < 1000 or len(te) < 100:
            continue

        pm, ps = fit_predict(model_kind, trees, Xv[tr], y[tr], Xv[te])

        ic = pd.Series(pm).corr(pd.Series(y[te]), method="spearman")
        if verbose:
            win = f"{train_years:g}yr roll" if train_years > 0 else "expanding"
            print(f"    fold {fi} {pd.Timestamp(t_start).date()}..{pd.Timestamp(t_end).date()}"
                  f" | {win} | train {len(tr):,}/{n_avail:,} test {len(te):,}"
                  f" | pooled rank-IC {ic:+.4f}", flush=True)
        preds.append(pd.DataFrame({
            "date": dates[te], "ticker": tick[te], "gpos": gpos[te],
            "fwd": fwd[te].astype(np.float32), "cost": cost[te].astype(np.float32),
            "net": net[te].astype(np.float32), "sig": sig[te].astype(np.float32),
            "med_dv": mdv[te].astype(np.float32), "advr": advr[te].astype(np.float32),
            "y": y[te].astype(np.float32), "pred": pm.astype(np.float32),
            "pred_sd": ps.astype(np.float32), "fold": np.int8(fi),
        }))
    return pd.concat(preds, ignore_index=True)


def build_context(adv_floor, feats_name, h, verbose=True, fit_end=None):
    """Load the feature panel, attach market context, compute cost, assemble X."""
    import cost_model as cm

    # `volume` is deliberately NOT in the feature panel — its level leaks the future (Rule 2), and
    # nothing downstream needs it: forward gathers use close/open, the universe uses bars/med_dv/
    # close_raw, and the cost model uses dollar_volume. `high`/`low` are needed by the cost model's
    # EDGE calibration (open/high/low/close), not by any XSEC feature.
    # Load ONLY what this run uses. Under --feats no_vol the VOL_BLOCK columns are never features,
    # so loading them cost ~0.5GB for nothing. ewm_sig is needed by ctx regardless of feature set.
    need = sorted(set(FEATURE_SETS[feats_name]) | {
        "ewm_sig", "med_dv", "dollar_volume", "close_raw",
        "open", "high", "low", "close", "bars"})
    p = sh_scan.load_features(columns=need, verbose=verbose)
    _mem("panel loaded", verbose)

    # Attach market context WITHOUT merge(): p.merge() builds an entire new ~4.5GB frame just to add
    # three per-date columns. Column-wise assignment via a unique-date lookup gives the same values
    # with no full-frame copy.
    cx = sh_scan.context()
    uni, inv = np.unique(p["date"].to_numpy(), return_inverse=True)
    al = cx.set_index("date").reindex(uni)
    for c in al.columns:
        p[c] = al[c].to_numpy()[inv]
    del cx, al, uni, inv
    gc.collect()
    _mem("context attached", verbose)

    # The panel is written sorted by (ticker, date), so sort_values() was a second full-frame copy
    # that changed nothing. What downstream needs is contiguous ticker blocks with ascending dates
    # inside each -- NOT lexicographic ticker order -- so verify exactly that, cheaply, and skip.
    codes = pd.factorize(p["ticker"], sort=False)[0]
    dts = p["date"].to_numpy("datetime64[ns]").astype("int64")
    dc, dd = np.diff(codes), np.diff(dts)
    ordered = bool((dc >= 0).all() and ((dc > 0) | (dd > 0)).all())
    del codes, dts, dc, dd
    gc.collect()
    if ordered:
        if verbose:
            print("  panel already ordered (ticker blocks contiguous, dates ascending) -- skip sort")
    else:
        print("  !! panel NOT ordered -- sorting (costs a full-frame copy)")
        p = p.sort_values(["ticker", "date"]).reset_index(drop=True)
    _mem("ordered", verbose)

    close, openp, tend = sh_scan._arrays(p)
    elig = sh_scan.eligible_mask(p, adv_floor=adv_floor, verbose=verbose)
    gpos = np.flatnonzero(elig)

    # cost from the calibrated curve. price_col='close_raw' — the tick floor is a real-world
    # constraint and needs the price the tape printed, not the adjusted one.
    # Sliced to only the columns cost_model touches: it does its own sort_values(...).copy() (plus
    # calibrate_edge_anchors's own per-ticker groupby), and running that over the full ~40-column
    # feature frame instead of these 7 is what pushed the smoke test past 16GB and got it OOM-killed.
    cost_cols = ["ticker", "date", "open", "high", "low", "close", "close_raw", "dollar_volume"]
    cf = cm.prepare_features(p[cost_cols], price_col="close_raw")
    # prepare_features must see the FULL history (dollar_adv and sigma_20 are trailing windows), but
    # the curve is FIT on the fitting window only when fit_end is set -- otherwise the holdout's own
    # spreads help calibrate the costs charged against it.
    if fit_end is None:
        model = cm.fit_cost_model(cf, verbose=verbose)
    else:
        fm = cf["date"] <= np.datetime64(fit_end)
        if verbose:
            print(f"  cost model fit on {int(fm.sum()):,}/{len(cf):,} rows "
                  f"(<= {pd.Timestamp(fit_end).date()}), then applied to all rows")
        model = cm.fit_cost_model(cf[fm], verbose=verbose)
    cost_all = cm.CostModel.cost_bps(model, cf["dollar_adv"], cf["sigma_20"],
                                     cf["sigma_ref"], cf["price"]) / 1e4
    del cf, model
    gc.collect()
    if verbose:
        print(f"  cost: median {np.nanmedian(cost_all[gpos])*1e4:.1f} bp round-trip on the universe")
    _mem("cost model done", verbose)

    # p.loc[elig] would copy ~6.8M x 40 float32 (~1.1GB) just to rank one column. Rank the extracted
    # arrays instead -- identical result (pct rank of med_dv within each date, eligible rows only).
    dt_e, md_e = p["date"].to_numpy()[gpos], p["med_dv"].to_numpy()[gpos]
    advr = np.full(len(p), np.nan)
    advr[gpos] = pd.Series(md_e).groupby(dt_e).rank(pct=True).to_numpy()

    X, names = assemble_features(p, elig, FEATURE_SETS[feats_name], RAW_FEATS, verbose=verbose,
                                 fit_end=fit_end)
    _mem("features assembled", verbose)

    # copy=True is load-bearing: .to_numpy() on a consolidated pandas block returns a VIEW, which
    # pins the entire ~4.5GB panel alive for the whole model fit. These three stay full-length
    # because run() indexes them by absolute panel position.
    ctx = (X, gpos,
           np.array(close, copy=True), np.array(openp, copy=True), np.array(tend, copy=True),
           dt_e, p["ticker"].to_numpy()[gpos],
           cost_all[gpos], p["ewm_sig"].to_numpy(np.float64)[gpos],
           md_e.astype(np.float64), advr[gpos])

    del p, close, openp, tend, cost_all, advr, elig
    gc.collect()
    _mem("panel released", verbose)
    return ctx, names


def main():
    global TRAIN_SUBSMP
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--h", type=int, default=PRIMARY_H)
    ap.add_argument("--trees", type=int, default=200)
    ap.add_argument("--adv-floor", type=float, default=25e6)
    ap.add_argument("--feats", default="no_vol", choices=list(FEATURE_SETS))
    ap.add_argument("--label", default="cz", choices=["cz", "rank", "raw"])
    ap.add_argument("--train-years", type=float, default=0,
                    help="0 = expanding window; >0 = rolling, N years before the embargo cut")
    ap.add_argument("--model", default="rf", choices=["rf", "xgb", "lgbm"])
    ap.add_argument("--holdout", action="store_true",
                    help="THE SINGLE LOOK: train through DEV_END-embargo, predict everything after")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--train-subsmp", type=int, default=TRAIN_SUBSMP,
                    help="flat per-fold cap on training rows, same number for every fold; "
                         "min_samples_leaf tracks it via LEAF_FRAC")
    args = ap.parse_args()
    TRAIN_SUBSMP = args.train_subsmp
    preflight()
    t0 = time.time()

    trees = 60 if args.smoke else args.trees
    folds = 2 if args.smoke else N_FOLDS
    tag = args.tag or f"adv{int(args.adv_floor/1e6)}"

    # For the holdout run, the winsor bounds and the cost curve are fit on DEV rows only, so no
    # holdout-period information reaches the features or the costs charged against it.
    ctx, names = build_context(args.adv_floor, args.feats, args.h,
                               fit_end=DEV_END if args.holdout else None)
    win = f"ROLLING {args.train_years:g}yr" if args.train_years > 0 else "EXPANDING"
    print(f"{len(names)} features ({args.feats}) | {args.model.upper()} {trees} trees × {folds} folds"
          f" | H={args.h} | {win} | DEV ≤ {DEV_END.date()}"
          f" | train cap {TRAIN_SUBSMP:,}/fold (flat)")

    print(f"\n▶ H={args.h} label={args.label} ADV≥${args.adv_floor/1e6:.0f}M", flush=True)
    if args.holdout:
        print("─" * 92)
        print("  HOLDOUT — the single look. Config frozen before running; nothing tuned here.")
        print("─" * 92, flush=True)
        oos = run_holdout(args.h, args.label, ctx, trees=trees, model_kind=args.model)
    else:
        oos = run(args.h, args.label, ctx, n_folds=folds, trees=trees,
                  train_years=args.train_years, model_kind=args.model)

    import sh_gates
    sh_gates.report(oos, args.h, args.label)

    if not args.smoke:
        sfx = "_holdout" if args.holdout else ""
        sfx += "" if args.train_years == 0 else f"_roll{args.train_years:g}"
        sfx += "" if args.model == "rf" else f"_{args.model}"
        out = os.path.join(sh_scan.DATA, f"sh_oos_H{args.h}_{args.label}_{tag}{sfx}.parquet")
        oos.to_parquet(out, index=False)
        print(f"\n💾 {out}  ({len(oos):,} rows)")
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
