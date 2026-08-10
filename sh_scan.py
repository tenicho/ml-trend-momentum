"""
sh_scan.py — feature panel, universe and benchmarks for the Sharadar build.

Replaces sc_scan.py (FMP). The feature maths is the same where it was already correct; what
changes is which column feeds which calculation, because Sharadar exposes the adjustment states
explicitly instead of making us reconstruct them.

THE COLUMN RULES THIS MODULE ENFORCES
  returns, indicators, labels   `close` / `open` / `high` / `low`   fully adjusted — a return is a
                                                                    ratio, so the factor cancels
  price LEVEL (feature, floor)  `close_raw`                        what the tape printed
  liquidity                     `dollar_volume`                    raw notional, split-invariant
  volume                        ratios only, never a level          split-adjusted; its level
                                                                    correlates with future returns

⚠️ The old FMP build had to reconstruct a point-in-time price from split history. Sharadar ships
`close_raw`, so that entire step is gone — along with the class of bug it existed to fix.

NEW HERE: signed dollar-flow features (`flow_10`, `flow_63`), an OBV-style idea rebuilt on
dollar_volume so it is split-invariant and bounded in [-1, +1].

Run:  python sh_scan.py --build          # build the feature panel (slow, once)
      python sh_scan.py --describe       # sanity-read an existing one
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd

DATA = "data_sharadar"
PANEL = os.path.join(DATA, "sh_panel.parquet")
BENCH = os.path.join(DATA, "sh_benchmarks.parquet")
TICKERS = os.path.join(DATA, "sh_tickers.parquet")
MEMBERSHIP = os.path.join(DATA, "sh_sp500_membership.parquet")
FEATURES = os.path.join(DATA, "sh_features.parquet")

EPS = 1e-12
ADV_WINDOW = 63                 # trailing sessions for the liquidity median
ENRICH_ADV_FLOOR = 10e6         # only enrich tickers that EVER reach this — see module note below
PRICE_FLOOR = 5.0               # applied to close_raw, never to the adjusted close
MIN_HISTORY = 252

# ⚠️ ENRICH_ADV_FLOOR IS NOT A LOOKAHEAD. It decides which tickers get features computed, not which
# rows are eligible to trade — eligibility stays point-in-time in `eligible_mask`. Any bar meeting a
# $25M trading floor necessarily belongs to a ticker that ever reached $10M, so no tradeable row is
# lost. It exists because enriching all 41.8M rows would not fit in memory; this keeps 23.6M.


# ── indicator helpers ─────────────────────────────────────────────────────────────────────────
def _wilder(s, n):
    return s.ewm(alpha=1 / n, adjust=False).mean()


def wilder_rsi(close, n):
    d = close.diff()
    up = _wilder(d.clip(lower=0), n)
    dn = _wilder(-d.clip(upper=0), n)
    return 100 - 100 / (1 + up / (dn + EPS))


def _mfi(h, l, c, v, n=14):
    tp = (h + l + c) / 3
    mf = tp * v
    pos = mf.where(tp > tp.shift(), 0.0).rolling(n).sum()
    neg = mf.where(tp < tp.shift(), 0.0).rolling(n).sum()
    return 100 - 100 / (1 + pos / (neg + EPS))


def _cmf(h, l, c, v, n=20):
    mult = ((c - l) - (h - c)) / ((h - l) + EPS)
    return (mult * v).rolling(n).sum() / (v.rolling(n).sum() + EPS)


def _adx(h, l, c, n=14):
    up, dn = h.diff(), -l.diff()
    plus = np.where((up > dn) & (up > 0), up, 0.0)
    minus = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = _wilder(tr, n)
    pdi = 100 * _wilder(pd.Series(plus, index=h.index), n) / (atr + EPS)
    mdi = 100 * _wilder(pd.Series(minus, index=h.index), n) / (atr + EPS)
    return _wilder((pdi - mdi).abs() / (pdi + mdi + EPS) * 100, n)


def _cmo(c, n=20):
    d = c.diff()
    up = d.clip(lower=0).rolling(n).sum()
    dn = (-d.clip(upper=0)).rolling(n).sum()
    return 100 * (up - dn) / (up + dn + EPS)


def _aroon(h, l, w=25):
    hi = h.rolling(w + 1).apply(lambda x: float(np.argmax(x)), raw=True)
    lo = l.rolling(w + 1).apply(lambda x: float(np.argmin(x)), raw=True)
    return (hi - lo) / w * 100


def _linreg_slope(c, w=30):
    x = np.arange(w)
    xc = x - x.mean()
    den = (xc ** 2).sum()
    return c.rolling(w).apply(lambda y: float((xc * (y - y.mean())).sum() / den), raw=True) / c


# ── per-ticker enrichment ─────────────────────────────────────────────────────────────────────
def _enrich(g: pd.DataFrame) -> pd.DataFrame:
    """All derived columns for one listing's continuous history (sorted by date)."""
    g = g.sort_values("date")
    o, h, l, c = g["open"], g["high"], g["low"], g["close"]     # ADJUSTED — correct for indicators
    v, dv = g["volume"], g["dollar_volume"]
    raw = g["close_raw"]                                        # the LEVEL

    f = pd.DataFrame({
        "date": g["date"].values, "ticker": g["ticker"].values,
        "permaticker": g["permaticker"].values,
        "open": o.values, "high": h.values, "low": l.values, "close": c.values,
        "close_raw": raw.values, "dollar_volume": dv.values,
        "is_delisted": g["is_delisted"].values, "sector": g["sector"].values,
    })
    ret = c.pct_change(fill_method=None)

    f["med_dv"] = dv.rolling(ADV_WINDOW).median().values
    f["bars"] = np.arange(len(g))

    # ── returns / momentum ──
    f["ret_1"] = ret.values
    f["ret_20"] = (c / c.shift(20) - 1).values
    f["ret_12_1"] = (c.shift(21) / c.shift(252) - 1).values      # 12-1, skipping the last month
    f["mom_3_1"] = (c.shift(21) / c.shift(63) - 1).values
    f["mom_6_1"] = (c.shift(21) / c.shift(126) - 1).values
    f["pct_52wk_high"] = (c / c.rolling(252).max() - 1).values
    updn = (ret < 0).rolling(231).mean() - (ret > 0).rolling(231).mean()
    f["fip"] = (np.sign(c.shift(21) / c.shift(252) - 1) * updn.shift(21)).values
    f["mom_consist"] = (((c / c.shift(21) - 1) > 0).rolling(231).mean().shift(21)).values
    f["gap"] = (o / c.shift(1) - 1).values

    # ── volatility ──
    f["vol20"] = ret.rolling(20).std().values
    f["vol60"] = ret.rolling(60).std().values
    f["ewm_sig"] = ret.ewm(span=20, adjust=False).std().values
    f["vol_regime"] = (ret.rolling(20).std() / (ret.rolling(60).std() + EPS)).values
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    f["atr14"] = (tr.rolling(14).mean() / c).values
    gk = 0.5 * np.log(h / l) ** 2 - (2 * np.log(2) - 1) * np.log(c / o) ** 2
    f["gk_vol"] = np.sqrt(gk.rolling(20).mean().clip(lower=0)).values

    # ── trend ──
    sma10, sma20 = c.rolling(10).mean(), c.rolling(20).mean()
    sma30, sma40 = c.rolling(30).mean(), c.rolling(40).mean()
    sma50, sma200 = c.rolling(50).mean(), c.rolling(200).mean()
    f["dist_sma20"] = ((c - sma20) / sma20).values
    f["dist_sma50"] = ((c - sma50) / sma50).values
    f["dist_sma200"] = ((c - sma200) / sma200).values
    f["ma_ribbon"] = (pd.concat([sma10, sma20, sma30, sma40, sma50], axis=1)
                        .std(axis=1) / c).values
    f["adx14"] = _adx(h, l, c, 14).values
    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    f["macd_dist"] = ((macd - macd.ewm(span=9, adjust=False).mean()) / c).values
    hi20, lo20 = h.rolling(20).max(), l.rolling(20).min()
    f["donch_pos"] = ((c - lo20) / (hi20 - lo20 + EPS)).values
    f["linreg30"] = _linreg_slope(c, 30).values
    ten = (h.rolling(9).max() + l.rolling(9).min()) / 2
    kij = (h.rolling(26).max() + l.rolling(26).min()) / 2
    spanB = (h.rolling(52).max() + l.rolling(52).min()) / 2
    f["kumo_dist"] = ((c - ((ten + kij) / 2 + spanB) / 2) / c).values

    # ── oscillators ──
    f["rsi2"] = wilder_rsi(c, 2).values
    f["rsi14"] = wilder_rsi(c, 14).values
    f["roc10"] = (c / c.shift(10) - 1).values
    hi14, lo14 = h.rolling(14).max(), l.rolling(14).min()
    f["stoch_k14"] = ((c - lo14) / (hi14 - lo14 + EPS)).values
    f["aroon_osc25"] = _aroon(h, l, 25).values
    f["cmo20"] = _cmo(c, 20).values
    f["mfi14"] = _mfi(h, l, c, v, 14).values
    f["cmf20"] = _cmf(h, l, c, v, 20).values

    # ── volume: RATIOS ONLY. A split cancels top and bottom, so these are safe (Rule 2). ──
    relvol = v / (v.rolling(20).mean() + EPS)
    f["vol_ratio"] = relvol.values
    f["vol_z"] = ((v - v.rolling(20).mean()) / (v.rolling(20).std() + EPS)).values
    f["rvm20"] = (ret * relvol).rolling(20).mean().values
    vu = v.where(ret > 0).rolling(63, min_periods=20).mean()
    vd = v.where(ret < 0).rolling(63, min_periods=20).mean()
    f["updn_vol"] = (vu / (vd + EPS)).values

    # ── signed dollar flow — the OBV replacement ──
    # (up-day dollars − down-day dollars) / all dollars, over W sessions. Built on dollar_volume,
    # which is split-invariant, so unlike OBV this carries no split leak. Bounded [-1, +1], so it
    # is already comparable across stocks with no scaling.
    signed = np.sign(c.diff().fillna(0.0)) * dv
    for w in (10, 63):
        num = signed.rolling(w, min_periods=w).sum()
        den = dv.rolling(w, min_periods=w).sum()
        f[f"flow_{w}"] = (num / (den + EPS)).values

    return f


_F32 = [c for c in (
    "open high low close close_raw dollar_volume med_dv ret_1 ret_20 ret_12_1 mom_3_1 mom_6_1 "
    "pct_52wk_high fip mom_consist gap vol20 vol60 ewm_sig vol_regime atr14 gk_vol dist_sma20 "
    "dist_sma50 dist_sma200 ma_ribbon adx14 macd_dist donch_pos linreg30 kumo_dist rsi2 rsi14 "
    "roc10 stoch_k14 aroon_osc25 cmo20 mfi14 cmf20 vol_ratio vol_z rvm20 updn_vol flow_10 flow_63"
).split()]


def build_feature_panel(chunk_tickers: int = 600, verbose: bool = True):
    """Enrich the Sharadar panel into a cached feature file, streamed in ticker chunks."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    t0 = time.time()
    if verbose:
        print("loading panel …")
    p = pd.read_parquet(PANEL)
    p["ticker"] = p["ticker"].astype("category")
    p = p.sort_values(["ticker", "date"]).reset_index(drop=True)

    adv = p.groupby("ticker", observed=True)["dollar_volume"].transform(
        lambda s: s.rolling(ADV_WINDOW, min_periods=ADV_WINDOW).median())
    ever = adv.groupby(p["ticker"], observed=True).max() >= ENRICH_ADV_FLOOR
    keep = ever[ever].index
    p = p[p["ticker"].isin(keep)].copy()
    p["ticker"] = p["ticker"].cat.remove_unused_categories()
    tickers = list(p["ticker"].cat.categories)
    if verbose:
        print(f"  {len(tickers):,} tickers ever reach ${ENRICH_ADV_FLOOR/1e6:.0f}M ADV "
              f"→ {len(p):,} rows to enrich")

    writer, n = None, 0
    for i in range(0, len(tickers), chunk_tickers):
        batch = set(tickers[i:i + chunk_tickers])
        sub = p[p["ticker"].isin(batch)]
        enr = sub.groupby("ticker", observed=True, group_keys=False).apply(_enrich)
        enr["ticker"] = enr["ticker"].astype(str)
        enr["sector"] = enr["sector"].astype(str)
        for col in _F32:
            if col in enr.columns:
                enr[col] = enr[col].astype("float32")
        tbl = pa.Table.from_pandas(enr, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(FEATURES, tbl.schema)
        writer.write_table(tbl)
        n += len(enr)
        if verbose:
            print(f"  chunk {i//chunk_tickers + 1}/{-(-len(tickers)//chunk_tickers)}: "
                  f"{n:,} rows  ({time.time()-t0:.0f}s)", flush=True)
        del sub, enr, tbl
    writer.close()
    if verbose:
        print(f"💾 {FEATURES}  ({n:,} rows, {time.time()-t0:.0f}s)")


# ── loading / universe / benchmarks ───────────────────────────────────────────────────────────
def load_features(columns=None, verbose=False) -> pd.DataFrame:
    if columns is not None:
        columns = list(dict.fromkeys(["ticker", "date"] + list(columns)))
    f = pd.read_parquet(FEATURES, columns=columns)
    f["date"] = pd.to_datetime(f["date"])
    if verbose:
        print(f"features {len(f):,} rows × {f.shape[1]} cols "
              f"({f.memory_usage(deep=False).sum()/1e9:.2f} GB)")
    return f


def eligible_mask(panel, adv_floor=25e6, price_floor=PRICE_FLOOR,
                  min_history=MIN_HISTORY, verbose=False):
    """Point-in-time tradeable universe.

    ⚠️ The price floor uses `close_raw`. On the adjusted close it would exclude 1998 Apple (really
    $16.25, adjusted $0.122) for the crime of splitting later, while admitting genuine penny stocks
    whose adjusted history got inflated by reverse splits.

    ⚠️ `is_delisted` is NOT used. A delisted name's bars are real and its last bar IS the delisting;
    filtering on it reintroduces the survivorship bias this dataset exists to remove.
    """
    m = (panel["bars"] >= min_history) & panel["med_dv"].notna()
    if price_floor is not None:
        m &= panel["close_raw"] >= price_floor
    m &= panel["med_dv"] >= adv_floor
    m = m.fillna(False).to_numpy()
    if verbose:
        nd = panel.loc[m, "date"].nunique()
        print(f"  universe: ADV ≥ ${adv_floor/1e6:.0f}M, price ≥ ${price_floor:.0f} → "
              f"{m.sum():,} rows, {m.sum()/nd:.0f} names/day over {nd:,} sessions")
    return m


def benchmarks(tickers=("SPY", "MDY", "IWM"), total_return_only=True) -> pd.DataFrame:
    """Benchmark closes, wide by ticker.

    ⚠️ Rule 6: ETFs carry dividends, indices do not. Comparing a total-return strategy against a
    price index understates the benchmark by every dividend ever paid — SPY's div_factor in 1998 is
    0.6116, so ~39% of its total return since is dividends. Default keeps ETFs only.
    """
    b = pd.read_parquet(BENCH)
    b["date"] = pd.to_datetime(b["date"])
    if total_return_only:
        b = b[~b["is_index"]]
    b = b[b["ticker"].isin(tickers)]
    return b.pivot_table(index="date", columns="ticker", values="close")


def context() -> pd.DataFrame:
    """Market-state series for the regime block. VIX is a LEVEL — never a return."""
    b = pd.read_parquet(BENCH)
    b["date"] = pd.to_datetime(b["date"])
    px = b.pivot_table(index="date", columns="ticker", values="close")
    out = pd.DataFrame(index=px.index)
    out["vix"] = px["^VIX"]
    spy = px["SPY"]
    out["spy_dist200"] = (spy - spy.rolling(200).mean()) / spy.rolling(200).mean()
    out["spy_5dvol"] = spy.pct_change(fill_method=None).rolling(5).std()
    return out.reset_index()


def sp500_membership() -> pd.DataFrame:
    m = pd.read_parquet(MEMBERSHIP)
    m["date"] = pd.to_datetime(m["date"])
    return m


def _arrays(panel):
    """Positional arrays for vectorised forward gathers. Panel must be sorted (ticker, date)."""
    close = panel["close"].to_numpy(np.float64)
    openp = panel["open"].to_numpy(np.float64)
    tick = panel["ticker"].to_numpy()
    n = len(panel)
    tend = np.empty(n, dtype=np.int64)
    change = np.flatnonzero(tick[1:] != tick[:-1])
    starts = np.concatenate(([0], change + 1))
    ends = np.concatenate((change, [n - 1]))
    for s, e in zip(starts, ends):
        tend[s:e + 1] = e
    return close, openp, tend


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--describe", action="store_true")
    args = ap.parse_args()
    if args.build:
        build_feature_panel()
    if args.describe or not args.build:
        f = load_features(columns=["close_raw", "med_dv", "bars", "flow_10", "flow_63",
                                   "ret_12_1", "ewm_sig"], verbose=True)
        print(f"  {f.ticker.nunique():,} tickers | {f.date.min().date()} → {f.date.max().date()}")
        for floor in (10e6, 25e6, 50e6):
            eligible_mask(f, adv_floor=floor, verbose=True)
        print("\n  flow features:")
        print(f[["flow_10", "flow_63"]].describe().T.to_string())


if __name__ == "__main__":
    main()
