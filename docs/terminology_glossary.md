## Terminology Glossary

Definitions used throughout this spec to avoid ambiguous or overloaded use of "regime."

### 1. Universe / Eligible Universe
The static (or slowly time-varying) screen applied before any signal logic runs. Defines which tickers are candidates on a given day.

- **Market cap band**: e.g., $10M–$100M = micro-cap eligible universe
- **Liquidity floor**: minimum ADV (average daily volume) threshold
- **Price floor**: minimum share price, if applicable

Universe membership is re-evaluated periodically (e.g., monthly) and does not depend on any technical signal.

### 2. Regime
A state variable describing the *environment* a stock or the market is currently in. Regimes condition strategy behavior; they are not selection criteria on their own.

- **Trend regime**: e.g., price > 200-day SMA = uptrend regime, else downtrend regime
- **Volatility regime**: e.g., realized vol or ATR percentile bucketed into low/medium/high
- **Sector-relative regime**: a stock's regime classification made relative to its sector benchmark rather than in absolute terms

In this pipeline, the 200-day SMA uptrend gate is a **trend regime filter**.

### 3. Signal / Trigger / Event
A specific, timestamped condition being met. Produces a labeled event for downstream processing. Not a universe criterion, not a regime.

- **Tier-1 signal / trigger**: RSI(2) < 15 (mean-reversion entry trigger)
- **Event**: the (ticker, timestamp) pair generated when the trigger fires

### 4. Cross-Sectional Factor / Anomaly
A ranking of the eligible universe on some characteristic at a point in time.

- **Momentum factor (12-1)**: standard Jegadeesh-Titman construction — trailing 12-month return, skipping the most recent month to avoid short-term reversal contamination
- **Short-term reversal factor (1-month)**: trailing 1-month return, used to identify "losers"; distinct from momentum, opposite sign effect over short horizons

### 5. Rank Bucket / Portfolio Sort
The formal name for "taking the worst/best N stocks by some factor." Fama-French terminology.

- **Decile/quintile sort**: dividing the ranked universe into equal-sized buckets
- **Sort portfolio**: e.g., "bottom decile sorted on 1-month return" = loser portfolio; "top decile sorted on 12-1 momentum" = winner portfolio

---

### Applied to This Pipeline

| Layer | Term | Example |
|---|---|---|
| Universe | Eligible universe | Micro-cap ($10M–$100M) ∩ ADV ≥ X, CIK→EDGAR sector-classified |
| Regime | Trend regime filter | 200-day SMA uptrend gate, sector-relative |
| Signal | Tier-1 trigger/event | RSI(2) < 15 |
| Label | Triple-barrier label | Vol-scaled, anchored to entry price |
| Classifier | Meta-classifier | LightGBM filters Tier-1 events for precision |
