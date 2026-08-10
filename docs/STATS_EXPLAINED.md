# Statistics Explained — what every number in this project actually means

Written for v10 and updated for v11; none of it is version-specific. Companion to
`terminology_glossary.md` (which
covers *strategy* vocabulary: universe, regime, signal, factor sort). This file covers the
*statistics* vocabulary. Every worked example uses real numbers from this project.

---

## 0 · The one problem all of this is trying to solve

You have a trading rule. It made money in a backtest. **There are only two possible reasons:**

1. The rule captures something real and repeatable.
2. The rule got lucky on this particular stretch of history.

Every statistic below exists to separate those two. That is the whole game. The reason this
project is so heavy on validation machinery is that reason #2 is *far* more common than it feels,
and a backtest gives you no natural warning when it happens.

---

## 1 · The basic descriptive numbers

| statistic | plain meaning | worth knowing |
|---|---|---|
| **mean return** | average outcome per trade | one huge winner can carry it — always read the median beside it |
| **median return** | the typical (middle) trade | if median ≪ mean, profits are concentrated in a few names |
| **hit rate** | % of trades that made money | a 56% hit rate is good; a strategy can be profitable at 35% if winners are big |
| **standard deviation (σ)** | how spread out the outcomes are | the raw material of "is this luck?" — more spread, less certainty |

**Concrete:** v11's top-1% picks (H=40) average **+2.53% net** per trade with a **54.5%** hit
rate, against a median of **+1.25%**. The median sitting well below the mean tells you the profit is
concentrated, and measuring it directly confirms how badly: of 47,358 top-1% trades, the best
**4.0%** account for *all* the net profit — the other 45,471 net to zero or below. That is why
breadth matters and why a 10-position version of this book is fragile.

---

## 2 · The t-statistic — the number that decides "skill or luck"

**The question it answers:** *if this rule truly had zero edge, how surprising would a result this
good be?*

$$t = \frac{\text{average return}}{\text{standard error of that average}}, \qquad
\text{standard error} = \frac{\sigma}{\sqrt{n}}$$

Read it as: **how many "error bars" away from zero is the result.**

- **|t| < 1** — indistinguishable from noise.
- **|t| ≈ 2** — the conventional line. Roughly a 5% chance of seeing this by luck alone.
- **|t| > 3** — strong.

The `√n` is where intuition usually goes wrong. **Averaging more observations shrinks the error
bar**, so more data makes the same average return more convincing. 100 trades averaging +2% is
weak; 10,000 trades averaging +2% is strong. This is exactly why the next section matters so much.

### The arithmetic, on this project's actual numbers

The daily rank-IC series has mean **+0.0487** and day-to-day standard deviation **0.1319**, over
**n = 4,228** days.

```
standard error = 0.1319 / sqrt(4228) = 0.002029
t              = 0.0487 / 0.002029   = 24.0        <- the NAIVE t
```

That 24.0 is wrong, for the reason in §3 — it assumes 4,228 independent days. After the Newey-West
correction of §4 the effective sample is about **228**, so:

```
standard error = 0.1319 / sqrt(228)  = 0.008734
t              = 0.0487 / 0.008734   = 5.58        <- the HONEST t
```

**5.58 is the number this project reports.** The naive 24.0 appears only to show the size of the
correction.

### Where the thresholds come from

A t maps to a probability through the normal distribution: `p = 2 × (1 − Φ(|t|))`, the two-sided
chance of seeing something at least this extreme when the true effect is exactly zero. Inverting it
gives the odds:

| t | p | if the true edge is zero, you'd see this |
|---|---|---|
| 1.6 | 0.11 | ~1 in 9 |
| 2.0 | 0.046 | ~1 in 22 |
| 3.0 | 0.0027 | ~1 in 370 |
| 4.0 | 0.000063 | ~1 in 15,800 |

⚠️ **Two cautions.**

**These are conditional, not verdicts.** "1 in 370" is the chance of a fluke *given the edge is
truly zero*. It is not "a 99.7% chance the edge is real" — that would need a prior on how likely a
real edge was before looking.

**Why 3 rather than 2 here.** The t ≈ 2 convention assumes **one pre-specified test**. Run twenty
variations and one clears t = 2 on noise alone. This project tested three model families, two label
horizons, several hold lengths, two thresholds and two training-set sizes — all against the same
data. Harvey, Liu and Zhu (2016) made exactly this argument about the published factor literature
and proposed **t > 3** as the bar for a mined result. That is the standard applied here.

**Far-tail values are not literal.** A t of 5.58 formally implies odds of about 1 in 40 million.
Nobody should believe that number; out there the normality assumption is doing all the work. Read
it as "not luck", and stop.

---

## 3 · ⚠️ Why our raw t-stats are almost always WRONG (and inflated)

The formula above hides an assumption: **every observation is an independent piece of news.**
In this project that assumption is badly violated, in two separate ways.

### 3a · Overlapping holding periods

We hold each position **H = 40 days** (or 20). A trade opened today and a trade opened tomorrow
share **39 of their 40 days**. They are not two independent bets on the world — they are almost
the same bet, counted twice.

So when we plug `n = 25,000 trades` into `σ/√n`, we are claiming 25,000 independent observations
when we have closer to `25,000 / 40 ≈ 625` worth of genuinely new information. The error bar comes
out about **√40 ≈ 6× too small**, and the t-stat comes out about **6× too big**.

### 3b · Everything on the same day shares the market

Every stock we hold on a given day rises and falls partly with the market. Twenty names on one day
is not twenty independent observations either — it is closer to one.

### The fixes, in the order we apply them

| step | what it does | effect |
|---|---|---|
| **1. collapse to one number per day** | average the picks within each day, so a day is one observation | kills problem 3b |
| **2. Newey–West correction** | inflate the error bar to account for day-to-day overlap | kills problem 3a |

**How much does this matter? Enormously.** In v9 the same edge measured:

- **t = 16.8** raw (meaningless)
- **t = 3.94** after both corrections (real, and the number that was actually reported)

**Rule for reading this project's output:** any t-stat labelled *naive*, *raw*, or *pooled* is for
comparing rows within one table only. The only t-stats that count as evidence are the ones labelled
**Newey–West (NW)**.

---

## 4 · Newey–West, in one paragraph

A plain variance says "total wiggle = sum of each day's own wiggle." Newey–West says "total wiggle
= each day's own wiggle **plus** how much neighbouring days wiggle *together*." Because our
overlapping trades make neighbouring days move together, adding those covariances back makes the
variance bigger, the error bar wider, and the t-stat smaller — which is the honest direction.

### What "NW-40" means

The **40 is the lag length**: how many days apart two observations can be and still be counted as
related. Covariances are down-weighted the further apart the days are (a **Bartlett** taper, falling
linearly to zero at the cutoff), and everything beyond the cutoff is ignored entirely.

The lag should match how far the overlap actually reaches, which is the label horizon:

| | |
|---|---|
| **NW-40** | what this project uses — matched to the H = 40 day label |
| NW-20 | what it used while the label was H = 20 |

Getting this wrong matters. The lag sat at 20 for a while after the label moved to 40, and it
under-corrected: measured IC autocorrelation is still **+0.25 at lag 20** and only reaches zero at
lag 40. That inflated the reported t from 5.58 to 6.60. The lag now tracks the horizon.

```
L = 0   ->  t = 24.0     (naive, no correction at all)
L = 20  ->  t = 6.60     (under-corrected: dependence still present at this lag)
L = 40  ->  t = 5.58     (matched to the label horizon -- what we report)
L = 60  ->  t = 5.28     (beyond the overlap; little further change)
```

The flattening between L = 40 and L = 60 is the confirmation that 40 is the right cutoff.

### The calculation, step by step

Everything reduces to four quantities. All values below are the real ones from this project's daily
rank-IC series (n = 4,228 days).

**1 · Naive variance, $g_0$**

The ordinary variance of the daily rank-IC, which assumes each day is independent of every other:

$$g_0 = \mathrm{Var}(IC_t) = 0.01740$$

**2 · Lagged covariances, $g_k$**

How much a day's IC co-moves with the IC $k$ days later:

$$g_k = \mathrm{Cov}(IC_t,\; IC_{t-k})$$

If days were independent these would all be ~0. They are not:

| lag $k$ | $g_k$ | correlation $g_k/g_0$ |
|---|---|---|
| 1 | 0.01558 | 0.895 |
| 2 | 0.01416 | 0.814 |
| 5 | 0.01128 | 0.649 |
| 10 | 0.00905 | 0.520 |
| 20 | 0.00429 | 0.247 |
| 30 | 0.00141 | 0.081 |
| **40** | **−0.00050** | **−0.029** |

Two days running are 90% correlated. The dependence decays steadily and hits zero at lag 40 —
exactly the label horizon, which is the evidence that overlapping windows are the cause.

**3 · Tapering weights, $w_k$**

Distant lags are estimated from fewer overlapping pairs and are noisier, so they are discounted.
The Bartlett taper falls linearly to zero at the cutoff:

$$w_k = 1 - \frac{k}{L+1} = 1 - \frac{k}{41}$$

So $w_1 \approx 0.976$ and $w_{40} \approx 0.024$.

**4 · Long-run variance, $v$**

Combine the naive variance with the weighted dependence. The factor of 2 counts each pair from both
directions:

$$v = g_0 + 2\sum_{k=1}^{40} w_k\, g_k = 0.01740 + 2(0.15232) = 0.32203$$

---

### What it produces

$$\frac{v}{g_0} = \frac{0.32203}{0.01740} = 18.51$$

Serial correlation inflates the variance about **18.5×**. Read it as: *the IC appears to vary by
0.0174 if you assume days are independent, but once the dependence between days is counted the
realistic figure is 0.3220.* The observations carry far less independent information than counting
trading days suggests.

Everything else follows. The standard error is the square root of variance over n, so the error bar
widens by $\sqrt{18.51} = 4.30$, and the t shrinks by the same factor:

```
t_naive = 24.01
t_NW    = 24.01 / 4.30 = 5.58
```

And the effective sample size is the same ratio expressed as a headcount:

$$n_{\text{eff}} = n \times \frac{g_0}{v} = \frac{4{,}228}{18.51} = 228$$

*4,228 overlapping days carry about as much information as 228 independent ones.*

Note this is **not** `n / H` = 4,228 / 40 = 106. That shortcut assumes each 40-day block is perfectly
redundant inside and perfectly independent outside. The measured version uses the actual decay above,
and lands roughly twice as high. The same shortcut applied per fold gives ~17 independent
observations where the measured figure is 32-54.

**Is this standard?** The Newey–West correction itself is completely standard (Newey & West, 1987)
and is what the significance claim rests on. Restating it as $n_{\text{eff}}$ is a presentational
convenience — the same variance-inflation idea that appears as "effective sample size" elsewhere in
time-series work. It is a legitimate restatement of the correction, not a separate test.

### One quirk worth knowing

Newey–West does not always *reduce* a t. If a series is **negatively** autocorrelated, the added
covariances are negative, the long-run variance comes out smaller than the naive one, and the
corrected t is slightly *larger*. That is what happens to the CAPM alpha residuals in this project —
their t rises from 1.52 to 1.61 under the correction. It is legitimate, but it also means the
reported `n_eff` can exceed the actual number of days, which is nonsense as a headcount. When that
happens, read it as "the overlap penalty does not apply here", not as extra evidence.

`nw_t()` in `notebooks/sharadar_results_v11.ipynb` implements this. Nothing is called significant in
this project unless it survives the correction.

---

## 5 · rank-IC (Information Coefficient) — how well the model *ranks*

**The question:** *within a single day, did the model put the stocks in the right order?*

### How it is calculated, step by step

1. **Take one date.** Pull every eligible stock that day — about **1,070** names on average.
2. **Rank them twice.** Once by the model's prediction (best → worst), once by the forward return
   they actually went on to deliver.
3. **Correlate the two rankings** with **Spearman**, which is just Pearson correlation applied to
   ranks rather than raw values. Using ranks matters: one stock going up 400% cannot dominate the
   measurement, so a single lottery ticket can't manufacture a signal.
4. That single number is the day's IC. **+1.0** = perfect ordering, **0** = no relationship,
   **−1.0** = exactly backwards.
5. **Repeat for all 4,228 days** and average.

The result is **rank-IC = +0.0487**.

Two details worth stating, because they are easy to get wrong:

- **It is computed per date, then averaged — never pooled across dates.** Pooling every stock-day
  into one big correlation mixes market-wide moves into the comparison and destroys the measurement.
  In this project the pooled version of fold 5 reads +0.0245 against +0.1041 done properly.
- **The forward return is the raw one**, but the model was trained on a within-date z-score. Ranking
  is invariant to that transformation, so the comparison is still apples to apples.

| rank-IC | meaning |
|---|---|
| +1.00 | perfect ordering |
| 0.00 | no relationship |
| −1.00 | exactly backwards |

**⚠️ The scale is deceptive.** +0.04 sounds like nothing. It is not. You are applying that ordering
to hundreds of names every single day, so a small consistent ordering advantage compounds into a
large return difference. **In equities, a sustained rank-IC of +0.03 to +0.05 is a genuinely useful
signal.** v11 runs **+0.0487**.

What matters as much as the level is **stability across folds**. v11's six folds are
`+0.0417, +0.0692, +0.0219, +0.0380, +0.0293, +0.0922` — all positive, but the strongest is over
4× the weakest. Every fold positive means it is not one lucky era; the spread means the edge is far
from constant.

⚠️ **Do not read a per-fold t as a significance test.** A fold is ~705 days, which is only about
**17 independent observations** once the 40-day overlap is accounted for. Corrected per-fold t
values run 1.20 to 5.02 and four of the six fall below 2 — which is what you would expect at that
sample size, not evidence against the signal. Significance is argued on the pooled series.

---

## 6 · Decile sorts and monotonicity — the practical version of rank-IC

Sort every day's stocks by the model's score, cut into 10 equal groups (**deciles**), and average
the actual return of each group. D1 = worst-scored, D10 = best-scored.

**What you want to see: each decile beats the one below it.** v11 (H=40, gross %):

```
D1     D2     D3     D4     D5     D6     D7     D8     D9     D10
-0.60  +0.66  +1.16  +1.43  +1.55  +1.68  +1.81  +1.91  +2.04  +2.35
```

Perfectly ordered — the decile-rank correlation is **+1.000**. Note D1 is *negative*: the bottom of
the ranking does not merely fail to outperform, it loses money, which is what makes a short leg
worth examining at all.

**Why this specific check exists.** v8's model was **U-shaped**: its *worst*-scored decile was one
of its most profitable. That means there is no threshold you can trade — "buy above score X" fails
because the bottom is good too. A model can have a fine average rank-IC and still be untradeable
this way, so monotonicity is checked separately and explicitly.

**⚠️ Deciles must be computed PER DATE, never pooled across all history.** Pooling lets a model
put zero picks on 64% of days and concentrate them in a few favourable periods — v8's headline
+3.92% was a pooling artifact; the tradeable number was +1.55%.

---

## 7 · Alpha, beta, and factor attribution — "skill, or a category you could have just bought?"

This is the single most important test in the project, and v9's whole conclusion turned on it.

The book buys stocks with recognisable traits: volatile, cheap, mid-liquidity. Those traits are
**categories** ("factors") anyone can buy without a model. So we run a regression:

```
book return = alpha + b1·market + b2·size + b3·momentum + b4·lowvol
                    + b5·reversal + b6·value + b7·lowprice + error
```

- **beta (b1…b7)** — how much of each category the book implicitly holds. β=0.76 to "market" means
  it behaves like 0.76 of a market index position.
- **alpha** — **what is left over that no category explains.** This is the candidate for skill.
- **R²** — what % of the book's day-to-day swings the categories explain. v11's top-1% long book:
  **69%** against SPY alone; the 80/20 hedge **52%**; the equal-weight S&P control **94%**, which is
  a polite way of saying "that control is an index fund."

### Why alpha is so much harder to prove than rank-IC

This is the single most confusing pair of numbers in the project: the ranking is significant at
**t = 5.58** while the alpha is not, at **t = 1.61**. Both are correct. They are different kinds of
measurement.

**Rank-IC is cross-sectional.** Every trading day it grades an entire ordering — ~1,070 stocks
against the returns they went on to deliver. One day produces one score built from a thousand
simultaneous comparisons.

**Alpha is time-series.** It grades one portfolio against one benchmark. A day contributes exactly
**one** number: what the book returned. There is no width to it.

So the model is graded a thousand-wide on ordering and one-wide on payoff. That asymmetry is the
entire explanation.

#### The arithmetic

The standard error of a mean is `σ / √n`. In annualised terms, for a return series measured over
some number of **years**:

```
SE(annual alpha) = residual volatility / sqrt(years)
```

"Residual" matters: alpha is what's left after removing beta × market, so the relevant scatter is
the *unexplained* part, not the book's total volatility. With R² against SPY, that is
`total vol × sqrt(1 − R²)`.

For the long book, over 2006-2022:

```
total volatility     33.6%
R2 vs SPY            0.69
residual volatility  33.6% x sqrt(1 - 0.69) = 33.6% x 0.557 = 18.7%
years                4,205 trading days / 252 = 16.69
SE(annual alpha)     18.7% / sqrt(16.69) = 18.7% / 4.08 = 4.58%
t                    6.95% / 4.58% = 1.52          (1.61 after Newey-West)
```

**The alpha and its own error bar are nearly the same size.** That is the whole problem, and it is
not a flaw in the strategy — it is what 17 years of a 33.6%-volatile book buys you.

#### What would actually fix it

The error bar shrinks with `√years`, so:

- To halve it, you need **4× the history** — about 67 years.
- To reach t = 3 at this alpha you would need SE ≈ 2.3%, which means ~66 years of the same book.

The other lever is the numerator: a **less volatile** version of the same strategy reaches
significance far sooner. The 80/20 hedge is a small step in that direction — residual volatility
16.1% against 18.7%, giving t = 1.60 naive on a slightly *smaller* alpha (+6.28%).

⚠️ **So "not significant" here does not mean "no edge."** It means 17 years is too short a rope to
tie this particular knot. The ranking evidence is strong; the payoff estimate is imprecise. Report
both, and do not let the first launder the second.

**Pre-registered gate: alpha > +3%/yr with Newey–West t > 2.**

**Why this killed v9.** v9's raw return was +17%/yr, which looks excellent — until you notice
market β=1.51 (so 1.5× market exposure, not skill) and a 0.75 loading on **low nominal price**, a
category that returned +15.6%/yr on its own. After paying for all seven categories, alpha was
**+4.73%/yr at t=1.14** — not distinguishable from zero. The model was an expensive way to buy
cheap volatile stocks.

**⚠️ Two honest caveats that do not go away:** there is no true fundamental *value* factor (that
needs EDGAR data, dead since v5), so real value exposure leaks into alpha; and returns are raw
rather than excess-of-cash, so alpha includes ~1%/yr of risk-free rate.

---

## 8 · Return and risk measures

**CAGR** — the smooth annual growth rate that would produce the same final wealth. This is the one
that matters for compounding.

**⚠️ CAGR is always lower than the average return, and the gap is not an error.** It is
**variance drag**: +50% then −50% averages 0% but leaves you at 0.75×. The drag is roughly
`σ²/2`. v11's long book runs at **33.6%** volatility, so the predicted drag is about **5.6%/yr**;
measured, it is **4.6%/yr** (arithmetic +20.4% against CAGR +15.7%). Any decomposition that mixes a
compounded book return with beta × compounded factor returns will not reconcile, and the difference
is exactly this drag masquerading as alpha — which is why factor attribution is done in *arithmetic*
terms.

**Sharpe ratio** — return per unit of volatility (`mean/σ`, annualized). Roughly: <0.5 poor,
0.5–1.0 decent, >1.0 strong. It is the honest comparator when two strategies have different risk:
v11 top-1% earns +15.7% at excess Sharpe 0.49 while SPY earns +8.8% at 0.34 — more return, and in
this case somewhat more return per unit of risk, but nothing like the gap the headline CAGR implies.

**Maximum drawdown** — the worst peak-to-trough loss. The number that decides whether a strategy is
survivable in practice. v11's is **−62.7%**, meaning capital lost nearly two thirds at the worst
point. Against SPY's −55.2% over the same window, that is the real cost of this book, and the
bootstrap 5th percentile is worse still at about −81%.

---

## 9 · How we avoid fooling ourselves during training

**Walk-forward validation.** Train on the past, test on the future, step forward, repeat. Never
score a model on data that existed before its training data. Six folds here.

**Purging + embargo.** Because a label needs H days to resolve, the last H days before a test
period *already contain* information about it. We delete a **63-day embargo** between train and
test — deliberately longer than the 40-day horizon.

**Held-out data.** 2023–2026 was quarantined so it could be used exactly once, against a frozen
written-down spec. The moment you peek and re-tune, it stops being evidence.

> ⚠️ **This is no longer true in v11.** The holdout has been evaluated more than once, so it is now
> confirmatory rather than a clean one-shot test, and there is no replacement period — the panel ends
> 2026-07-31. The principle above still stands; this project has spent it.

**Pre-registration.** Gates are written down *before* the run. This is the difference between "the
model passed the test" and "we found a test the model passes."

---

## 10 · Label overlap and "uniqueness" (why `--stride` exists)

Restating §3a from the training side. With H=40, one stock contributes ~250 rows a year that are
nearly the same observation. The forest therefore sees the same episode hundreds of times and
over-weights whichever periods a name happened to be around for.

Two responses, both implemented in `sh_regress.py`:

1. **Uniqueness weighting** (López de Prado, *Advances in Financial ML* ch. 4) — weight each row
   by how little its label window overlaps others.
   **⚠️ Measured here, this is nearly a no-op.** On a daily panel with fixed H, concurrency in the
   interior of a name's history is exactly H for everyone, so the weights come out at 0.0244–0.0245
   against a theoretical 0.0250. It does correct the edges — short histories, names near delisting,
   and names drifting in and out of the ADV band — but it is not the √40 fix one might hope for.
2. **Stride sampling** (`--stride k`) — keep only every k-th row per name. `stride = H` gives
   labels that do not overlap **at all**. This genuinely removes the redundancy; the price is
   throwing away ~95% of the rows.

**⚠️ Neither fixes the t-stats.** Decorrelating the *training set* does nothing to decorrelate the
*returns* of a portfolio that holds overlapping positions. Newey–West stays mandatory.

---

## 11 · Gross vs net

**Gross** = the raw price move. **Net** = after trading costs.

We judge the *signal* on gross (does the model rank stocks correctly?) and apply cost at
*deployment* (can we keep what it finds?). Keeping them separate stops a cost assumption from
quietly deciding a modelling question — and it is why the label is trained on gross returns, so the
model never learns to avoid volatile names merely for being expensive.

**⚠️ Cost assumptions decide conclusions, so never report just one.** In v11 the modelled cost is
19.8 bp median on the universe, rising to 30.5 bp at the traded top 1% as picks concentrate into
smaller names. Every result is reported net of that curve, and the curve itself is calibrated rather
than assumed.

Related: **Corwin–Schultz is retired.** It is a high-low *range* estimator, and once
volatility is partialled out it separates the most illiquid ADV decile from the most liquid by
**18 bp total** — it was ~95% measuring volatility, not spread. It charged v9's book 103 bp per
round-trip on names with $61M median daily volume, where the realistic figure is ~20 bp.

---

## 12 · Reading guide — which number to trust

| if you want to know… | look at | not at |
|---|---|---|
| does the model rank stocks correctly? | rank-IC, per fold | pooled rank-IC |
| is there a tradeable threshold? | per-date decile monotonicity | pooled deciles |
| is this skill or a buyable category? | **factor alpha + NW t-stat** | raw CAGR |
| is the edge real or luck? | **any Newey–West t-stat** | any naive/raw t-stat |
| can I actually live with this? | max drawdown, Sharpe | CAGR |
| does the conclusion survive assumptions? | the **cost ladder** | any single cost number |

**The one-line version:** a big CAGR proves nothing, a big raw t-stat proves nothing, and the only
claims that count here are factor-adjusted alpha with a Newey–West t-stat, reported across more
than one cost assumption.
