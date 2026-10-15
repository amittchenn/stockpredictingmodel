# Stock Volatility & Direction Forecasting

A quantitative research project that asks a simple question honestly: **can machine
learning forecast a stock's behavior from price and technical data — and if so, what,
and does it beat a simple statistical baseline?**

The short answer, arrived at empirically:
- **Next-day / 5-day price direction is not predictable** from these features (≈ coin flip).
- **Volatility *is* partly forecastable**, but a one-parameter **EWMA model beats a
  200-tree Random Forest** for most stocks.
- **Which model wins depends on the stock** — ML edges ahead only for stable, structured
  names like MSFT.
- **Pooled across 30 stocks and trained to correct EWMA**, XGBoost cuts 5-day volatility
  forecast error by **15%** vs. EWMA (`vol_model.py`).
- **News sentiment (FinBERT on ~20k Finnhub headlines) adds nothing**: not to next-day
  price/direction forecasts, and not to volatility forecasts (`multi_model.py`,
  `vol_model.py`).

## What the pipeline does

`stock.py` builds a full supervised-learning dataset from a single ticker and evaluates
several models with time-series-correct methodology:

1. **Data** — daily OHLCV via `yfinance` (10 years) plus forward-looking analyst EPS.
2. **Features** — technical indicators (SMA/EMA, RSI, MACD, Bollinger Bands) plus
   lagged returns and rolling volatility. All price-level features are converted to
   **scale-free ratios** (e.g. `close / sma_10 - 1`) so they mean the same thing at $40
   or $400.
3. **Targets** — two experiments:
   - *Direction*: will the close be higher N days from now? (classification)
   - *Volatility*: the change in log realized volatility over the next 5 days (regression)
4. **Split** — strictly **chronological** (train on the past, test on the recent tail).
   No shuffling, so the model never sees the future.
5. **Models** — Logistic Regression / Random Forest (direction), Random Forest Regressor
   and Gradient Boosting Regressor vs. Persistence vs. EWMA (volatility).
6. **Evaluation** — accuracy vs. a majority-class baseline (direction); MAE and R² vs.
   persistence and EWMA baselines (volatility); `TimeSeriesSplit` cross-validation
   throughout.

## Key findings

### 1. Direction is unpredictable
Across 1-day and 5-day horizons, logistic regression and random forests, every
configuration landed **at or below the majority-class baseline** (~50–57%). This is the
expected efficient-market result: if daily direction were predictable from public price
data, the edge would already be arbitraged away.

### 2. Volatility is forecastable — but simple models win
Forecasting future 5-day realized volatility, compared on identical out-of-sample data:

| Ticker | Persistence MAE | EWMA MAE | Random Forest MAE | Gradient Boosting MAE | Winner |
|--------|-----------------|----------|-------------------|-----------------------|--------|
| ORCL   | 0.01581         | **0.01425** | 0.01636        | —                     | EWMA |
| MU     | 0.01692         | **0.01589** | 0.01657        | —                     | EWMA |
| MSFT   | 0.00824         | 0.00768  | **0.00691**       | 0.00695               | Random Forest |

MSFT figures are from a run on data through 2026-10-02; ORCL and MU predate the
gradient-boosting comparison.

- For **ORCL** and **MU**, the one-parameter EWMA model beats the feature-rich forest.
  Their volatility is spike-driven, and spikes are best captured by pure persistence.
- For **MSFT** — a stable mega-cap with lower, more *structured* volatility — the forest
  finds real signal in the technical features and wins (test R² of +0.13 on the
  change-in-volatility target).
- A **gradient boosting regressor** (300 shallow trees, learning rate 0.03, 80% subsample)
  ties the forest on MSFT (MAE 0.00695 vs. 0.00691, R² +0.10 vs. +0.13). Boosting's
  usual edge — fitting residual structure — buys nothing here, which suggests the
  forest is already extracting what signal the features contain.

**The lesson:** no single model dominates. Model choice should be validated per asset;
complexity helps only when the data has structure the simple model can't capture.

### 3. Tuning EWMA's decay (lambda) is marginal
A sweep of lambda ∈ {0.85, 0.90, 0.94, 0.97, 0.99}, chosen on the training set and
confirmed on the held-out test set, produced clean U-shaped tuning curves but only
rounding-error improvements out-of-sample — and the train-optimal lambda sometimes did
*worse* on test. The RiskMetrics default of 0.94 is a fine choice.

## Methodology notes (things that were easy to get wrong)

- **Chronological splitting** — a random `train_test_split` would leak future information
  into training and inflate results.
- **Scaler fit on train only** — fitting on the full dataset leaks test-set statistics.
- **Stationary features** — absolute price levels (raw SMAs, band levels) don't generalize
  because train and test price ranges don't overlap; ratios do.
- **Trees can't extrapolate** — a random forest cannot predict a volatility higher than
  anything in its training set, so it structurally under-predicts spikes.
- **MAE vs. R²** — R² is dominated by rare huge errors (spikes); MAE is robust to them.
  They can disagree, and both are worth reporting.
- **Tune on train, confirm on test** — picking a hyperparameter by its test-set score is
  peeking, and will fool you.

## Next-day price forecast with gradient boosting (`price_model.py`)

A second experiment: predict tomorrow's closing price with XGBoost and see whether it
beats the naive forecast (tomorrow's close = today's close).

- **Ticker and history are chosen at runtime**: the script asks for a ticker and a
  `yfinance` period (`1y`, `5y`, `max`, ...) and re-prompts on invalid input.
- **Predict the return, not the price**: the model predicts tomorrow's return and converts
  it with `close × (1 + predicted return)`. Trees can't extrapolate, so a model trained
  on raw prices can never predict a price above its training range.
- **Price features**: today's return and the close relative to its 20- and 50-day SMAs
  (ratios again, not price levels).
- **Sentiment features** (`news.py`): headlines from Finnhub's free company-news API,
  each scored by **FinBERT** (`ProsusAI/finbert`, a BERT model fine-tuned on financial
  text) as P(positive) - P(negative). Scores are averaged per trading day, plus a 3-day
  average and the day's headline count relative to its 20-day average.
  - Timestamps are converted to New York time. Headlines at or after the 4pm close go to
    the **next** trading day and weekend news to Monday, so a feature never uses news
    published after the close it sits next to.
  - News is fetched one week per call and cached in `cache/`, and FinBERT scores are cached
    per headline, so reruns are fast. The current week is always re-fetched.
  - Finnhub's free tier covers about one year of news, so when sentiment is on, both models
    are trained and tested only on the days that have news coverage.
  - Alpha Vantage's `NEWS_SENTIMENT` was the first source, but it moved behind a paid plan
    during the project.
- **Ablation**: the model is trained twice, price-only and price + sentiment, with the
  same splits and tuning, so the effect of sentiment is measured rather than assumed.
- **Loss**: `reg:absoluteerror`, so one-off spikes (e.g. ORCL's +36% day in Sept 2025)
  don't dominate training the way they do with squared error.
- **Train / validation / test split (60/20/20, chronological)**: three hyperparameter
  settings, from "original" to "very cautious", are compared on validation only. The
  winner is retrained on train + validation and scored once on test.
- **Reported metrics**: price MAE vs. the naive forecast, direction accuracy, and the
  test period's "always up" baseline, plus a prediction for the next close.

The first runs on ORCL showed the same pattern as the direction experiment: the
unregularized model made large, confident daily calls (predicted-return std ≈ 3%, close to
the real returns' own spread) and lost to the naive forecast. The more cautious settings
are there to pull it back toward "no change" unless the features carry real signal.

### Results: ORCL, Oct 2025 – Oct 2026 (247 trading days with news)

Train 148 days, validation 49, test 50. Test period only:

| Model | Price MAE | Naive MAE | Direction accuracy |
|-------|-----------|-----------|--------------------|
| Price only (XGBoost) | $4.00 | $3.92 | 54.0% |
| Price + FinBERT sentiment | $3.93 | $3.92 | 52.0% |
| "Always up" baseline | | | 48.0% |

Neither model beats the naive forecast, and sentiment doesn't measurably change the
result. With 50 test days, one standard error of a direction accuracy is about ±7
points, so 52–54% can't be told apart from a coin flip. The validation set shows
the same thing: price-only hit 61% direction accuracy on validation and fell to 54%
on test. The honest conclusion for a single stock over one year is that **no edge is
detectable**. A real test needs more data, e.g. pooling many tickers.

## Pooled across 30 stocks (`multi_model.py`)

One stock over one year leaves ~50 test days, far too few to detect a small edge. This
script runs the same pipeline on 30 large caps across sectors and trains **one** XGBoost
model on all of them together (~7,500 stock-days).

- Splits are made on **calendar dates**, not rows, so no stock's future can leak into
  another stock's training data.
- Errors are measured on returns (%) instead of dollars, so a $500 stock doesn't dominate
  a $50 one. The naive forecast is a 0% return.
- Price-only vs. price + sentiment is compared on the same stock-days, with the paired
  difference in absolute error and its standard error. Stocks move together on the same
  day, so these errors understate the real uncertainty.
- The first run takes about 30 minutes (Finnhub's 60 calls/minute × 52 weeks × 30 stocks,
  plus FinBERT scoring). Everything is cached, so reruns take a minute or two.

### Results: 30 stocks, Oct 2025 – Oct 2026

7,410 stock-days: train 4,440, validation 1,470, test 1,500 (Aug 4 – Oct 13, 2026).

| Model | Test MAE (daily return) | Direction accuracy |
|-------|-------------------------|--------------------|
| Naive (0% change) | 1.333% | |
| "Always up" | | 49.7% |
| Price only (XGBoost) | 1.337% | 50.0% ± 1.3% |
| Price + FinBERT sentiment | 1.336% | 49.7% ± 1.3% |

Paired difference in absolute error (price only minus price + sentiment):
**0.0014% ± 0.0016%**, under one standard error, and the error bar is optimistic because
stocks move together on the same day.

**Conclusion:** across 30 large caps, neither price features nor same-day FinBERT
headline sentiment improves next-day forecasts over the naive no-change forecast. Public
headline sentiment appears to be priced in by the close. Validation picked the most
regularized settings for both models, which is what a model with no signal to learn
from would do: stay close to "no change".

## Does news forecast volatility? (`vol_model.py`)

Same-day sentiment didn't predict direction, but direction is the hardest target.
Volatility is partly forecastable (see above), so this asks whether news **volume** and
**sentiment** help forecast realized volatility over the next 5 days.

- Instead of competing with EWMA from scratch, XGBoost predicts the **correction** to it:
  the target is `log(future 5-day vol / EWMA vol)` and the forecast is
  `EWMA × exp(prediction)`. This is the "build on top of the strong baseline" extension.
- Price features: 10-day vol vs EWMA, today's absolute return vs EWMA, 5-day return, close
  vs 20-day SMA. News features: today's and 3-day headline counts vs their 20-day average,
  FinBERT sentiment and its absolute value.
- Same 30 stocks, split by calendar date, with a 5-trading-day gap before each cut, since
  every target looks 5 days ahead and would otherwise overlap the next split.
- Reports MAE on realized vol for persistence, EWMA, XGBoost price-only and XGBoost
  price + news, with paired standard errors.

### Results: 30 stocks, 5-day realized volatility

Train 4,200, validation 1,320, test 1,470 stock-days (Jul 30 – Oct 7, 2026).

| Forecast | Test MAE (daily vol) | vs. EWMA |
|----------|----------------------|----------|
| Persistence (10-day vol) | 0.00773 | -5.0% |
| EWMA (λ = 0.94) | 0.00736 | |
| XGBoost correction, price features | 0.00623 | **+15.4%** (0.00113 ± 0.00006) |
| XGBoost correction, price + news | 0.00623 | +15.5% (0.00114 ± 0.00006) |

- **Building on EWMA works.** The single-stock forest lost to EWMA, but a pooled model that
  learns a *correction* to EWMA cuts its error by 15%. That's about 18 standard errors;
  even if overlapping windows and co-movement made the true error bar several times
  wider, the improvement would still be well outside noise.
- **News adds nothing on top.** Price + news vs. price-only: 0.000008 ± 0.000012 (0.1%),
  within one standard error. Headline volume and FinBERT sentiment carry no volatility
  information beyond what recent returns already show.
- Caveat: the test window is about 10 weeks of one market regime. Walk-forward testing
  across more periods is the next step before trusting the 15% figure.

## Running it

```bash
pip install pandas numpy yfinance pandas_ta scikit-learn xgboost requests python-dotenv transformers torch
pip install pandas numpy yfinance scikit-learn
python stock.py
python price_model.py
python multi_model.py
python vol_model.py
```

For sentiment, get a free key at finnhub.io and put it in `.env` (see `.env.example`).
The first run downloads FinBERT (about 400 MB).
Without a key, `price_model.py` runs price-only.

Change the ticker and history at the top of the file:

```python
TICKER = 'MSFT'
YEARS_OF_HISTORY = 10
```

## Possible extensions

- Feed `ewma_vol` to the forest as a feature (let ML build *on top* of the strong
  baseline instead of competing with it from scratch).
- Compare a GARCH(1,1) model against EWMA.
- Test across a broader universe of tickers to map where ML helps vs. where it doesn't.
