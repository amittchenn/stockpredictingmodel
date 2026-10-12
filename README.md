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

| Ticker | Persistence MAE | EWMA MAE | Random Forest MAE | Winner |
|--------|-----------------|----------|-------------------|--------|
| ORCL   | 0.01581         | **0.01425** | 0.01636        | EWMA |
| MU     | 0.01692         | **0.01589** | 0.01657        | EWMA |
| MSFT   | 0.00816         | 0.00760  | **0.00676**       | Random Forest |

- For **ORCL** and **MU**, the one-parameter EWMA model beats the feature-rich forest.
  Their volatility is spike-driven, and spikes are best captured by pure persistence.
- For **MSFT** — a stable mega-cap with lower, more *structured* volatility — the forest
  finds real signal in the technical features and wins (test R² of +0.14 on the
  change-in-volatility target).

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

## Running it

```bash
pip install pandas numpy yfinance scikit-learn
python stock.py
```

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
