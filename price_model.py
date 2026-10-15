import numpy as np
import pandas as pd
import yfinance as yf
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

from news import API_KEY, collect_scores, daily_sentiment

VALID_PERIODS = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
PRICE_FEATURES = ['Daily_Return', 'Close_vs_SMA20', 'Close_vs_SMA50']
SENTIMENT_FEATURES = ['Sent_Mean', 'Sent_3d', 'News_vs_Avg']

SETTINGS = {
    "original":      dict(max_depth=3, n_estimators=200, learning_rate=0.05),
    "cautious":      dict(max_depth=2, n_estimators=100, learning_rate=0.03,
                          min_child_weight=20, subsample=0.8),
    "very cautious": dict(max_depth=1, n_estimators=50, learning_rate=0.02,
                          min_child_weight=50, subsample=0.8),
}


def ask_ticker():
    return input('Enter a ticker symbol: ').strip().upper()


def ask_time_period():
    print('Valid time periods:', VALID_PERIODS)
    period = input('Enter a time period: ').strip().lower()
    while period not in VALID_PERIODS:
        print(f'{period} is not a valid time period')
        period = input('Enter a time period: ').strip().lower()
    return period


def get_prices(ticker, time_period):
    return yf.download(ticker, period=time_period, auto_adjust=True, multi_level_index=False)


def add_features(df):
    df = df.copy()
    df['Daily_Return'] = df['Close'].pct_change()
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['Close_vs_SMA20'] = df['Close'] / df['SMA_20'] - 1
    df['Close_vs_SMA50'] = df['Close'] / df['SMA_50'] - 1
    df['Target'] = df['Daily_Return'].shift(-1)
    return df


def add_sentiment_features(df, daily):
    df = df.join(daily)
    df['Sent_3d'] = df['Sent_Mean'].rolling(3, min_periods=1).mean()
    # raw article counts drift with the API's coverage, so compare to the recent average instead
    avg_count = df['News_Count'].rolling(20, min_periods=5).mean()
    df['News_vs_Avg'] = (df['News_Count'] / avg_count).replace([np.inf, -np.inf], np.nan)
    return df


def split_by_time(data):
    n = len(data)
    train = data.iloc[: int(n * 0.6)]
    val = data.iloc[int(n * 0.6): int(n * 0.8)]
    test = data.iloc[int(n * 0.8):]
    return train, val, test


def make_model(params):
    return XGBRegressor(objective='reg:absoluteerror', random_state=42, **params)


def evaluate(model, part, features):
    pred_return = model.predict(part[features])

    actual_price = part['Close'] * (1 + part['Target'])
    pred_price = part['Close'] * (1 + pred_return)
    naive_price = part['Close']

    model_mae = mean_absolute_error(actual_price, pred_price)
    naive_mae = mean_absolute_error(actual_price, naive_price)
    direction = ((pred_return > 0) == (part['Target'] > 0)).mean()
    return model_mae, naive_mae, direction


def tune_and_fit(train, val, features):
    val_scores = {}
    for name, params in SETTINGS.items():
        model = make_model(params)
        model.fit(train[features], train['Target'])
        model_mae, naive_mae, direction = evaluate(model, val, features)
        val_scores[name] = model_mae
        print(f"  {name:14} model ${model_mae:.2f}   naive ${naive_mae:.2f}   direction {direction:.1%}")

    best = min(val_scores, key=val_scores.get)
    print(f"  best on validation: {best}")

    train_val = pd.concat([train, val])
    model = make_model(SETTINGS[best])
    model.fit(train_val[features], train_val['Target'])
    return model, val_scores[best]


ticker = ask_ticker()
time_period = ask_time_period()
df = get_prices(ticker, time_period)

while df.empty:
    print(f'No data found for {ticker}. Check the symbol and try again.')
    ticker = ask_ticker()
    df = get_prices(ticker, time_period)

df = add_features(df)
feature_sets = {"price only": PRICE_FEATURES}

if API_KEY:
    news, covered = collect_scores(ticker, df.index.min(), df.index.max())
    df = add_sentiment_features(df, daily_sentiment(news, covered, df.index))
    feature_sets["price + sentiment"] = PRICE_FEATURES + SENTIMENT_FEATURES
else:
    print('No FINNHUB_API_KEY found in .env, running price-only.')

data = df.dropna(subset=PRICE_FEATURES + ['Target'])
if 'News_Count' in data:
    # compare both models on the same days, and only days the news history covers
    data = data[data['News_Count'].notna()]
    print(f'Using the {len(data)} days with news coverage for both models')
train, val, test = split_by_time(data)

for name, part in [("Train", train), ("Val", val), ("Test", test)]:
    line = f"{name:5} {part.index.min().date()} → {part.index.max().date()}  {len(part)} rows"
    if 'News_Count' in part:
        line += f"   sentiment coverage {part['News_Count'].notna().mean():.0%}"
    print(line)

results = {}
for set_name, features in feature_sets.items():
    print(f"\n--- Validation: {set_name} ---")
    model, val_mae = tune_and_fit(train, val, features)
    results[set_name] = (model, features, val_mae, *evaluate(model, test, features))

print("\n--- Test (looked at once) ---")
print(f"test baseline 'always up': {(test['Target'] > 0).mean():.1%}")
for set_name, (_, _, _, model_mae, naive_mae, direction) in results.items():
    print(f"{set_name:18} model ${model_mae:.2f}   naive ${naive_mae:.2f}   direction {direction:.1%}")

chosen = min(results, key=lambda k: results[k][2])
model, features = results[chosen][0], results[chosen][1]
latest = df.iloc[[-1]]
last_close = latest['Close'].iloc[0]
pred_ret = model.predict(latest[features])[0]
print(f"\nUsing '{chosen}' (best on validation)")
print(f"{ticker} last close ${last_close:.2f} → predicted next close "
      f"${last_close * (1 + pred_ret):.2f} ({pred_ret:+.2%})")
