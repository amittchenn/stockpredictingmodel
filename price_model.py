import yfinance as yf
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

VALID_PERIODS = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
FEATURES = ['Daily_Return', 'Close_vs_SMA20', 'Close_vs_SMA50']

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


def split_by_time(data):
    n = len(data)
    train = data.iloc[: int(n * 0.6)]
    val = data.iloc[int(n * 0.6): int(n * 0.8)]
    test = data.iloc[int(n * 0.8):]
    return train, val, test


def make_model(params):
    return XGBRegressor(objective='reg:absoluteerror', random_state=42, **params)


def evaluate(model, part):
    pred_return = model.predict(part[FEATURES])

    actual_price = part['Close'] * (1 + part['Target'])
    pred_price = part['Close'] * (1 + pred_return)
    naive_price = part['Close']

    model_mae = mean_absolute_error(actual_price, pred_price)
    naive_mae = mean_absolute_error(actual_price, naive_price)
    direction = ((pred_return > 0) == (part['Target'] > 0)).mean()
    return model_mae, naive_mae, direction


ticker = ask_ticker()
time_period = ask_time_period()
df = get_prices(ticker, time_period)

while df.empty:
    print(f'No data found for {ticker}. Check the symbol and try again.')
    ticker = ask_ticker()
    df = get_prices(ticker, time_period)

df = add_features(df)
data = df.dropna(subset=FEATURES + ['Target'])

train, val, test = split_by_time(data)
for name, part in [("Train", train), ("Val", val), ("Test", test)]:
    print(f"{name:5} {part.index.min().date()} → {part.index.max().date()}  {len(part)} rows")

# pick hyperparameters on validation only, test stays untouched until the end
print("\n--- Validation ---")
val_scores = {}
for name, params in SETTINGS.items():
    model = make_model(params)
    model.fit(train[FEATURES], train['Target'])
    model_mae, naive_mae, direction = evaluate(model, val)
    val_scores[name] = model_mae
    print(f"{name:14} model ${model_mae:.2f}   naive ${naive_mae:.2f}   direction {direction:.1%}")

best = min(val_scores, key=val_scores.get)
print("Best on validation:", best)

final_model = make_model(SETTINGS[best])
train_val = pd.concat([train, val])
final_model.fit(train_val[FEATURES], train_val['Target'])

model_mae, naive_mae, direction = evaluate(final_model, test)
print("\n--- Test ---")
print(f"model ${model_mae:.2f}   naive ${naive_mae:.2f}   direction {direction:.1%}")
print(f"test baseline 'always up': {(test['Target'] > 0).mean():.1%}")

latest = df.iloc[[-1]]
last_close = latest['Close'].iloc[0]
pred_ret = final_model.predict(latest[FEATURES])[0]
print(f"\n{ticker} last close ${last_close:.2f} → predicted next close "
      f"${last_close * (1 + pred_ret):.2f} ({pred_ret:+.2%})")
