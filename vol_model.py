import numpy as np
import pandas as pd

from news import API_KEY, collect_scores, daily_sentiment
from price_model import SETTINGS, get_prices, add_features, add_sentiment_features, make_model
from multi_model import TICKERS, PERIOD

HORIZON = 5
LAMBDA = 0.94

PRICE_FEATURES = ['Vol10_vs_EWMA', 'AbsRet_vs_EWMA', 'Ret_5d', 'Close_vs_SMA20']
NEWS_FEATURES = ['News_vs_Avg', 'News3_vs_Avg', 'Sent_Mean', 'Sent_Abs']


def add_vol_features(df):
    df = df.copy()
    r2 = np.log1p(df['Daily_Return']) ** 2
    df['EWMA_Vol'] = np.sqrt(r2.ewm(alpha=1 - LAMBDA).mean())
    df['Vol_10d'] = np.sqrt(r2.rolling(10).mean())
    df['Future_Vol'] = np.sqrt(r2.rolling(HORIZON).mean().shift(-HORIZON))

    # the model predicts how far realized vol will land from EWMA, not vol itself
    df['Target_Vol'] = np.log(df['Future_Vol'] / df['EWMA_Vol'])

    df['Vol10_vs_EWMA'] = np.log(df['Vol_10d'] / df['EWMA_Vol'])
    df['AbsRet_vs_EWMA'] = np.sqrt(r2) / df['EWMA_Vol']
    df['Ret_5d'] = df['Close'].pct_change(5)

    avg_count = df['News_Count'].rolling(20, min_periods=5).mean()
    df['News3_vs_Avg'] = df['News_Count'].rolling(3).sum() / (3 * avg_count)
    df['Sent_Abs'] = df['Sent_Mean'].abs()
    return df.replace([np.inf, -np.inf], np.nan)


def build_ticker(ticker):
    df = get_prices(ticker, PERIOD)
    if df.empty:
        raise ValueError('no price data')
    df = add_features(df)
    news, covered = collect_scores(ticker, df.index.min(), df.index.max())
    df = add_sentiment_features(df, daily_sentiment(news, covered, df.index))
    df = add_vol_features(df)
    df = df.dropna(subset=PRICE_FEATURES + ['Target_Vol', 'Vol_10d'])
    df = df[df['News_Count'].notna()].copy()
    df['Ticker'] = ticker
    return df


def build_panel(tickers):
    frames = []
    for i, ticker in enumerate(tickers, 1):
        print(f'[{i}/{len(tickers)}] {ticker}')
        try:
            frames.append(build_ticker(ticker))
        except Exception as e:
            print(f'  skipped {ticker}: {e}')
    return pd.concat(frames).sort_index()


def split_with_gap(panel):
    # targets look HORIZON days ahead, so drop that many dates before each cut
    dates = panel.index.unique().sort_values()
    i1, i2 = int(len(dates) * 0.6), int(len(dates) * 0.8)
    train = panel[panel.index < dates[i1 - HORIZON]]
    val = panel[(panel.index >= dates[i1]) & (panel.index < dates[i2 - HORIZON])]
    test = panel[panel.index >= dates[i2]]
    return train, val, test


def vol_errors(model, part, features):
    pred_vol = part['EWMA_Vol'] * np.exp(model.predict(part[features]))
    return np.abs(part['Future_Vol'] - pred_vol).to_numpy()


def tune_and_fit(train, val, features):
    val_scores = {}
    for name, params in SETTINGS.items():
        model = make_model(params)
        model.fit(train[features], train['Target_Vol'])
        val_scores[name] = vol_errors(model, val, features).mean()
        print(f"  {name:14} MAE {val_scores[name]:.5f}")

    best = min(val_scores, key=val_scores.get)
    print(f"  best on validation: {best}")

    train_val = pd.concat([train, val])
    model = make_model(SETTINGS[best])
    model.fit(train_val[features], train_val['Target_Vol'])
    return model


def paired(a, b):
    diff = a - b
    return diff.mean(), diff.std(ddof=1) / np.sqrt(len(diff))


def main():
    if not API_KEY:
        raise SystemExit('FINNHUB_API_KEY not found in .env')

    panel = build_panel(TICKERS)
    train, val, test = split_with_gap(panel)

    print(f"\n{panel['Ticker'].nunique()} stocks, forecasting realized vol over the next {HORIZON} days")
    for name, part in [("Train", train), ("Val", val), ("Test", test)]:
        print(f"{name:5} {part.index.min().date()} → {part.index.max().date()}  {len(part)} rows")

    errors = {
        'persistence (10d vol)': np.abs(test['Future_Vol'] - test['Vol_10d']).to_numpy(),
        'EWMA': np.abs(test['Future_Vol'] - test['EWMA_Vol']).to_numpy(),
    }
    for set_name, features in [('XGB price', PRICE_FEATURES),
                               ('XGB price + news', PRICE_FEATURES + NEWS_FEATURES)]:
        print(f"\n--- Validation: {set_name} ---")
        model = tune_and_fit(train, val, features)
        errors[set_name] = vol_errors(model, test, features)

    print(f"\n--- Test, looked at once ({len(test)} stock-days) ---")
    for name, err in errors.items():
        print(f"{name:22} MAE {err.mean():.5f}")

    print()
    for a, b in [('EWMA', 'XGB price'), ('EWMA', 'XGB price + news'), ('XGB price', 'XGB price + news')]:
        mean, se = paired(errors[a], errors[b])
        print(f"{b} vs {a}: lowers MAE by {mean:.6f} ± {se:.6f} ({mean / errors[a].mean():+.1%})")
    print("± is one standard error; overlapping 5-day windows and same-day co-movement make the real uncertainty larger.")


if __name__ == "__main__":
    main()
