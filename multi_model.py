import numpy as np
import pandas as pd

from news import API_KEY, collect_scores, daily_sentiment
from price_model import (PRICE_FEATURES, SENTIMENT_FEATURES, SETTINGS, get_prices,
                         add_features, add_sentiment_features, make_model)

TICKERS = [
    'AAPL', 'MSFT', 'NVDA', 'ORCL', 'GOOGL', 'AMZN', 'META', 'TSLA', 'AMD', 'INTC',
    'JPM', 'BAC', 'GS', 'V', 'MA', 'JNJ', 'PFE', 'UNH', 'MRK', 'XOM',
    'CVX', 'WMT', 'KO', 'PEP', 'MCD', 'DIS', 'NKE', 'BA', 'CAT', 'HD',
]
PERIOD = '2y'


def build_ticker(ticker):
    df = get_prices(ticker, PERIOD)
    if df.empty:
        raise ValueError('no price data')
    df = add_features(df)
    news, covered = collect_scores(ticker, df.index.min(), df.index.max())
    df = add_sentiment_features(df, daily_sentiment(news, covered, df.index))
    df = df.dropna(subset=PRICE_FEATURES + ['Target'])
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


def split_by_date(panel):
    # split on calendar dates, not rows, so no stock's future leaks into another's training
    dates = panel.index.unique().sort_values()
    cut1 = dates[int(len(dates) * 0.6)]
    cut2 = dates[int(len(dates) * 0.8)]
    return (panel[panel.index < cut1],
            panel[(panel.index >= cut1) & (panel.index < cut2)],
            panel[panel.index >= cut2])


def abs_errors(model, part, features):
    pred = model.predict(part[features])
    return pred, np.abs(part['Target'].to_numpy() - pred)


def tune_and_fit(train, val, features):
    naive = np.abs(val['Target']).mean()
    val_scores = {}
    for name, params in SETTINGS.items():
        model = make_model(params)
        model.fit(train[features], train['Target'])
        pred, err = abs_errors(model, val, features)
        direction = ((pred > 0) == (val['Target'] > 0)).mean()
        val_scores[name] = err.mean()
        print(f"  {name:14} MAE {err.mean():.3%}   naive {naive:.3%}   direction {direction:.1%}")

    best = min(val_scores, key=val_scores.get)
    print(f"  best on validation: {best}")

    train_val = pd.concat([train, val])
    model = make_model(SETTINGS[best])
    model.fit(train_val[features], train_val['Target'])
    return model


def main():
    if not API_KEY:
        raise SystemExit('FINNHUB_API_KEY not found in .env')

    panel = build_panel(TICKERS)
    train, val, test = split_by_date(panel)

    print(f"\n{panel['Ticker'].nunique()} stocks, {len(panel)} rows")
    for name, part in [("Train", train), ("Val", val), ("Test", test)]:
        print(f"{name:5} {part.index.min().date()} → {part.index.max().date()}  {len(part)} rows")

    feature_sets = {"price only": PRICE_FEATURES,
                    "price + sentiment": PRICE_FEATURES + SENTIMENT_FEATURES}
    up = (test['Target'] > 0).to_numpy()
    results = {}
    for set_name, features in feature_sets.items():
        print(f"\n--- Validation: {set_name} ---")
        model = tune_and_fit(train, val, features)
        pred, err = abs_errors(model, test, features)
        results[set_name] = (err, (pred > 0) == up)

    n = len(test)
    print(f"\n--- Test, looked at once ({n} stock-days) ---")
    print(f"{'naive (no change)':18} MAE {np.abs(test['Target']).mean():.3%}")
    print(f"{'always up':18} direction {up.mean():.1%}")
    for set_name, (err, correct) in results.items():
        acc = correct.mean()
        se = np.sqrt(acc * (1 - acc) / n)
        print(f"{set_name:18} MAE {err.mean():.3%}   direction {acc:.1%} ± {se:.1%}")

    diff = results["price only"][0] - results["price + sentiment"][0]
    se = diff.std(ddof=1) / np.sqrt(n)
    print(f"\nSentiment lowers MAE by {diff.mean():.4%} ± {se:.4%} (positive = sentiment helps)")
    print("± is one standard error; stocks move together on the same day, so the real uncertainty is larger.")


if __name__ == "__main__":
    main()
