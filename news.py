import os
import json
import time
from datetime import date

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv('FINNHUB_API_KEY')
CACHE_DIR = 'cache'
SCORES_PATH = os.path.join(CACHE_DIR, 'finbert_scores.json')
HISTORY_DAYS = 365


class NewsUnavailable(Exception):
    pass


def get_news(ticker, date_from, date_to):
    for attempt in range(2):
        time.sleep(1.1)
        response = requests.get('https://finnhub.io/api/v1/company-news', params={
            'symbol': ticker,
            'from': date_from,
            'to': date_to,
            'token': API_KEY,
        })
        if response.status_code == 429:
            print('Finnhub rate limit hit, waiting a minute...')
            time.sleep(60)
            continue
        data = response.json()
        if not isinstance(data, list):
            raise NewsUnavailable(data)
        return data
    raise NewsUnavailable('still rate limited after waiting')


def get_news_cached(ticker, date_from, date_to, cacheable=True):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f'finnhub_{ticker}_{date_from}_{date_to}.json')

    if cacheable and os.path.exists(path):
        with open(path) as f:
            return json.load(f)

    articles = get_news(ticker, date_from, date_to)
    if cacheable:
        with open(path, 'w') as f:
            json.dump(articles, f)
    return articles


def collect_headlines(ticker, start, end):
    today = pd.Timestamp(date.today())
    start = max(start, today - pd.Timedelta(days=HISTORY_DAYS))
    articles, covered = [], []

    # weeks are anchored on Mondays so the cache file names stay the same between runs
    for monday in pd.date_range(start, end, freq='W-MON'):
        sunday = monday + pd.Timedelta(days=6)
        finished = sunday < today
        date_to = min(sunday, today)
        try:
            week = get_news_cached(ticker, f'{monday:%Y-%m-%d}', f'{date_to:%Y-%m-%d}', cacheable=finished)
        except NewsUnavailable as e:
            print(f'Finnhub stopped at week of {monday:%Y-%m-%d}: {e}')
            break
        articles += week
        covered.append((monday, sunday + pd.Timedelta(days=1)))

    print(f'{len(articles)} articles over {len(covered)} weeks')
    return articles, covered


_finbert = None


def load_finbert():
    global _finbert
    if _finbert is None:
        from transformers import pipeline
        _finbert = pipeline('text-classification', model='ProsusAI/finbert', top_k=None)
    return _finbert


def score_headlines(headlines):
    scores = {}
    if os.path.exists(SCORES_PATH):
        with open(SCORES_PATH) as f:
            scores = json.load(f)

    missing = sorted(set(headlines) - set(scores))
    if missing:
        print(f'Scoring {len(missing)} new headlines with FinBERT...')
        finbert = load_finbert()
        for headline, result in zip(missing, finbert(missing, batch_size=32, truncation=True)):
            probs = {r['label']: r['score'] for r in result}
            scores[headline] = probs['positive'] - probs['negative']
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(SCORES_PATH, 'w') as f:
            json.dump(scores, f)

    return [scores[h] for h in headlines]


def collect_scores(ticker, start, end):
    articles, covered = collect_headlines(ticker, start, end)
    news = pd.DataFrame(articles, columns=['datetime', 'headline'])
    news = news[news['headline'].str.strip() != ''].drop_duplicates('headline')
    if news.empty:
        return pd.DataFrame(columns=['time', 'score']), covered

    published = pd.to_datetime(news['datetime'], unit='s', utc=True)
    news['time'] = published.dt.tz_convert('America/New_York').dt.tz_localize(None)
    news['score'] = score_headlines(news['headline'].tolist())
    return news[['time', 'score']], covered


def daily_sentiment(news, covered, trading_days):
    daily = pd.DataFrame(index=trading_days, columns=['Sent_Mean', 'News_Count'], dtype=float)
    for start, end in covered:
        daily.loc[(trading_days >= start) & (trading_days < end), 'News_Count'] = 0

    if news.empty:
        return daily

    # news after the 4pm close can only move the next session
    day = news['time'].dt.normalize() + pd.to_timedelta((news['time'].dt.hour >= 16).astype(int), unit='D')
    pos = trading_days.searchsorted(day)

    news = news[pos < len(trading_days)].copy()
    news['date'] = trading_days[pos[pos < len(trading_days)]]

    per_day = news.groupby('date')['score'].agg(['mean', 'size'])
    daily.loc[per_day.index, 'Sent_Mean'] = per_day['mean']
    daily.loc[per_day.index, 'News_Count'] = per_day['size']
    return daily
