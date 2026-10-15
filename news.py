import os
import json
import time
from datetime import datetime

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv('ALPHAVANTAGE_API_KEY')
CACHE_DIR = 'cache'
MAX_ARTICLES = 1000


class NewsUnavailable(Exception):
    pass


def get_news(ticker, time_from, time_to):
    time.sleep(1.2)
    response = requests.get("https://www.alphavantage.co/query", params={
        "function": "NEWS_SENTIMENT",
        "tickers": ticker,
        "limit": MAX_ARTICLES,
        "time_from": time_from,
        "time_to": time_to,
        "apikey": API_KEY,
    })
    data = response.json()

    if 'feed' not in data:
        raise NewsUnavailable(data)

    return data['feed']


def cache_path(ticker, time_from, time_to):
    return os.path.join(CACHE_DIR, f'news_{ticker}_{time_from}_{time_to}.json')


def get_news_cached(ticker, time_from, time_to, allow_api=True, cacheable=True):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = cache_path(ticker, time_from, time_to)

    if cacheable and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    if not allow_api:
        return None

    articles = get_news(ticker, time_from, time_to)
    if cacheable:
        with open(path, 'w') as f:
            json.dump(articles, f)
    return articles


def extract_scores(articles, ticker):
    results = []
    for article in articles:
        for ts in article['ticker_sentiment']:
            if ts['ticker'] == ticker:
                results.append({
                    'time': article['time_published'],
                    'score': float(ts['ticker_sentiment_score']),
                    'relevance': float(ts['relevance_score']),
                })
    return results


def collect_scores(ticker, start, end):
    rows, covered = [], []
    allow_api = True
    now = pd.Timestamp(datetime.now()).floor('min')
    months = pd.date_range(start.replace(day=1), end, freq='MS')

    for month in reversed(months):
        next_month = month + pd.offsets.MonthBegin(1)
        # the current month isn't over yet: ask only up to now and don't freeze it in the cache
        finished = next_month <= now
        time_from = month.strftime('%Y%m%dT%H%M')
        time_to = min(next_month, now).strftime('%Y%m%dT%H%M')

        try:
            articles = get_news_cached(ticker, time_from, time_to, allow_api, cacheable=finished)
        except NewsUnavailable as e:
            print(f'News API stopped at {month:%Y-%m}, using cache only from here: {e}')
            allow_api = False
            articles = None

        if articles is None:
            continue
        if len(articles) >= MAX_ARTICLES:
            print(f'Warning: {month:%Y-%m} hit the {MAX_ARTICLES}-article cap, some news is missing')

        rows += extract_scores(articles, ticker)
        covered.append((month, next_month))

    return pd.DataFrame(rows, columns=['time', 'score', 'relevance']), covered


def daily_sentiment(news, covered, trading_days):
    daily = pd.DataFrame(index=trading_days, columns=['Sent_Mean', 'News_Count'], dtype=float)
    for start, end in covered:
        daily.loc[(trading_days >= start) & (trading_days < end), 'News_Count'] = 0

    if news.empty:
        return daily

    published = pd.to_datetime(news['time'], format='%Y%m%dT%H%M%S')
    # news after the 4pm close can only move the next session
    day = published.dt.normalize() + pd.to_timedelta((published.dt.hour >= 16).astype(int), unit='D')
    pos = trading_days.searchsorted(day)

    news = news[pos < len(trading_days)].copy()
    news['date'] = trading_days[pos[pos < len(trading_days)]]
    news['weighted'] = news['score'] * news['relevance']

    per_day = news.groupby('date').agg(weighted=('weighted', 'sum'),
                                       relevance=('relevance', 'sum'),
                                       count=('score', 'size'))
    daily.loc[per_day.index, 'Sent_Mean'] = per_day['weighted'] / per_day['relevance']
    daily.loc[per_day.index, 'News_Count'] = per_day['count']
    return daily
