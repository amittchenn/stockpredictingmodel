import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

TICKER = 'MSFT'
YEARS_OF_HISTORY = 10

def get_price_data(ticker:str, years: int) -> pd.DataFrame:
    df = yf.download(ticker, period=f'{years}y', interval='1d', multi_level_index=False)
    df['log_return'] = np.log(df['Close']/df['Close'].shift(1))
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df
    
def get_quarterly_eps(ticker: str) -> pd.DataFrame:
    tick = yf.Ticker(ticker)
    income = tick.get_income_stmt(freq='quarterly')
    if income is None or income.empty:
        raise ValueError(f'No quarterly income statement data for {ticker}')
    eps_row = None
    for candidate in ['DilutedEPS', 'BasicEPS']:
        if candidate in income.index:
            eps_row = income.loc[candidate]
            break
    if eps_row is None:
        raise ValueError('Could not find EPS in quarterly income statement')
    eps_df = eps_row.sort_index().to_frame(name='quarterly_eps')
    eps_df.index = pd.to_datetime(eps_df.index)
    return eps_df

def get_future_eps(ticker: str) -> dict:
    tick = yf.Ticker(ticker)
    est= tick.get_earnings_estimate()
    if est is None or est.empty:
        raise ValueError(f'No earnings estimate data for {ticker}')
    if '+1y' not in est.index:
        raise ValueError(f'No +1y earnings estimate for {ticker}')
    return{
        'forward_eps': est.loc['+1y', 'avg'],
        'growth_rate': est.loc['+1y', 'growth']
    }

def build_data_set(ticker: str, years: int) -> pd.DataFrame:
    price = get_price_data(ticker, years)
    eps_f = get_future_eps(ticker)

    merged = price.copy()
    merged['forward_pe_ratio'] = merged['Close'] / eps_f['forward_eps']
    merged['forward_peg_ratio'] = merged['forward_pe_ratio'] / (eps_f['growth_rate'] * 100)
    merged['future_vol'] = merged['log_return'].rolling(5).std().shift(-5)

    merged['ewma_vol'] = np.sqrt((merged['log_return']**2).ewm(alpha=0.06).mean())
    return merged

def add_tech_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df['sma_10'] = ta.sma(df['Close'], length=10)
    df['sma_20'] = ta.sma(df['Close'], length=20)
    df['ema_10'] = ta.ema(df['Close'], length=10)
    df['rsi_14'] = ta.rsi(df['Close'], length=14)

    macd = ta.macd(df['Close'])
    df = pd.concat([df, macd], axis=1)

    bbands = ta.bbands(df['Close'], length=20)
    df = pd.concat([df, bbands], axis=1)

    df['close_vs_sma10'] = df['Close'] / df['sma_10'] - 1
    df['close_vs_sma20'] = df['Close'] / df['sma_20'] - 1
    df['sma10_vs_sma20'] = df['sma_10'] / df['sma_20'] - 1
    df['macd_norm']      = df['MACD_12_26_9'] / df['Close']

    df['10d_volatility'] = df['log_return'].rolling(10).std()
    df['return_lag_1']   = df['log_return'].shift(1)
    df['return_5d']      = df['Close'].pct_change(5)
    df['return_10d']     = df['Close'].pct_change(10)

    return df

def add_label(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    df = df.copy()
    future_vol = df['log_return'].rolling(horizon).std().shift(-horizon)
    df['target'] = np.log(future_vol) - np.log(df['10d_volatility'])   # change in log-vol
    return df

pd.set_option('display.max_columns', None)

result = build_data_set(TICKER, YEARS_OF_HISTORY)
result = add_tech_features(result)
result = add_label(result)

result_clean = result.dropna()
print(result_clean)
print(f"Rows before dropna: {len(result)}, after: {len(result_clean)}")

drop_cols = ['Close', 'High', 'Low', 'Open', 'Volume', 'target',
             'future_vol', 'ewma_vol',          # <-- add these
             'sma_10', 'sma_20', 'ema_10',
             'forward_pe_ratio', 'forward_peg_ratio',
             'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0',
             'MACD_12_26_9', 'MACDs_12_26_9']
feature_columns =  [c for c in result_clean.columns if c not in drop_cols]
X = result_clean[feature_columns]
Y = result_clean['target']
split_index = int(len(X)*0.8);
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
Y_train, Y_test = Y.iloc[:split_index], Y.iloc[split_index:]
print(f"Train: {len(X_train)} rows, Test: {len(X_test)} rows")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

mod = RandomForestRegressor(n_estimators=200, random_state=42)
mod.fit(X_train, Y_train)
Y_predict = mod.predict(X_test)

baseline_pred = np.zeros(len(Y_test))
print("Baseline MAE (persistence):", mean_absolute_error(Y_test, baseline_pred))
print("Model MAE:", mean_absolute_error(Y_test, Y_predict))
print("Model R²:", r2_score(Y_test, Y_predict))

tscv = TimeSeriesSplit(n_splits=5)
scores = cross_val_score(mod, X, Y, cv=tscv, scoring='r2')
print("CV R² scores:", scores)
print("Mean CV R²:", scores.mean())
test = result_clean.iloc[split_index:]      
truth = test['future_vol']

persistence_fc = test['10d_volatility']                    
ewma_fc= test['ewma_vol']             
rf_fc= test['10d_volatility'] * np.exp(Y_predict)  

print("\n--- Forecasting future 5-day volatility ---")
for name, fc in [('Persistence', persistence_fc),
                 ('EWMA',        ewma_fc),
                 ('RandomForest', rf_fc)]:
    print(f"{name:13s}  MAE: {mean_absolute_error(truth, fc):.5f}   R²: {r2_score(truth, fc):.3f}")

# --- Gradient boosting: many shallow trees, each correcting the last one's errors ---
gbr = GradientBoostingRegressor(
    n_estimators=300,      # more trees, smaller steps -> less overfitting on noisy rows
    learning_rate=0.03,
    max_depth=3,           # weak learners; the forest uses deep trees, boosting should not
    subsample=0.8,         # each tree sees 80% of rows, same variance reduction as bagging
    random_state=42,
)
gbr.fit(X_train, Y_train)
gbr_pred = gbr.predict(X_test)
gbr_fc = test['10d_volatility'] * np.exp(gbr_pred)

print("\n--- Gradient Boosting (volatility) ---")
print(f"Model MAE (log-vol change): {mean_absolute_error(Y_test, gbr_pred):.5f}")
print(f"Model R²  (log-vol change): {r2_score(Y_test, gbr_pred):.3f}")
print(f"{'GradBoost':13s}  MAE: {mean_absolute_error(truth, gbr_fc):.5f}   R²: {r2_score(truth, gbr_fc):.3f}")

gb_scores = cross_val_score(gbr, X, Y, cv=tscv, scoring='r2')
print("GB CV R² scores:", gb_scores)
print("Mean GB CV R²:", gb_scores.mean())

train = result_clean.iloc[:split_index]
test  = result_clean.iloc[split_index:]

def ewma_vol_for(lam):
    # EWMA volatility over the full history, for a given decay lambda
    return np.sqrt((result['log_return']**2).ewm(alpha=1 - lam).mean())

lambdas = [0.85, 0.90, 0.94, 0.97, 0.99]

print("\n--- Tuning EWMA lambda (scored on TRAIN) ---")
best_lam, best_mae = None, np.inf
for lam in lambdas:
    fc = ewma_vol_for(lam).reindex(train.index)
    mae = mean_absolute_error(train['future_vol'], fc)
    print(f"lambda={lam:.2f}   train MAE: {mae:.5f}")
    if mae < best_mae:
        best_mae, best_lam = mae, lam

print(f"\nBest lambda on train: {best_lam}")

fc_test = ewma_vol_for(best_lam).reindex(test.index)
print(f"Test MAE @ lambda={best_lam}: {mean_absolute_error(test['future_vol'], fc_test):.5f}")
print(f"Test R²  @ lambda={best_lam}: {r2_score(test['future_vol'], fc_test):.3f}")