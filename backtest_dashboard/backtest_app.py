"""
Forecast Verification Dashboard
Backtest & Real-time Tracking — Flask App on port 5001
Connects to kltn_stock_db MySQL database
"""
import os
import sys
import numpy as np
import pickle
import xgboost as xgb
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from flask import Flask, render_template, request, jsonify
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

app = Flask(__name__)

DB_HOST = "localhost"
DB_USER = "root"
DB_PASS = "170804"
DB_NAME = "kltn_stock_db"

# Paths to model artifacts (in parent directory)
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PARENT_DIR, 'xgb_model.json')
FEATURES_PATH = os.path.join(PARENT_DIR, 'feature_cols.pkl')

_engine = None

def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(f"mysql+mysqlconnector://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}")
    return _engine


def ensure_verification_table():
    """Create prediction_verification table if not exists"""
    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS prediction_verification (
                ticker VARCHAR(10) NOT NULL,
                prediction_date DATE NOT NULL,
                predict_target_date DATE NOT NULL,
                prediction INT,
                probability_up DOUBLE,
                confidence DOUBLE,
                actual_outcome INT DEFAULT NULL,
                is_correct TINYINT DEFAULT NULL,
                actual_return DOUBLE DEFAULT NULL,
                verified_at DATETIME DEFAULT NULL,
                PRIMARY KEY (ticker, prediction_date)
            ) ENGINE=InnoDB;
        """))
        conn.commit()


def load_model():
    """Load XGBoost model and feature columns"""
    if not os.path.exists(MODEL_PATH):
        return None, []
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    feature_cols = []
    if os.path.exists(FEATURES_PATH):
        with open(FEATURES_PATH, 'rb') as f:
            feature_cols = pickle.load(f)
    return model, feature_cols


# ═══════════════════════════════════════════════════════════
#  ROUTES
# ═══════════════════════════════════════════════════════════

@app.route('/')
def index():
    return render_template('backtest.html')


# ═══════════════════════════════════════════════════════════
#  BACKTEST APIs
# ═══════════════════════════════════════════════════════════

@app.route('/api/backtest/summary')
def backtest_summary():
    """Run backtest on test set (last 20% by time) and return overall metrics"""
    model, feature_cols = load_model()
    if model is None:
        return jsonify({'error': 'Model not found'}), 400

    engine = get_engine()
    df = pd.read_sql(text("SELECT * FROM model_training_data ORDER BY date"), engine)

    if df.empty:
        return jsonify({'error': 'No training data'}), 400

    # Recreate target from close_LogReturn (same logic as train_model.py)
    # target column already exists in the table
    df = df.dropna(subset=['target'])

    # Time-based split: last 20%
    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:].copy()

    X_test = test_df[feature_cols].values
    y_true = test_df['target'].values.astype(int)

    y_pred = model.predict(X_test).astype(int)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_proba)
    except:
        auc = 0.0

    cm = confusion_matrix(y_true, y_pred).tolist()

    return jsonify({
        'total_samples': len(test_df),
        'accuracy': round(acc, 4),
        'precision': round(prec, 4),
        'recall': round(rec, 4),
        'f1_score': round(f1, 4),
        'auc_roc': round(auc, 4),
        'confusion_matrix': cm,
        'test_date_range': {
            'start': str(test_df['date'].min()),
            'end': str(test_df['date'].max())
        },
        'class_distribution': {
            'up': int((y_true == 1).sum()),
            'down': int((y_true == 0).sum())
        }
    })


@app.route('/api/backtest/by-ticker')
def backtest_by_ticker():
    """Accuracy breakdown by ticker on test set"""
    model, feature_cols = load_model()
    if model is None:
        return jsonify({'error': 'Model not found'}), 400

    engine = get_engine()
    df = pd.read_sql(text("SELECT * FROM model_training_data ORDER BY date"), engine)
    df = df.dropna(subset=['target'])

    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:].copy()

    test_df['y_pred'] = model.predict(test_df[feature_cols].values).astype(int)
    test_df['y_proba'] = model.predict_proba(test_df[feature_cols].values)[:, 1]

    results = []
    for ticker, grp in test_df.groupby('ticker'):
        y_true = grp['target'].values.astype(int)
        y_pred = grp['y_pred'].values
        if len(y_true) < 5:
            continue
        acc = accuracy_score(y_true, y_pred)
        correct = int((y_true == y_pred).sum())
        total = len(y_true)
        up_preds = int((y_pred == 1).sum())
        down_preds = int((y_pred == 0).sum())

        results.append({
            'ticker': ticker,
            'accuracy': round(acc, 4),
            'correct': correct,
            'total': total,
            'up_predictions': up_preds,
            'down_predictions': down_preds
        })

    results.sort(key=lambda x: x['accuracy'], reverse=True)
    return jsonify(results)


@app.route('/api/backtest/timeline')
def backtest_timeline():
    """Monthly rolling accuracy to detect model drift"""
    model, feature_cols = load_model()
    if model is None:
        return jsonify({'error': 'Model not found'}), 400

    engine = get_engine()
    df = pd.read_sql(text("SELECT * FROM model_training_data ORDER BY date"), engine)
    df = df.dropna(subset=['target'])

    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:].copy()
    test_df['y_pred'] = model.predict(test_df[feature_cols].values).astype(int)

    test_df['month'] = pd.to_datetime(test_df['date']).dt.to_period('M').astype(str)

    timeline = []
    for month, grp in test_df.groupby('month'):
        y_true = grp['target'].values.astype(int)
        y_pred = grp['y_pred'].values
        acc = accuracy_score(y_true, y_pred)
        timeline.append({
            'month': month,
            'accuracy': round(acc, 4),
            'total': len(grp),
            'correct': int((y_true == y_pred).sum())
        })

    return jsonify(timeline)


@app.route('/api/backtest/predictions')
def backtest_predictions():
    """Detailed prediction list with pagination"""
    model, feature_cols = load_model()
    if model is None:
        return jsonify({'error': 'Model not found'}), 400

    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 50, type=int)
    ticker_filter = request.args.get('ticker', '').upper()

    engine = get_engine()
    df = pd.read_sql(text("SELECT * FROM model_training_data ORDER BY date"), engine)
    df = df.dropna(subset=['target'])

    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:].copy()
    test_df['y_pred'] = model.predict(test_df[feature_cols].values).astype(int)
    test_df['y_proba'] = model.predict_proba(test_df[feature_cols].values)[:, 1]
    test_df['correct'] = (test_df['target'] == test_df['y_pred']).astype(int)

    if ticker_filter:
        test_df = test_df[test_df['ticker'] == ticker_filter]

    # Sort by date descending
    test_df = test_df.sort_values('date', ascending=False)

    total = len(test_df)
    start = (page - 1) * per_page
    end = start + per_page
    page_df = test_df.iloc[start:end]

    predictions = []
    for _, row in page_df.iterrows():
        predictions.append({
            'ticker': row['ticker'],
            'date': str(row['date']),
            'predicted': int(row['y_pred']),
            'actual': int(row['target']),
            'correct': int(row['correct']),
            'probability_up': round(float(row['y_proba']), 4)
        })

    return jsonify({
        'predictions': predictions,
        'total': total,
        'page': page,
        'per_page': per_page,
        'total_pages': (total + per_page - 1) // per_page
    })


# ═══════════════════════════════════════════════════════════
#  REAL-TIME TRACKING APIs
# ═══════════════════════════════════════════════════════════

@app.route('/api/realtime/summary')
def realtime_summary():
    """Summary of real-time prediction tracking"""
    engine = get_engine()

    with engine.connect() as conn:
        # Sync predictions to verification table first
        conn.execute(text("""
            INSERT IGNORE INTO prediction_verification
                (ticker, prediction_date, predict_target_date, prediction, probability_up, confidence)
            SELECT ticker, date, predict_date, prediction, probability_up, confidence
            FROM model_predictions
        """))
        conn.commit()

        row = conn.execute(text("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN actual_outcome IS NOT NULL THEN 1 ELSE 0 END) as verified,
                SUM(CASE WHEN actual_outcome IS NULL THEN 1 ELSE 0 END) as pending,
                SUM(CASE WHEN is_correct = 1 THEN 1 ELSE 0 END) as correct,
                SUM(CASE WHEN is_correct = 0 THEN 1 ELSE 0 END) as wrong
            FROM prediction_verification
        """)).fetchone()

    total = int(row[0]) if row[0] else 0
    verified = int(row[1]) if row[1] else 0
    pending = int(row[2]) if row[2] else 0
    correct = int(row[3]) if row[3] else 0
    wrong = int(row[4]) if row[4] else 0
    hit_rate = round(correct / verified, 4) if verified > 0 else 0.0

    return jsonify({
        'total_predictions': total,
        'verified': verified,
        'pending': pending,
        'correct': correct,
        'wrong': wrong,
        'hit_rate': hit_rate
    })


@app.route('/api/realtime/predictions')
def realtime_predictions():
    """List all real-time predictions with verification status"""
    engine = get_engine()
    df = pd.read_sql(text("""
        SELECT * FROM prediction_verification
        ORDER BY prediction_date DESC, ticker
    """), engine)

    predictions = []
    for _, row in df.iterrows():
        status = 'pending'
        if row['is_correct'] == 1:
            status = 'correct'
        elif row['is_correct'] == 0:
            status = 'wrong'

        predictions.append({
            'ticker': row['ticker'],
            'prediction_date': str(row['prediction_date']),
            'target_date': str(row['predict_target_date']),
            'prediction': int(row['prediction']),
            'probability_up': round(float(row['probability_up']), 4) if row['probability_up'] else None,
            'confidence': round(float(row['confidence']), 4) if row['confidence'] else None,
            'actual_outcome': int(row['actual_outcome']) if row['actual_outcome'] is not None else None,
            'is_correct': int(row['is_correct']) if row['is_correct'] is not None else None,
            'actual_return': round(float(row['actual_return']), 4) if row['actual_return'] is not None else None,
            'status': status
        })

    return jsonify(predictions)


@app.route('/api/realtime/verify', methods=['POST'])
def realtime_verify():
    """Verify pending predictions against actual price data"""
    engine = get_engine()
    verified_count = 0
    errors = []

    with engine.connect() as conn:
        # Get all unverified predictions where target date has passed
        pending = conn.execute(text("""
            SELECT pv.ticker, pv.prediction_date, pv.predict_target_date, pv.prediction
            FROM prediction_verification pv
            WHERE pv.actual_outcome IS NULL
              AND pv.predict_target_date <= CURDATE()
        """)).fetchall()

        for row in pending:
            ticker = row[0]
            pred_date = row[1]
            target_date = row[2]
            prediction = row[3]

            # Get close price on prediction date and target date from daily_raw_data
            prices = conn.execute(text("""
                SELECT date, close FROM daily_raw_data
                WHERE ticker = :ticker
                  AND date IN (:pred_date, :target_date)
                ORDER BY date
            """), {'ticker': ticker, 'pred_date': pred_date, 'target_date': target_date}).fetchall()

            if len(prices) < 2:
                # Try to find nearest available dates
                price_range = conn.execute(text("""
                    SELECT date, close FROM daily_raw_data
                    WHERE ticker = :ticker
                      AND date BETWEEN :start AND :end
                    ORDER BY date
                """), {
                    'ticker': ticker,
                    'start': str(pred_date),
                    'end': str(target_date + timedelta(days=3))
                }).fetchall()

                if len(price_range) < 2:
                    errors.append(f"{ticker}: insufficient price data")
                    continue

                base_price = float(price_range[0][1])
                target_price = float(price_range[-1][1])
            else:
                base_price = float(prices[0][1])
                target_price = float(prices[1][1])

            actual_return = (target_price - base_price) / base_price
            actual_outcome = 1 if actual_return > 0 else 0
            is_correct = 1 if actual_outcome == prediction else 0

            conn.execute(text("""
                UPDATE prediction_verification
                SET actual_outcome = :outcome,
                    is_correct = :correct,
                    actual_return = :ret,
                    verified_at = NOW()
                WHERE ticker = :ticker AND prediction_date = :pred_date
            """), {
                'outcome': actual_outcome,
                'correct': is_correct,
                'ret': round(actual_return, 6),
                'ticker': ticker,
                'pred_date': pred_date
            })
            verified_count += 1

        conn.commit()

    return jsonify({
        'verified_count': verified_count,
        'errors': errors,
        'message': f'Đã xác minh {verified_count} dự báo thành công.'
    })


@app.route('/api/realtime/accuracy-chart')
def realtime_accuracy_chart():
    """Cumulative accuracy over prediction dates"""
    engine = get_engine()
    df = pd.read_sql(text("""
        SELECT prediction_date, is_correct
        FROM prediction_verification
        WHERE actual_outcome IS NOT NULL
        ORDER BY prediction_date
    """), engine)

    if df.empty:
        return jsonify([])

    # Group by date, compute cumulative accuracy
    chart_data = []
    cum_correct = 0
    cum_total = 0

    for date, grp in df.groupby('prediction_date'):
        cum_correct += int(grp['is_correct'].sum())
        cum_total += len(grp)
        chart_data.append({
            'date': str(date),
            'cumulative_accuracy': round(cum_correct / cum_total, 4),
            'daily_accuracy': round(grp['is_correct'].mean(), 4),
            'daily_total': len(grp),
            'cumulative_total': cum_total
        })

    return jsonify(chart_data)


# ═══════════════════════════════════════════════════════════
#  INTERACTIVE BACKTEST APIs
# ═══════════════════════════════════════════════════════════

@app.route('/api/backtest/available-dates')
def backtest_available_dates():
    """Get available test dates for a specific ticker"""
    ticker = request.args.get('ticker', '').upper()
    if not ticker:
        return jsonify({'error': 'ticker parameter required'}), 400

    engine = get_engine()
    df = pd.read_sql(text("""
        SELECT date FROM model_training_data
        WHERE ticker = :ticker AND target IS NOT NULL
        ORDER BY date
    """), engine, params={'ticker': ticker})

    if df.empty:
        return jsonify({'dates': [], 'tickers': []})

    # Only return dates in the test period (last 20%)
    all_data = pd.read_sql(text("SELECT DISTINCT date FROM model_training_data WHERE target IS NOT NULL ORDER BY date"), engine)
    split_idx = int(len(all_data) * 0.8)
    test_dates = set(all_data.iloc[split_idx:]['date'].astype(str).tolist())

    dates = [str(d) for d in df['date'] if str(d) in test_dates]

    return jsonify({'dates': dates})


@app.route('/api/backtest/tickers-list')
def backtest_tickers_list():
    """Get all unique tickers"""
    engine = get_engine()
    df = pd.read_sql(text("SELECT DISTINCT ticker FROM model_training_data ORDER BY ticker"), engine)
    return jsonify(df['ticker'].tolist())


@app.route('/api/backtest/simulate', methods=['POST'])
def backtest_simulate():
    """
    Simulate a prediction for a specific ticker on a specific date.
    Data is cut at the selected date to prevent look-ahead bias.
    Returns prediction, probability, feature values, and actual outcome.
    """
    model, feature_cols = load_model()
    if model is None:
        return jsonify({'error': 'Model not found'}), 400

    data = request.json
    ticker = data.get('ticker', '').upper()
    sim_date = data.get('date', '')

    if not ticker or not sim_date:
        return jsonify({'error': 'ticker and date are required'}), 400

    engine = get_engine()

    # Get the row for this ticker on this date
    df_row = pd.read_sql(text("""
        SELECT * FROM model_training_data
        WHERE ticker = :ticker AND date = :date
    """), engine, params={'ticker': ticker, 'date': sim_date})

    if df_row.empty:
        return jsonify({'error': f'Không tìm thấy dữ liệu cho {ticker} ngày {sim_date}'}), 404

    row = df_row.iloc[0]
    X = np.array([[float(row[f]) for f in feature_cols]])

    # Predict
    pred = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0].tolist()
    prob_up = proba[1]
    prob_down = proba[0]

    # Actual outcome from target column
    actual = int(row['target']) if pd.notna(row['target']) else None

    # Feature values for display
    features = []
    for f in feature_cols:
        features.append({
            'name': f,
            'value': round(float(row[f]), 4) if pd.notna(row[f]) else 0
        })

    # Get actual close price on sim_date and T+5
    # Find T+5 date (next 5 trading days)
    df_future = pd.read_sql(text("""
        SELECT date, close_LogReturn FROM model_training_data
        WHERE ticker = :ticker AND date > :date
        ORDER BY date LIMIT 5
    """), engine, params={'ticker': ticker, 'date': sim_date})

    target_date_str = None
    actual_return_pct = None
    if len(df_future) > 0:
        target_date_str = str(df_future['date'].iloc[-1])
        cum_return = df_future['close_LogReturn'].sum()
        actual_return_pct = round((np.exp(cum_return) - 1) * 100, 2)

    # Fetch close prices from daily_raw_data
    price_t0 = None
    price_t5 = None
    with engine.connect() as conn:
        res_t0 = conn.execute(text("SELECT close FROM daily_raw_data WHERE ticker = :ticker AND date = :date"),
                              {'ticker': ticker, 'date': sim_date}).fetchone()
        if res_t0:
            price_t0 = float(res_t0[0])

        if target_date_str:
            res_t5 = conn.execute(text("SELECT close FROM daily_raw_data WHERE ticker = :ticker AND date = :date"),
                                  {'ticker': ticker, 'date': target_date_str}).fetchone()
            if res_t5:
                price_t5 = float(res_t5[0])

    is_correct = None
    if actual is not None:
        is_correct = 1 if pred == actual else 0

    return jsonify({
        'ticker': ticker,
        'date': sim_date,
        'target_date': target_date_str,
        'prediction': pred,
        'label': 'Tăng (Mua)' if pred == 1 else 'Giảm (Bán)',
        'probability_up': round(prob_up, 4),
        'probability_down': round(prob_down, 4),
        'confidence': round(max(prob_up, prob_down), 4),
        'actual_outcome': actual,
        'actual_label': ('Tăng' if actual == 1 else 'Giảm') if actual is not None else None,
        'is_correct': is_correct,
        'actual_return_pct': actual_return_pct,
        'price_t0': price_t0,
        'price_t5': price_t5,
        'features': features
    })


@app.route('/api/backtest/ticker-detail')
def backtest_ticker_detail():
    """Get all predictions for a specific ticker in the test set"""
    model, feature_cols = load_model()
    if model is None:
        return jsonify({'error': 'Model not found'}), 400

    ticker = request.args.get('ticker', '').upper()
    if not ticker:
        return jsonify({'error': 'ticker parameter required'}), 400

    engine = get_engine()
    df = pd.read_sql(text("SELECT * FROM model_training_data ORDER BY date"), engine)
    df = df.dropna(subset=['target'])

    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:].copy()
    ticker_df = test_df[test_df['ticker'] == ticker].copy()

    if ticker_df.empty:
        return jsonify({'error': f'Không tìm thấy dữ liệu test cho {ticker}'}), 404

    ticker_df['y_pred'] = model.predict(ticker_df[feature_cols].values).astype(int)
    ticker_df['y_proba'] = model.predict_proba(ticker_df[feature_cols].values)[:, 1]
    ticker_df['correct'] = (ticker_df['target'] == ticker_df['y_pred']).astype(int)

    y_true = ticker_df['target'].values.astype(int)
    y_pred = ticker_df['y_pred'].values
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    predictions = []
    for _, row in ticker_df.iterrows():
        predictions.append({
            'date': str(row['date']),
            'predicted': int(row['y_pred']),
            'actual': int(row['target']),
            'correct': int(row['correct']),
            'probability_up': round(float(row['y_proba']), 4)
        })

    return jsonify({
        'ticker': ticker,
        'accuracy': round(acc, 4),
        'precision': round(prec, 4),
        'recall': round(rec, 4),
        'f1_score': round(f1, 4),
        'total': len(ticker_df),
        'correct_count': int(ticker_df['correct'].sum()),
        'predictions': predictions
    })


# ═══════════════════════════════════════════════════════════
#  BOOT
# ═══════════════════════════════════════════════════════════

if __name__ == '__main__':
    ensure_verification_table()
    print("=" * 60)
    print("  FORECAST VERIFICATION DASHBOARD")
    print("  Backtest & Real-time Tracking")
    print("  http://localhost:5001")
    print("=" * 60)
    app.run(debug=True, port=5001)
