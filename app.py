"""
Flask Web App v2 - XGBoost Stock Market Dashboard
with per-ticker selection support and SHAP XAI
Dynamic MySQL Database Integration
"""
import os
import json
import numpy as np
import pickle
import xgboost as xgb
import shap
import pandas as pd
from sqlalchemy import create_engine, text
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

MODEL_PATH   = 'xgb_model.json'
METRICS_PATH = 'model_metrics.json'
FEATURES_PATH= 'feature_cols.pkl'
TICKER_PATH  = 'ticker_stats.json'

DB_HOST = "localhost"
DB_USER = "root"
DB_PASS = "170804"
DB_NAME = "kltn_stock_db"

FEATURE_LABELS_VI = {
    'price_vs_sma50': 'Giá so với đường trung bình SMA 50',
    'volatility_20': 'Độ biến động giá 20 phiên',
    'volume_ratio_20': 'Tỷ lệ khối lượng giao dịch 20 phiên',
    'return_3d': 'Tỷ suất sinh lời 3 ngày',
    'return_5d': 'Tỷ suất sinh lời 5 ngày',
    'return_10d': 'Tỷ suất sinh lời 10 ngày',
    'return_20d': 'Tỷ suất sinh lời 20 ngày',
    'sma_50_LogReturn': 'Tốc độ tăng trưởng SMA 50',
    'volume_LogReturn': 'Tốc độ tăng trưởng khối lượng',
    'PCA_Trend': 'Thành phần xu hướng giá (PCA)',
    'PCA_Oscillators': 'Thành phần chỉ báo dao động (PCA)',
    'PCA_MACD': 'Thành phần MACD (PCA)',
    'PCA_ShortReturns': 'Thành phần tỷ suất ngắn hạn (PCA)',
    'atr_14': 'Chỉ số biên độ biến động ATR 14',
    'high_low': 'Biên độ dao động trong phiên (High - Low)',
    'market_return': 'Tỷ suất sinh lời của VN-Index',
    'rs': 'Chỉ số RRG RS-Ratio (Sức mạnh tương đối)',
    'rm': 'Chỉ số RRG RS-Momentum (Động lượng xoay vòng)'
}

_cached_resources = {
    'model': None,
    'metrics': {},
    'feature_cols': [],
    'ticker_stats': {},
    'explainer': None,
    'mtime_model': 0,
    'mtime_metrics': 0,
    'mtime_features': 0
}

def load_resources():
    global _cached_resources
    
    try:
        mtime_model = os.path.getmtime(MODEL_PATH) if os.path.exists(MODEL_PATH) else 0
        mtime_metrics = os.path.getmtime(METRICS_PATH) if os.path.exists(METRICS_PATH) else 0
        mtime_features = os.path.getmtime(FEATURES_PATH) if os.path.exists(FEATURES_PATH) else 0
    except Exception:
        mtime_model = mtime_metrics = mtime_features = 0

    if _cached_resources['model'] is None or mtime_model != _cached_resources['mtime_model']:
        if os.path.exists(MODEL_PATH):
            model = xgb.XGBClassifier()
            model.load_model(MODEL_PATH)
            _cached_resources['model'] = model
            _cached_resources['mtime_model'] = mtime_model
            _cached_resources['explainer'] = shap.TreeExplainer(model)
            
    if not _cached_resources['metrics'] or mtime_metrics != _cached_resources['mtime_metrics']:
        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH, 'r', encoding='utf-8') as f:
                _cached_resources['metrics'] = json.load(f)
            _cached_resources['mtime_metrics'] = mtime_metrics
            
    if not _cached_resources['feature_cols'] or mtime_features != _cached_resources['mtime_features']:
        if os.path.exists(FEATURES_PATH):
            with open(FEATURES_PATH, 'rb') as f:
                _cached_resources['feature_cols'] = pickle.load(f)
            _cached_resources['mtime_features'] = mtime_features
            
    if not _cached_resources['ticker_stats']:
        try:
            if os.path.exists(TICKER_PATH):
                print("Loading ticker statistics from JSON...")
                with open(TICKER_PATH, 'r', encoding='utf-8') as f:
                    _cached_resources['ticker_stats'] = json.load(f)
                print("Loaded ticker stats from JSON successfully.")
            else:
                raise FileNotFoundError("ticker_stats.json not found")
        except Exception as e:
            print(f"Error loading ticker stats from JSON: {e}. Querying MySQL...")
            try:
                engine = create_engine(f"mysql+mysqlconnector://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}")
                df = pd.read_sql("SELECT ticker, date, price_vs_sma50, volatility_20, volume_ratio_20, return_3d, return_5d, return_10d, return_20d, sma_50_LogReturn, volume_LogReturn, PCA_Trend, PCA_Oscillators, PCA_MACD, PCA_ShortReturns, atr_14, high_low, market_return, rs, rm FROM model_training_data", engine)
                
                ticker_stats = {}
                for ticker, group in df.groupby('ticker'):
                    ticker_stats[ticker] = {
                        'median_features': group.drop(columns=['ticker', 'date']).median().to_dict(),
                        'total_samples': len(group),
                        'latest_date': str(group['date'].max()),
                        'test_accuracy': 0.51,
                        'predictability': 'medium'
                    }
                _cached_resources['ticker_stats'] = ticker_stats
                print("Loaded ticker stats from MySQL successfully.")
            except Exception as ex:
                print(f"Failed to query MySQL: {ex}")
            
    return (
        _cached_resources['model'],
        _cached_resources['metrics'],
        _cached_resources['feature_cols'],
        _cached_resources['ticker_stats'],
        _cached_resources['explainer']
    )


@app.route('/')
def index():
    _, metrics, _, _, _ = load_resources()
    return render_template('index.html', metrics=metrics)


@app.route('/api/metrics')
def api_metrics():
    _, metrics, _, _, _ = load_resources()
    return jsonify(metrics)


@app.route('/api/tickers')
def api_tickers():
    """Trả về danh sách tất cả mã cổ phiếu kèm theo dự báo AI mới nhất từ MySQL"""
    model, _, feature_cols, ticker_stats, _ = load_resources()
    if model is None:
        return jsonify({'error': 'Model chua duoc training'}), 400
        
    threshold_offset = request.args.get('threshold_offset', default=0.0, type=float)
        
    try:
        engine = create_engine(f"mysql+mysqlconnector://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}")
        with engine.connect() as conn:
            df_pred = pd.read_sql("""
                SELECT p.* 
                FROM model_predictions p
                INNER JOIN (
                    SELECT ticker, MAX(date) as max_date 
                    FROM model_predictions 
                    GROUP BY ticker
                ) m ON p.ticker = m.ticker AND p.date = m.max_date
            """, conn)
            
        results = []
        for _, row in df_pred.iterrows():
            ticker = row['ticker']
            pred = int(row['prediction'])
            prob_up = float(row['probability_up'])
            prob_down = float(row['probability_down'])
            conf = max(prob_up, prob_down)
            
            # Lọc theo độ tự tin
            if threshold_offset > 0.0:
                if 0.5 - threshold_offset < prob_up < 0.5 + threshold_offset:
                    continue
            
            # Fetch stats
            stats = ticker_stats.get(ticker, {})
            medians = stats.get('median_features', {})
            total_samples = stats.get('total_samples', 0)
            test_accuracy = stats.get('test_accuracy', 0.51)
            predictability = stats.get('predictability', 'medium')
            
            results.append({
                'ticker': ticker,
                'prediction': pred,
                'label': 'Tang (Mua)' if pred == 1 else 'Giam (Ban)',
                'probability_up': round(prob_up, 4),
                'probability_down': round(prob_down, 4),
                'confidence': round(conf, 4),
                'total_samples': total_samples,
                'latest_date': str(row['date']),
                'predict_date': str(row['predict_date']),
                'median_features': medians,
                'test_accuracy': round(test_accuracy, 4),
                'predictability': predictability
            })
            
        results.sort(key=lambda x: x['ticker'])
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/ticker/<ticker_name>')
def api_ticker_detail(ticker_name):
    """Trả về thông tin dự báo AI và SHAP của một mã dựa trên phiên giao dịch mới nhất từ MySQL"""
    model, _, feature_cols, ticker_stats, explainer = load_resources()
    ticker_upper = ticker_name.upper()
    
    try:
        engine = create_engine(f"mysql+mysqlconnector://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}")
        with engine.connect() as conn:
            # Query latest row from daily_normalized_data
            df_norm = pd.read_sql(text("""
                SELECT * FROM daily_normalized_data 
                WHERE ticker = :ticker 
                ORDER BY date DESC LIMIT 1
            """), conn, params={'ticker': ticker_upper})
            
            # Query prediction details
            df_pred = pd.read_sql(text("""
                SELECT * FROM model_predictions 
                WHERE ticker = :ticker 
                ORDER BY date DESC LIMIT 1
            """), conn, params={'ticker': ticker_upper})
            
        if df_norm.empty or df_pred.empty:
            return jsonify({'error': f'Khong tim thay du lieu cho ma {ticker_upper}'}), 404
            
        latest_row = df_norm.iloc[0]
        pred_row = df_pred.iloc[0]
        
        values = [float(latest_row[f]) for f in feature_cols]
        X = np.array([values])
        pred = int(pred_row['prediction'])
        prob_up = float(pred_row['probability_up'])
        prob_down = float(pred_row['probability_down'])
        confidence = max(prob_up, prob_down)
        
        # Calculate SHAP values
        shap_explanations = []
        p_base = 0.5
        if explainer is not None:
            shap_vals = explainer.shap_values(X)[0]
            sum_shap = float(np.sum(shap_vals))
            
            # Baseline probability
            base_val = float(explainer.expected_value[0] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value)
            p_base = 1.0 / (1.0 + np.exp(-base_val))
            p_pred = prob_up
            delta_p = p_pred - p_base
            
            for feat, val, sv in zip(feature_cols, values, shap_vals):
                label_vi = FEATURE_LABELS_VI.get(feat, feat)
                
                # Proportional probability contribution
                prob_contrib = 0.0
                if abs(sum_shap) > 1e-6:
                    prob_contrib = delta_p * (sv / sum_shap)
                    
                shap_explanations.append({
                    'feature': feat,
                    'label': label_vi,
                    'value': round(val, 4),
                    'shap_value': round(float(sv), 6),
                    'prob_contribution': round(float(prob_contrib), 6),
                    'impact': 'positive' if sv > 0 else 'negative'
                })
            # Sort by absolute SHAP value descending
            shap_explanations.sort(key=lambda x: abs(x['shap_value']), reverse=True)
            
        stats = ticker_stats.get(ticker_upper, {})
        medians = stats.get('median_features', {})
        total_samples = stats.get('total_samples', 0)
        test_accuracy = stats.get('test_accuracy', 0.51)
        predictability = stats.get('predictability', 'medium')
        
        return jsonify({
            'ticker': ticker_upper,
            'total_samples': total_samples,
            'prediction': pred,
            'label': 'Tang (Mua)' if pred == 1 else 'Giam (Ban)',
            'probability_up': round(prob_up, 4),
            'probability_down': round(prob_down, 4),
            'confidence': round(confidence, 4),
            'p_base': round(p_base, 4),
            'shap_features': shap_explanations,
            'median_features': medians,
            'latest_date': str(pred_row['date']),
            'predict_date': str(pred_row['predict_date']),
            'test_accuracy': round(test_accuracy, 4),
            'predictability': predictability
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict', methods=['POST'])
def predict():
    model, metrics, feature_cols, _, explainer = load_resources()
    if model is None:
        return jsonify({'error': 'Model chua duoc training'}), 400

    data = request.json
    try:
        values = [float(data.get(f, 0)) for f in feature_cols]
        X = np.array([values])
        pred  = int(model.predict(X)[0])
        proba = model.predict_proba(X)[0].tolist()
        
        # Calculate SHAP values
        shap_explanations = []
        p_base = 0.5
        if explainer is not None:
            shap_vals = explainer.shap_values(X)[0]
            sum_shap = float(np.sum(shap_vals))
            
            # Baseline probability
            base_val = float(explainer.expected_value[0] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value)
            p_base = 1.0 / (1.0 + np.exp(-base_val))
            p_pred = proba[1]
            delta_p = p_pred - p_base
            
            for feat, val, sv in zip(feature_cols, values, shap_vals):
                label_vi = FEATURE_LABELS_VI.get(feat, feat)
                
                # Proportional probability contribution
                prob_contrib = 0.0
                if abs(sum_shap) > 1e-6:
                    prob_contrib = delta_p * (sv / sum_shap)
                    
                shap_explanations.append({
                    'feature': feat,
                    'label': label_vi,
                    'value': round(val, 4),
                    'shap_value': round(float(sv), 6),
                    'prob_contribution': round(float(prob_contrib), 6),
                    'impact': 'positive' if sv > 0 else 'negative'
                })
            # Sort by absolute SHAP value descending
            shap_explanations.sort(key=lambda x: abs(x['shap_value']), reverse=True)

        return jsonify({
            'prediction':       pred,
            'label':            'Tang (Mua)' if pred == 1 else 'Giam (Ban)',
            'probability_up':   round(proba[1], 4),
            'probability_down': round(proba[0], 4),
            'confidence':       round(max(proba), 4),
            'p_base':           round(p_base, 4),
            'shap_features':    shap_explanations
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 400


@app.route('/api/portfolio/optimize', methods=['POST'])
def api_portfolio_optimize():
    model, _, feature_cols, ticker_stats, explainer = load_resources()
    if model is None:
        return jsonify({'error': 'Model chua duoc training'}), 400

    data = request.json or {}
    capital = float(data.get('capital', 100000000.0))
    risk_appetite = data.get('risk_appetite', 'balanced') # safe, balanced, growth, custom
    target_return_min = float(data.get('target_return', 15.0)) / 100.0 # e.g. 15% -> 0.15
    selected_tickers = data.get('selected_tickers', [])
    
    if not selected_tickers or len(selected_tickers) < 2:
        return jsonify({'error': 'Vui lòng chọn ít nhất 2 mã cổ phiếu để tối ưu danh mục.'}), 400
        
    # Standardize tickers to uppercase
    selected_tickers = [t.upper() for t in selected_tickers]
    
    # Check if all tickers are in our database
    valid_tickers = [t for t in selected_tickers if t in ticker_stats]
    if len(valid_tickers) < 2:
        return jsonify({'error': 'Không đủ mã cổ phiếu hợp lệ trong cơ sở dữ liệu.'}), 400
        
    try:
        # 1. Load historical training returns from MySQL instead of data2.xlsx
        engine = create_engine(f"mysql+mysqlconnector://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}")
        tickers_str = ", ".join([f"'{t}'" for t in valid_tickers])
        with engine.connect() as conn:
            df_sel = pd.read_sql(text(f"""
                SELECT ticker, date, close_LogReturn 
                FROM model_training_data 
                WHERE ticker IN ({tickers_str})
            """), conn)
        
        if df_sel.empty:
            return jsonify({'error': 'Không thể tải dữ liệu lịch sử cho các mã cổ phiếu đã chọn.'}), 500
            
        pivot_df = df_sel.pivot(index='date', columns='ticker', values='close_LogReturn')
        # Drop columns with all NaN or fill them
        pivot_df = pivot_df.dropna(how='all').fillna(0.0)
        
        # Clip outliers to [-0.15, 0.15] to prevent splits/adjustment anomalies
        pivot_df = pivot_df.clip(-0.15, 0.15)
        
        # Convert log returns to simple returns
        returns = np.exp(pivot_df) - 1
        
        # Compute mean and covariance matrix
        mean_returns = returns.mean() * 252 # Annualized
        cov_matrix = returns.cov() * 252 # Annualized
        
        # 2. Get AI predictions for selected tickers to adjust returns
        adjusted_returns = mean_returns.copy()
        ticker_details_map = {}
        
        for t in valid_tickers:
            stats = ticker_stats[t]
            medians = stats.get('median_features', {})
            values = [float(medians.get(f, 0)) for f in feature_cols]
            X = np.array([values])
            proba = model.predict_proba(X)[0].tolist()
            prob_up = proba[1]
            
            # Adjust expected return: E(R) = historical + 0.3 * (prob_up - 0.5)
            adjusted_returns[t] = mean_returns[t] + 0.3 * (prob_up - 0.5)
            
            # Store predictions & SHAP values for narrative generation
            shap_explanations = []
            if explainer is not None:
                shap_vals = explainer.shap_values(X)[0]
                sum_shap = float(np.sum(shap_vals))
                base_val = float(explainer.expected_value[0] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value)
                p_base = 1.0 / (1.0 + np.exp(-base_val))
                delta_p = prob_up - p_base
                
                for feat, val, sv in zip(feature_cols, values, shap_vals):
                    label_vi = FEATURE_LABELS_VI.get(feat, feat)
                    prob_contrib = 0.0
                    if abs(sum_shap) > 1e-6:
                        prob_contrib = delta_p * (sv / sum_shap)
                    shap_explanations.append({
                        'feature': feat,
                        'label': label_vi,
                        'value': val,
                        'shap_value': sv,
                        'prob_contribution': prob_contrib,
                        'impact': 'positive' if sv > 0 else 'negative'
                    })
                shap_explanations.sort(key=lambda x: abs(x['shap_value']), reverse=True)
                
            ticker_details_map[t] = {
                'prob_up': prob_up,
                'pred_label': 'Tang (Mua)' if prob_up > 0.5 else 'Giam (Ban)',
                'shap': shap_explanations
            }
            
        # 3. Monte Carlo Simulation (2000 portfolios)
        n_simulations = 2000
        n_assets = len(valid_tickers)
        results = []
        weights_list = []
        
        np.random.seed(42) # Deterministic results
        
        for _ in range(n_simulations):
            w = np.random.dirichlet(np.ones(n_assets))
            port_ret = float(np.dot(w, adjusted_returns))
            port_vol = float(np.sqrt(np.dot(w, np.dot(cov_matrix.values, w))))
            rf = 0.045 # Risk-free rate 4.5%
            sharpe = (port_ret - rf) / port_vol if port_vol > 0 else 0
            results.append((port_ret, port_vol, sharpe))
            weights_list.append(w)
            
        results_arr = np.array(results)
        
        # 4. Find the optimal portfolio based on appetite
        if risk_appetite == 'safe':
            idx = results_arr[:, 1].argmin()
        elif risk_appetite == 'growth':
            idx = results_arr[:, 0].argmax()
        elif risk_appetite == 'custom':
            valid_indices = np.where(results_arr[:, 0] >= target_return_min)[0]
            if len(valid_indices) > 0:
                sub_vols = results_arr[valid_indices, 1]
                idx = valid_indices[sub_vols.argmin()]
            else:
                idx = results_arr[:, 0].argmax()
        else: # balanced
            idx = results_arr[:, 2].argmax()
            
        opt_weights = weights_list[idx]
        opt_return = results_arr[idx, 0]
        opt_volatility = results_arr[idx, 1]
        opt_sharpe = results_arr[idx, 2]
        
        # 5. Fetch actual latest prices from MySQL instead of hardcoding or hashing
        prices = {}
        try:
            tickers_str = ", ".join([f"'{t}'" for t in valid_tickers])
            with engine.connect() as conn:
                df_prices = pd.read_sql(text(f"""
                    SELECT r.ticker, r.close 
                    FROM daily_raw_data r
                    INNER JOIN (
                        SELECT ticker, MAX(date) as max_date 
                        FROM daily_raw_data 
                        GROUP BY ticker
                    ) m ON r.ticker = m.ticker AND r.date = m.max_date
                    WHERE r.ticker IN ({tickers_str})
                """), conn)
                for _, row in df_prices.iterrows():
                    prices[row['ticker']] = float(row['close'])
        except Exception as price_err:
            print(f"Error fetching live prices: {price_err}")
            
        def get_ticker_price(ticker):
            if ticker in prices:
                return prices[ticker]
            # fallback
            return 20000.0
            
        holdings = []
        total_allocated = 0.0
        
        for t, w in zip(valid_tickers, opt_weights):
            price = get_ticker_price(t)
            target_value = capital * w
            
            qty = round(target_value / price / 100) * 100
            actual_value = qty * price
            
            detail = ticker_details_map[t]
            p_up = detail['prob_up']
            shaps = detail['shap']
            
            narrative = f"Phân bổ {w*100:.1f}% danh mục. "
            if w > 0.05:
                if p_up > 0.5:
                    pos_drivers = [s['label'] for s in shaps if s['impact'] == 'positive'][:2]
                    narrative += f"AI dự báo Tăng (xác suất {p_up*100:.1f}%). Động lực chính: {', '.join(pos_drivers)}."
                else:
                    neg_drivers = [s['label'] for s in shaps if s['impact'] == 'negative'][:2]
                    narrative += f"AI dự báo Giảm (xác suất {(1-p_up)*100:.1f}%). Yếu tố cản trở: {', '.join(neg_drivers)}."
            else:
                narrative += f"Hạn chế tỷ trọng tối đa để giảm thiểu rủi ro cho danh mục chung."
                
            holdings.append({
                'ticker': t,
                'price': price,
                'target_weight': round(float(w) * 100, 2),
                'actual_weight': 0.0,
                'quantity': int(qty),
                'allocated_amount': float(actual_value),
                'xai_explanation': narrative
            })
            total_allocated += actual_value
            
        for h in holdings:
            h['actual_weight'] = round((h['allocated_amount'] / total_allocated * 100), 2) if total_allocated > 0 else 0.0
            
        cash_left = capital - total_allocated
        
        frontier_points = []
        step = max(1, n_simulations // 120)
        for i in range(0, n_simulations, step):
            frontier_points.append({
                'volatility': round(float(results_arr[i, 1]) * 100, 2),
                'expected_return': round(float(results_arr[i, 0]) * 100, 2),
                'sharpe': round(float(results_arr[i, 2]), 3)
            })
            
        optimal_point = {
            'volatility': round(float(opt_volatility) * 100, 2),
            'expected_return': round(float(opt_return) * 100, 2),
            'sharpe': round(float(opt_sharpe), 3)
        }
        
        appetite_labels = {
            'safe': 'Tối thiểu rủi ro (Minimum Volatility)',
            'balanced': 'Cân bằng hiệu quả (Maximum Sharpe Ratio)',
            'growth': 'Tối đa hóa lợi nhuận (Maximum Return)',
            'custom': 'Tùy chỉnh mục tiêu lợi nhuận'
        }
        
        portfolio_narrative = (
            f"Danh mục được tối ưu theo tiêu chí <strong>{appetite_labels.get(risk_appetite, risk_appetite)}</strong>. "
            f"Từ số vốn ban đầu <strong>{capital:,.0f} VND</strong>, hệ thống đề xuất phân bổ thực tế "
            f"<strong>{total_allocated:,.0f} VND</strong> vào các cổ phiếu được chọn và giữ lại <strong>{cash_left:,.0f} VND</strong> tiền mặt để tuân thủ quy định lô giao dịch 100 của sàn HOSE. "
            f"Danh mục tối ưu đạt tỷ suất sinh lời kỳ vọng <strong>{(opt_return*100):.2f}% / năm</strong> với độ lệch chuẩn rủi ro (Volatility) là <strong>{(opt_volatility*100):.2f}% / năm</strong>, "
            f"đạt chỉ số Sharpe <strong>{opt_sharpe:.2f}</strong>."
        )
        
        return jsonify({
            'capital': capital,
            'total_allocated': total_allocated,
            'cash_left': cash_left,
            'expected_return': round(float(opt_return) * 100, 2),
            'volatility': round(float(opt_volatility) * 100, 2),
            'sharpe_ratio': round(float(opt_sharpe), 3),
            'holdings': holdings,
            'efficient_frontier': frontier_points,
            'optimal_point': optimal_point,
            'portfolio_narrative': portfolio_narrative
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
