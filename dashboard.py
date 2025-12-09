import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import feedparser
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from datetime import datetime, timedelta, time
import time as time_module 
import pytz
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import warnings
import pickle
warnings.filterwarnings('ignore')

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="Pro Market Terminal",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    .stApp { background-color: #0E1117; }
    div[data-testid="stMetric"] {
        background-color: #1A1C24;
        border: 1px solid #333;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .stPlotlyChart {
        background-color: #1A1C24;
        border: 1px solid #333;
        border-radius: 12px;
        padding: 15px;
    }
    .prediction-update {
        background: linear-gradient(90deg, #1A1C24 0%, #2A2C34 100%);
        padding: 10px;
        border-radius: 8px;
        border-left: 4px solid #00F0FF;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

# --- 2. DATA UNIVERSE ---
INDICES = {
    "NIFTY 50": "^NSEI",
    "NIFTY BANK": "^NSEBANK",
    "NIFTY IT": "^CNXIT",
    "SENSEX": "^BSESN"
}

CONSTITUENTS = {
    "NIFTY 50": ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "ITC.NS", "BHARTIARTL.NS", "LT.NS", "SBIN.NS", "HINDUNILVR.NS", "BAJFINANCE.NS", "MARUTI.NS", "AXISBANK.NS", "HCLTECH.NS", "TITAN.NS", "ASIANPAINT.NS", "SUNPHARMA.NS", "ULTRACEMCO.NS", "TATASTEEL.NS", "NTPC.NS", "POWERGRID.NS", "M&M.NS", "TATAMOTORS.NS", "ADANIENT.NS", "ADANIPORTS.NS", "BAJAJFINSV.NS", "COALINDIA.NS", "ONGC.NS", "GRASIM.NS", "JSWSTEEL.NS", "TECHM.NS", "HINDALCO.NS", "WIPRO.NS", "DIVISLAB.NS", "CIPLA.NS", "APOLLOHOSP.NS", "DRREDDY.NS", "EICHERMOT.NS", "NESTLEIND.NS", "TATACONSUM.NS", "BRITANNIA.NS", "SBILIFE.NS", "HDFCLIFE.NS", "HEROMOTOCO.NS", "KOTAKBANK.NS", "INDUSINDBK.NS", "BPCL.NS", "SHRIRAMFIN.NS", "LTIM.NS"],
    "NIFTY BANK": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "AXISBANK.NS", "KOTAKBANK.NS", "INDUSINDBK.NS", "BANKBARODA.NS", "PNB.NS", "IDFCFIRSTB.NS", "AUBANK.NS", "FEDERALBNK.NS", "BANDHANBNK.NS"],
    "NIFTY IT": ["TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS", "LTIM.NS", "PERSISTENT.NS", "COFORGE.NS", "MPHASIS.NS", "OFSS.NS"],
    "SENSEX": ["RELIANCE.BO", "HDFCBANK.BO", "ICICIBANK.BO", "INFY.BO", "ITC.BO", "TCS.BO", "L&T.BO", "AXISBANK.BO", "KOTAKBANK.BO", "HINDUNILVR.BO", "SBIN.BO", "BHARTIARTL.BO", "BAJFINANCE.BO", "TATASTEEL.BO", "M&M.BO", "MARUTI.BO", "TITAN.BO", "SUNPHARMA.BO", "NESTLEIND.BO", "ADANIENT.BO", "ULTRACEMCO.BO", "JSWSTEEL.BO", "POWERGRID.BO", "TATASTR.BO", "INDUSINDBK.BO", "NTPC.BO", "HCLTECH.BO", "TECHM.BO", "WIPRO.BO", "ASIANPAINT.BO"]
}

NEWS_FEEDS = ["https://www.livemint.com/rss/markets", "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms"]

# --- 3. MARKET STATUS & DATA ENGINE ---

def get_market_status():
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    if now.weekday() < 5 and (time(9,15) <= now.time() <= time(15,30)):
        return True, "🟢 LIVE", 60 
    return False, "🔴 CLOSED", 300

@st.cache_data(ttl=300)  # 5-min cache for intraday
def fetch_main_data(ticker):
    stock = yf.Ticker(ticker)
    try:
        hist_max = stock.history(period="2y", interval="1d")
        hist_intra = stock.history(period="1d", interval="5m")
        if hist_max.empty: return pd.DataFrame(), pd.DataFrame(), {}
        
        metrics = {
            "price": float(hist_max['Close'].iloc[-1]),
            "prev": float(hist_max['Close'].iloc[-2]),
            "open": float(hist_max['Open'].iloc[-1]),
            "high": float(hist_max['High'].iloc[-1]),
            "low": float(hist_max['Low'].iloc[-1]),
            "volatility": float(hist_max['Close'].pct_change().std() * 100)
        }
        return hist_max, hist_intra, metrics
    except: return pd.DataFrame(), pd.DataFrame(), {}

@st.cache_data(ttl=3600)
def fetch_macro_indicators():
    indicators = {'vix': 15.0, 'usdinr': 83.0, 'oil': 80.0}
    try:
        vix = yf.Ticker("^INDIAVIX").history(period="5d")
        if not vix.empty: indicators['vix'] = float(vix['Close'].iloc[-1])
        
        usdinr = yf.Ticker("INR=X").history(period="5d")
        if not usdinr.empty: indicators['usdinr'] = float(usdinr['Close'].iloc[-1])
        
        oil = yf.Ticker("CL=F").history(period="5d")
        if not oil.empty: indicators['oil'] = float(oil['Close'].iloc[-1])
    except: pass
    return indicators

def get_sentiment():
    sia = SentimentIntensityAnalyzer()
    articles = []
    for feed in NEWS_FEEDS:
        try:
            parsed = feedparser.parse(feed)
            for entry in parsed.entries[:5]: articles.append(entry.title)
        except: continue
    score = np.mean([sia.polarity_scores(a)['compound'] for a in articles]) if articles else 0
    return score, articles

@st.cache_data(ttl=3600)
def fetch_movers_batch(const_tickers):
    chunk_size = 10
    chunks = [const_tickers[i:i + chunk_size] for i in range(0, len(const_tickers), chunk_size)]
    all_data = []
    seen = set()
    
    prog = st.progress(0, "Scanning Market Depth...")
    for i, chunk in enumerate(chunks):
        try:
            data = yf.download(chunk, period="2d", group_by='ticker', progress=False, threads=False)
            for t in chunk:
                if t in seen: continue
                try:
                    if len(chunk) == 1: df_t = data
                    else: 
                        if t not in data.columns.levels[0]: continue
                        df_t = data[t]
                    if len(df_t) < 2: continue
                    
                    latest = float(df_t['Close'].iloc[-1])
                    prev = float(df_t['Close'].iloc[-2])
                    if pd.isna(latest) or prev == 0: continue
                    
                    all_data.append({
                        "Company": t.replace(".NS","").replace(".BO",""),
                        "Price": latest,
                        "Change %": ((latest - prev) / prev) * 100
                    })
                    seen.add(t)
                except: continue
            time_module.sleep(0.2)
            prog.progress((i + 1) / len(chunks))
        except: continue
    prog.empty()
    df = pd.DataFrame(all_data)
    if not df.empty: df = df.drop_duplicates(subset=['Company'])
    return df

# --- 4. INTRADAY FEATURE EXTRACTION ---

def extract_intraday_features(intraday_df, current_time_ist):
    """Extract real-time features from intraday data"""
    if intraday_df.empty or len(intraday_df) < 2:
        return None
    
    try:
        features = {}
        close = intraday_df['Close'].values
        volume = intraday_df['Volume'].values
        high = intraday_df['High'].values
        low = intraday_df['Low'].values
        
        # Price momentum
        features['return_5m'] = (close[-1] / close[-2] - 1) * 100 if len(close) >= 2 else 0
        features['return_30m'] = (close[-1] / close[-7] - 1) * 100 if len(close) >= 7 else 0
        features['return_1h'] = (close[-1] / close[-13] - 1) * 100 if len(close) >= 13 else 0
        
        # Volatility (recent)
        if len(close) >= 13:
            returns_1h = np.diff(close[-13:]) / close[-14:-1]
            features['volatility_1h'] = np.std(returns_1h) * 100
        else:
            features['volatility_1h'] = 0
        
        # Volume analysis
        if len(volume) >= 13:
            mean_vol = np.mean(volume[-13:])
            features['volume_ratio'] = volume[-1] / mean_vol if mean_vol > 0 else 1.0
        else:
            features['volume_ratio'] = 1.0
        
        # Price range
        if len(high) >= 13 and len(low) >= 13:
            recent_range = np.mean(high[-13:]) - np.mean(low[-13:])
            features['range_position'] = (close[-1] - np.mean(low[-13:])) / recent_range if recent_range > 0 else 0.5
        else:
            features['range_position'] = 0.5
        
        # Time-based features
        features['time_of_day'] = current_time_ist.hour + current_time_ist.minute / 60
        features['minutes_into_session'] = max(0, (current_time_ist.hour - 9) * 60 + (current_time_ist.minute - 15))
        
        # Trend strength
        if len(close) >= 13:
            features['trend_strength'] = (close[-1] - close[-13]) / close[-13] * 100
        else:
            features['trend_strength'] = 0
        
        return features
    except Exception as e:
        return None

def extract_daily_features(hist_df, macro, sentiment_score):
    """Extract daily/historical features"""
    features = {}
    if not hist_df.empty and len(hist_df) >= 22:
        close = hist_df['Close'].values
        price_slice = close[-21:]
        returns = np.diff(price_slice) / price_slice[:-1]
        features['volatility_20d'] = np.std(returns) * 100
        features['return_1d'] = (close[-1] / close[-2] - 1) * 100
    else:
        features['volatility_20d'] = 0.0
        features['return_1d'] = 0.0
        
    features['vix'] = macro['vix']
    features['news'] = sentiment_score
    features['oil'] = macro['oil']
    features['usdinr'] = macro['usdinr']
    
    return features

# --- 5. ONLINE LEARNING MODEL ---

class IntradayLearningModel:
    """Self-learning model that adapts throughout the trading day"""
    
    def __init__(self):
        self.daily_model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
        self.intraday_model = GradientBoostingRegressor(n_estimators=30, max_depth=3, learning_rate=0.1, random_state=42)
        self.scaler_daily = StandardScaler()
        self.scaler_intraday = StandardScaler()
        self.is_trained = False
        self.intraday_ready = False
        
        # Learning tracking
        self.prediction_history = []  # (timestamp, predicted, actual)
        self.error_history = []
        self.confidence_score = 0.5
        
    def train_daily_model(self, hist_df, macro, sentiment):
        """Train on historical daily data"""
        X, y = [], []
        if len(hist_df) < 60: return False
        
        for i in range(30, len(hist_df)-1):
            window = hist_df.iloc[i-30:i]
            f = {
                'volatility_20d': window['Close'].pct_change().std()*100,
                'return_1d': (window['Close'].iloc[-1]/window['Close'].iloc[-2]-1)*100,
                'vix': macro['vix'],
                'news': sentiment,
                'oil': macro['oil'],
                'usdinr': macro['usdinr']
            }
            target = (hist_df['Close'].iloc[i+1] / hist_df['Close'].iloc[i] - 1) * 100
            
            row = [f['volatility_20d'], f['return_1d'], f['vix'], f['news'], f['oil'], f['usdinr']]
            X.append(row)
            y.append(target)
        
        try:
            X = np.array(X)
            y = np.array(y)
            X_scaled = self.scaler_daily.fit_transform(X)
            self.daily_model.fit(X_scaled, y)
            self.is_trained = True
            return True
        except:
            return False
    
    def train_intraday_model(self, intraday_df_history):
        """Train on intraday patterns from current session"""
        if len(intraday_df_history) < 20:
            return False
        
        X, y = [], []
        ist = pytz.timezone('Asia/Kolkata')
        
        for i in range(10, len(intraday_df_history)-1):
            try:
                window = intraday_df_history.iloc[i-10:i]
                
                # Handle timezone-aware/naive timestamps
                timestamp = window.index[-1]
                if timestamp.tzinfo is None:
                    current_time = ist.localize(timestamp)
                else:
                    current_time = timestamp.astimezone(ist)
                
                feats = extract_intraday_features(window, current_time)
                if feats is None:
                    continue
                
                # Target: next 5-minute return
                target = (intraday_df_history['Close'].iloc[i+1] / intraday_df_history['Close'].iloc[i] - 1) * 100
                
                row = [feats['return_5m'], feats['return_30m'], feats['return_1h'], 
                       feats['volatility_1h'], feats['volume_ratio'], feats['range_position'],
                       feats['time_of_day'], feats['minutes_into_session'], feats['trend_strength']]
                X.append(row)
                y.append(target)
            except Exception as e:
                continue
        
        if len(X) < 10:
            return False
        
        try:
            X = np.array(X)
            y = np.array(y)
            X_scaled = self.scaler_intraday.fit_transform(X)
            self.intraday_model.fit(X_scaled, y)
            self.intraday_ready = True
            return True
        except Exception as e:
            return False
    
    def predict_next_movement(self, daily_features, intraday_features=None):
        """Predict next movement combining both models"""
        if not self.is_trained:
            return None, None
        
        # Daily prediction
        daily_row = [daily_features['volatility_20d'], daily_features['return_1d'], 
                     daily_features['vix'], daily_features['news'],
                     daily_features['oil'], daily_features['usdinr']]
        X_daily = np.array([daily_row])
        X_daily_scaled = self.scaler_daily.transform(X_daily)
        daily_pred = self.daily_model.predict(X_daily_scaled)[0]
        
        # Intraday prediction (if available)
        if self.intraday_ready and intraday_features:
            intraday_row = [intraday_features['return_5m'], intraday_features['return_30m'],
                           intraday_features['return_1h'], intraday_features['volatility_1h'],
                           intraday_features['volume_ratio'], intraday_features['range_position'],
                           intraday_features['time_of_day'], intraday_features['minutes_into_session'],
                           intraday_features['trend_strength']]
            X_intra = np.array([intraday_row])
            X_intra_scaled = self.scaler_intraday.transform(X_intra)
            intraday_pred = self.intraday_model.predict(X_intra_scaled)[0]
            
            # Weighted ensemble: intraday gets more weight during market hours
            weight_intraday = 0.7
            combined_pred = daily_pred * (1 - weight_intraday) + intraday_pred * weight_intraday
            
            return combined_pred, intraday_pred
        
        return daily_pred, None
    
    def update_with_actual(self, predicted, actual):
        """Learn from prediction errors (online learning)"""
        error = abs(predicted - actual)
        self.error_history.append(error)
        self.prediction_history.append((datetime.now(), predicted, actual))
        
        # Update confidence based on recent accuracy
        if len(self.error_history) > 10:
            recent_errors = self.error_history[-10:]
            avg_error = np.mean(recent_errors)
            # Confidence decreases with error
            self.confidence_score = max(0.1, 1.0 - (avg_error / 5.0))  # 5% error = 0 confidence
        
        # Keep only last 100 predictions
        if len(self.prediction_history) > 100:
            self.prediction_history = self.prediction_history[-100:]
            self.error_history = self.error_history[-100:]

def model_monte_carlo(price, vol, days=5):
    dt=1; mu=0.0005; sigma=vol/100
    paths=[]
    for _ in range(500):
        p=price; path=[p]
        for _ in range(days):
            p = p * np.exp((mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*np.random.normal(0,1))
            path.append(p)
        paths.append(path)
    return np.mean(paths, axis=0)

# --- 6. INTRADAY PREDICTION ENGINE ---

def generate_intraday_predictions(current_price, intraday_df, model, daily_feats, ist_now):
    """Generate predictions for rest of trading day"""
    predictions = []
    
    if not model.is_trained:
        return predictions
    
    # Current features
    intraday_feats = extract_intraday_features(intraday_df, ist_now)
    if intraday_feats is None:
        return predictions
    
    # Predict next 30 minutes in 5-min intervals
    current_price_sim = current_price
    for mins_ahead in [5, 10, 15, 30]:
        # Adjust time feature
        future_time = ist_now + timedelta(minutes=mins_ahead)
        intraday_feats['time_of_day'] = future_time.hour + future_time.minute / 60
        intraday_feats['minutes_into_session'] = (future_time.hour - 9) * 60 + (future_time.minute - 15)
        
        pred_return, intra_pred = model.predict_next_movement(daily_feats, intraday_feats)
        
        if pred_return is not None:
            predicted_price = current_price_sim * (1 + pred_return / 100)
            predictions.append({
                'time': future_time.strftime('%H:%M'),
                'price': predicted_price,
                'confidence': model.confidence_score
            })
            current_price_sim = predicted_price  # Chain predictions
    
    return predictions

# --- 7. APP EXECUTION ---

if 'model' not in st.session_state:
    st.session_state['model'] = IntradayLearningModel()
if 'model_ready' not in st.session_state:
    st.session_state['model_ready'] = False
if 'last_intraday_train' not in st.session_state:
    st.session_state['last_intraday_train'] = None
if 'movers_data' not in st.session_state:
    st.session_state['movers_data'] = pd.DataFrame()
if 'last_price_logged' not in st.session_state:
    st.session_state['last_price_logged'] = None

is_open, status_msg, refresh_rate = get_market_status()

st.sidebar.title("🦅 Market Watch")
selected_index = st.sidebar.selectbox("Select Index", list(INDICES.keys()))
ticker = INDICES[selected_index]

# Fetch Data
hist_max, hist_intra, metrics = fetch_main_data(ticker)
if hist_max.empty:
    st.error("⚠️ Data connection failed. Reloading...")
    st.stop()

macro = fetch_macro_indicators()
news_score, headlines = get_sentiment()

# --- TRAIN DAILY MODEL (Once) ---
if not st.session_state['model_ready']:
    with st.spinner("🧠 Initializing AI Brain..."):
        success = st.session_state['model'].train_daily_model(hist_max, macro, news_score)
        st.session_state['model_ready'] = success

# --- TRAIN INTRADAY MODEL (During market hours, every 15 min) ---
ist = pytz.timezone('Asia/Kolkata')
now_ist = datetime.now(ist)

if is_open and not hist_intra.empty and len(hist_intra) >= 20:
    # Retrain every 15 minutes with new data
    try:
        if (st.session_state['last_intraday_train'] is None or 
            (now_ist - st.session_state['last_intraday_train']).total_seconds() > 900):
            
            with st.spinner("📚 Learning from live patterns..."):
                success = st.session_state['model'].train_intraday_model(hist_intra)
                if success:
                    st.session_state['last_intraday_train'] = now_ist
    except Exception as e:
        # Silently continue if training fails
        pass

# --- GENERATE PREDICTIONS ---
daily_feats = extract_daily_features(hist_max, macro, news_score)

# 1. Monte Carlo (5-day forecast)
mc_path = model_monte_carlo(metrics['price'], metrics['volatility'])
mc_target = mc_path[-1]

# 2. ML Ensemble
ml_target = metrics['price']
intraday_predictions = []

if st.session_state['model_ready']:
    intraday_feats = None
    if is_open and not hist_intra.empty:
        intraday_feats = extract_intraday_features(hist_intra, now_ist)
    
    pred_return, intra_pred = st.session_state['model'].predict_next_movement(daily_feats, intraday_feats)
    
    if pred_return:
        ml_target = metrics['price'] * (1 + pred_return/100)
    
    # Generate intraday predictions
    if is_open and not hist_intra.empty and st.session_state['model'].intraday_ready:
        intraday_predictions = generate_intraday_predictions(
            metrics['price'], hist_intra, st.session_state['model'], daily_feats, now_ist
        )

consensus_target = (mc_target + ml_target) / 2

# --- TRACK ACTUAL VS PREDICTED (Online Learning) ---
if is_open and st.session_state['last_price_logged']:
    last_price, last_pred = st.session_state['last_price_logged']
    current_actual_return = (metrics['price'] / last_price - 1) * 100
    st.session_state['model'].update_with_actual(last_pred, current_actual_return)

if is_open:
    st.session_state['last_price_logged'] = (metrics['price'], pred_return if pred_return else 0)

# --- VISUAL DASHBOARD ---

st.title(f"{selected_index} Live Command Center")
st.caption(f"Status: {status_msg} | AI Confidence: {st.session_state['model'].confidence_score:.0%} | Update: {refresh_rate}s")

m = st.columns(5)
m[0].metric("Price", f"₹{metrics['price']:,.2f}", f"{((metrics['price']-metrics['prev'])/metrics['prev'])*100:.2f}%")
m[1].metric("Open", f"₹{metrics['open']:,.2f}", delta_color="off")
m[2].metric("High", f"₹{metrics['high']:,.2f}", delta_color="off")
m[3].metric("Low", f"₹{metrics['low']:,.2f}", delta_color="off")
c_col = "normal" if consensus_target > metrics['price'] else "inverse"
m[4].metric("AI Target (EOD)", f"₹{consensus_target:,.2f}", f"{((consensus_target/metrics['price']-1)*100):.2f}%", delta_color=c_col)

st.markdown("---")

# Intraday Predictions Display
if is_open and intraday_predictions:
    st.markdown('<div class="prediction-update">', unsafe_allow_html=True)
    st.subheader("🔮 Live Intraday Forecast")
    pred_cols = st.columns(len(intraday_predictions))
    for i, pred in enumerate(intraday_predictions):
        with pred_cols[i]:
            change = ((pred['price'] / metrics['price']) - 1) * 100
            pred_cols[i].metric(
                f"@ {pred['time']}", 
                f"₹{pred['price']:,.2f}",
                f"{change:+.2f}%",
                delta_color="normal" if change > 0 else "inverse"
            )
    st.caption(f"📊 Predictions update every {refresh_rate}s | Confidence: {st.session_state['model'].confidence_score:.0%}")
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("---")

g_col, s_col = st.columns([3, 1])

with g_col:
    with st.expander("🧠 AI Model Breakdown", expanded=False):
        c1, c2 = st.columns(2)
        c1.metric("Monte Carlo (5D)", f"₹{mc_target:,.2f}", "Long-term Trend")
        c2.metric("ML Ensemble (Next Move)", f"₹{ml_target:,.2f}", "Real-time Learning")
        
        if st.session_state['model'].prediction_history:
            st.caption(f"📈 Learning History: {len(st.session_state['model'].prediction_history)} predictions tracked")
            if len(st.session_state['model'].error_history) >= 5:
                recent_accuracy = 100 - np.mean(st.session_state['model'].error_history[-5:])
                st.caption(f"🎯 Recent Accuracy: {recent_accuracy:.1f}%")

    st.subheader("📈 Trend Analysis")
    time_range = st.radio("Range", ["1D", "1M", "6M", "YTD", "1Y", "MAX"], horizontal=True, label_visibility="collapsed")
    
    fig = go.Figure()
    
    if time_range == "1D":
        if not hist_intra.empty:
            fig.add_trace(go.Scatter(
                x=hist_intra.index, 
                y=hist_intra['Close'], 
                mode='lines', 
                name='Price', 
                line=dict(color='#00F0FF', width=2)
            ))
            
            # Add intraday predictions
            if intraday_predictions:
                pred_times = [now_ist + timedelta(minutes=5*i) for i in range(1, len(intraday_predictions)+1)]
                pred_prices = [p['price'] for p in intraday_predictions]
                fig.add_trace(go.Scatter(
                    x=pred_times,
                    y=pred_prices,
                    mode='lines+markers',
                    name='AI Forecast',
                    line=dict(color='#FFA500', dash='dot', width=2),
                    marker=dict(size=8)
                ))
        else:
            st.warning("Intraday data hidden (Market Closed)")
    else:
        df_p = hist_max.copy()
        end_d = df_p.index[-1]
        if time_range == "1M": start_d = end_d - timedelta(days=30)
        elif time_range == "6M": start_d = end_d - timedelta(days=180)
        elif time_range == "1Y": start_d = end_d - timedelta(days=365)
        elif time_range == "YTD": start_d = datetime(end_d.year, 1, 1).replace(tzinfo=end_d.tzinfo)
        else: start_d = df_p.index[0]
        
        df_p = df_p[df_p.index >= start_d]
        fig.add_trace(go.Scatter(x=df_p.index, y=df_p['Close'], mode='lines', name='Price', line=dict(color='#00F0FF')))
        
        fdates = [df_p.index[-1] + timedelta(days=i) for i in range(1, 6)]
        fig.add_trace(go.Scatter(x=fdates, y=mc_path[1:], mode='lines+markers', name='5D AI Projection', line=dict(color='#FFA500', dash='dot')))

    fig.update_layout(template="plotly_dark", height=500, xaxis_rangeslider_visible=False, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

with s_col:
    st.subheader("📰 Sentiment")
    st.metric("News Score", f"{news_score:.2f}", "-1 (Bear) to +1 (Bull)")
    st.metric("Market VIX", f"{macro['vix']:.2f}", "Volatility Index")
    
    with st.container(border=True):
        st.write("**Headlines:**")
        for h in headlines[:5]: 
            st.caption(f"• {h}")
    
    # Learning Stats
    if st.session_state['model'].prediction_history:
        st.subheader("🧠 AI Learning Stats")
        with st.container(border=True):
            st.caption(f"📊 Predictions Made: {len(st.session_state['model'].prediction_history)}")
            if len(st.session_state['model'].error_history) >= 3:
                avg_error = np.mean(st.session_state['model'].error_history[-10:])
                st.caption(f"⚡ Avg Error: {avg_error:.2f}%")
            st.caption(f"🎯 Confidence: {st.session_state['model'].confidence_score:.0%}")
            
            if st.session_state['model'].intraday_ready:
                st.caption("✅ Intraday Model: Active")
            else:
                st.caption("⏳ Intraday Model: Building...")

st.markdown("---")

# --- MOVERS SECTION ---
st.subheader("📊 Market Components")

if st.button("🚀 Load Market Movers"):
    st.session_state['movers_data'] = fetch_movers_batch(CONSTITUENTS[selected_index])

if not st.session_state['movers_data'].empty:
    df_m = st.session_state['movers_data']
    t_all, t_gain, t_loss = st.tabs(["📋 Full List", "🟢 Top Gainers", "🔴 Top Losers"])
    cfg = {
        "Price": st.column_config.NumberColumn(format="₹%.2f"),
        "Change %": st.column_config.ProgressColumn("Change %", format="%.2f%%", min_value=-5, max_value=5),
    }
    with t_all: st.dataframe(df_m.sort_values("Company"), column_config=cfg, use_container_width=True, hide_index=True)
    with t_gain: st.dataframe(df_m.sort_values("Change %", ascending=False).head(10), column_config=cfg, use_container_width=True, hide_index=True)
    with t_loss: st.dataframe(df_m.sort_values("Change %", ascending=True).head(10), column_config=cfg, use_container_width=True, hide_index=True)
else:
    st.info("Click the button above to scan constituent stocks.")

# Learning Dashboard (Expandable)
if st.session_state['model'].prediction_history:
    with st.expander("📈 Detailed Learning Analytics", expanded=False):
        if len(st.session_state['model'].prediction_history) >= 5:
            recent = st.session_state['model'].prediction_history[-20:]
            
            pred_df = pd.DataFrame([
                {
                    'Time': p[0].strftime('%H:%M:%S'),
                    'Predicted %': p[1],
                    'Actual %': p[2],
                    'Error': abs(p[1] - p[2])
                }
                for p in recent
            ])
            
            st.dataframe(pred_df, use_container_width=True, hide_index=True)
            
            # Error chart
            error_fig = go.Figure()
            error_fig.add_trace(go.Scatter(
                y=st.session_state['model'].error_history[-20:],
                mode='lines+markers',
                name='Prediction Error',
                line=dict(color='#FF4B4B')
            ))
            error_fig.update_layout(
                title="Prediction Error Over Time",
                template="plotly_dark",
                height=250,
                margin=dict(l=0, r=0, t=30, b=0)
            )
            st.plotly_chart(error_fig, use_container_width=True)

# Auto-refresh
time_module.sleep(refresh_rate)
st.rerun()
