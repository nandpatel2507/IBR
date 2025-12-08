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
warnings.filterwarnings('ignore')

# === CONFIG ===
st.set_page_config(page_title="AI Market Terminal", page_icon="🦅", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""<style>
.stApp { background-color: #0E1117; }
div[data-testid="stMetric"] {
    background-color: #1A1C24; border: 1px solid #333; padding: 15px;
    border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); transition: transform 0.2s;
}
div[data-testid="stMetric"]:hover { transform: translateY(-2px); border-color: #00F0FF; }
.stPlotlyChart { background-color: #1A1C24; border: 1px solid #333; border-radius: 12px; padding: 15px; }
</style>""", unsafe_allow_html=True)

try: nltk.data.find('vader_lexicon')
except: nltk.download('vader_lexicon', quiet=True)

# === DATA CONSTANTS ===
INDICES = {"NIFTY 50": "^NSEI", "NIFTY BANK": "^NSEBANK", "NIFTY IT": "^CNXIT", "SENSEX": "^BSESN"}
NEWS_FEEDS = ["https://www.livemint.com/rss/markets", "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms"]

# === CORE FUNCTIONS ===
def get_market_status():
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    if now.weekday() < 5 and time(9,15) <= now.time() <= time(15,30):
        return True, "🟢 LIVE", 60
    return False, "🔴 CLOSED", 300

@st.cache_data(ttl=600)
def fetch_main_data(ticker):
    stock = yf.Ticker(ticker)
    try:
        hist_max = stock.history(period="max", interval="1d")
        hist_intra = stock.history(period="1d", interval="5m")
        if hist_max.empty: return pd.DataFrame(), pd.DataFrame(), {}
        
        metrics = {
            "price": float(hist_max['Close'].iloc[-1]), "prev": float(hist_max['Close'].iloc[-2]),
            "open": float(hist_max['Open'].iloc[-1]), "high": float(hist_max['High'].iloc[-1]),
            "low": float(hist_max['Low'].iloc[-1]), "volatility": hist_max['Close'].pct_change().std() * 100
        }
        return hist_max, hist_intra, metrics
    except: return pd.DataFrame(), pd.DataFrame(), {}

@st.cache_data(ttl=3600)
def fetch_macro_indicators():
    indicators = {}
    try:
        vix = yf.Ticker("^INDIAVIX").history(period="5d")
        indicators['vix'] = float(vix['Close'].iloc[-1]) if not vix.empty else 15.0
        
        usdinr = yf.Ticker("INR=X").history(period="5d")
        indicators['usdinr'] = float(usdinr['Close'].iloc[-1]) if not usdinr.empty else 83.0
        indicators['usdinr_change'] = float(usdinr['Close'].pct_change().iloc[-1]) if len(usdinr) > 1 else 0.0
        
        oil = yf.Ticker("CL=F").history(period="5d")
        indicators['oil_price'] = float(oil['Close'].iloc[-1]) if not oil.empty else 80.0
        indicators['oil_change'] = float(oil['Close'].pct_change().iloc[-1]) if len(oil) > 1 else 0.0
        
        gold = yf.Ticker("GC=F").history(period="5d")
        indicators['gold_price'] = float(gold['Close'].iloc[-1]) if not gold.empty else 2000.0
        
        sp500 = yf.Ticker("^GSPC").history(period="5d")
        indicators['sp500_change'] = float(sp500['Close'].pct_change().iloc[-1]) if len(sp500) > 1 else 0.0
        
        us10y = yf.Ticker("^TNX").history(period="5d")
        indicators['us10y'] = float(us10y['Close'].iloc[-1]) if not us10y.empty else 4.5
    except:
        indicators = {'vix': 15.0, 'usdinr': 83.0, 'usdinr_change': 0.0, 'oil_price': 80.0, 
                      'oil_change': 0.0, 'gold_price': 2000.0, 'sp500_change': 0.0, 'us10y': 4.5}
    return indicators

@st.cache_data(ttl=1800)
def get_sentiment():
    sia = SentimentIntensityAnalyzer()
    articles, sentiments = [], []
    for feed in NEWS_FEEDS:
        try:
            parsed = feedparser.parse(feed)
            for entry in parsed.entries[:10]:
                articles.append(entry.title)
                sentiments.append(sia.polarity_scores(entry.title)['compound'])
        except: continue
    news_score = np.mean(sentiments) if sentiments else 0.0
    news_volatility = np.std(sentiments) if len(sentiments) > 1 else 0.0
    return news_score, news_volatility, articles[:10]

def calculate_technical_indicators(df):
    if df.empty or len(df) < 50: return {}
    close, volume = df['Close'].values, df['Volume'].values
    
    indicators = {
        'sma_20': np.mean(close[-20:]), 'sma_50': np.mean(close[-50:]) if len(close) >= 50 else np.mean(close),
        'ema_12': close[-1], 'ema_26': np.mean(close[-26:]) if len(close) >= 26 else np.mean(close)
    }
    
    # RSI
    deltas = np.diff(close[-15:])
    gains, losses = np.where(deltas > 0, deltas, 0), np.where(deltas < 0, -deltas, 0)
    avg_gain, avg_loss = np.mean(gains) if len(gains) > 0 else 0.0001, np.mean(losses) if len(losses) > 0 else 0.0001
    indicators['rsi'] = 100 - (100 / (1 + avg_gain / avg_loss))
    
    indicators['macd'] = indicators['ema_12'] - indicators['ema_26']
    
    # Bollinger Bands
    std_20 = np.std(close[-20:])
    indicators['bb_upper'], indicators['bb_lower'] = indicators['sma_20'] + 2*std_20, indicators['sma_20'] - 2*std_20
    # Safe BB Position calculation
    denom = indicators['bb_upper'] - indicators['bb_lower']
    indicators['bb_position'] = (close[-1] - indicators['bb_lower']) / denom if denom != 0 else 0
    
    # Volume Ratio with safety check
    vol_mean = np.mean(volume[-20:])
    indicators['volume_ratio'] = volume[-1] / vol_mean if vol_mean > 0 else 1.0
    
    indicators['momentum_5'] = (close[-1] / close[-5] - 1) * 100 if len(close) >= 5 else 0
    indicators['momentum_10'] = (close[-1] / close[-10] - 1) * 100 if len(close) >= 10 else 0
    
    return indicators

def extract_features(hist_df, macro, sentiment_data, tech_ind):
    features = {}
    if not hist_df.empty and len(hist_df) >= 21: # Requires at least 21 days for volatility
        close = hist_df['Close'].values
        features['return_1d'] = (close[-1] / close[-2] - 1) * 100
        features['return_5d'] = (close[-1] / close[-5] - 1) * 100
        features['return_20d'] = (close[-1] / close[-20] - 1) * 100
        
        # --- FIXED VOLATILITY CALCULATION ---
        # We define a slice of 21 prices to get 20 daily returns
        price_slice = close[-21:]
        returns = np.diff(price_slice) / price_slice[:-1]
        features['volatility_20d'] = np.std(returns) * 100
    else:
        # Fallback for insufficient data
        features['return_1d'] = 0.0
        features['return_5d'] = 0.0
        features['return_20d'] = 0.0
        features['volatility_20d'] = 0.0
    
    features.update(macro)
    features['news_sentiment'], features['news_volatility'] = sentiment_data[0], sentiment_data[1]
    features.update(tech_ind)
    features['day_of_week'] = datetime.now().weekday()
    features['is_month_end'] = 1 if datetime.now().day >= 25 else 0
    return features

# === MACHINE LEARNING MODEL ===
class AdaptiveEnsemble:
    def __init__(self):
        self.models = {
            'rf': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
            'gb': GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
        }
        self.scaler = StandardScaler()
        self.is_trained, self.feature_names = False, None
        self.weights = {'rf': 0.5, 'gb': 0.5}
    
    def prepare_training_data(self, hist_df, macro, sentiment_score, sentiment_vol):
        X, y = [], []
        if len(hist_df) < 60: return None, None
        
        for i in range(50, len(hist_df) - 5):
            window = hist_df.iloc[i-50:i]
            tech_ind = calculate_technical_indicators(window)
            features = extract_features(window, macro, (sentiment_score, sentiment_vol, []), tech_ind)
            
            future_price, current_price = hist_df['Close'].iloc[i+5], hist_df['Close'].iloc[i]
            target = (future_price / current_price - 1) * 100
            
            # Ensure features are a flat list of numbers
            row = []
            for k, v in features.items():
                if isinstance(v, (int, float, np.number)): row.append(v)
                else: row.append(0.0) # Safety for non-numeric
            
            X.append(row)
            y.append(target)
            if i == 50: self.feature_names = list(features.keys())
        
        return np.array(X), np.array(y)
    
    def train(self, X, y):
        if X is None or len(X) < 20: return False
        split = int(0.8 * len(X))
        X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        for name, model in self.models.items():
            model.fit(X_train_scaled, y_train)
            score = model.score(X_test_scaled, y_test)
            self.weights[name] = max(0.1, score)
        
        total = sum(self.weights.values())
        self.weights = {k: v/total for k, v in self.weights.items()}
        self.is_trained = True
        return True
    
    def predict(self, features):
        if not self.is_trained: return None
        # Convert dictionary to ordered list of values matching training
        row = []
        for k, v in features.items():
            if isinstance(v, (int, float, np.number)): row.append(v)
            else: row.append(0.0)
            
        X = np.array([row])
        X_scaled = self.scaler.transform(X)
        
        predictions = {name: model.predict(X_scaled)[0] for name, model in self.models.items()}
        ensemble_pred = sum(predictions[name] * self.weights[name] for name in self.models.keys())
        
        return ensemble_pred, predictions, self.weights

def model_monte_carlo(price, vol, days=5):
    dt, mu, sigma = 1, 0.0005, vol / 100
    if vol > 20: mu = 0.0
    
    paths = []
    for _ in range(1000):
        p, path = price, [price]
        for _ in range(days):
            shock = np.random.normal(0, 1)
            p = p * np.exp((mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*shock)
            path.append(p)
        paths.append(path)
    
    mean_path = np.mean(paths, axis=0)
    confidence_95 = np.percentile(paths, [2.5, 97.5], axis=0)
    return mean_path, mean_path[-1], confidence_95

# === MAIN APP ===
if 'ensemble_model' not in st.session_state:
    st.session_state['ensemble_model'] = AdaptiveEnsemble()
if 'model_trained' not in st.session_state:
    st.session_state['model_trained'] = False
if 'current_index' not in st.session_state:
    st.session_state['current_index'] = None

is_open, status_msg, refresh_rate = get_market_status()

st.sidebar.title("🦅 Market Watch")
selected_index = st.sidebar.selectbox("Select Index", list(INDICES.keys()))
ticker = INDICES[selected_index]

if st.session_state['current_index'] != selected_index:
    st.session_state['current_index'] = selected_index
    st.session_state['model_trained'] = False

hist_max, hist_intra, metrics = fetch_main_data(ticker)
if hist_max.empty:
    st.error("⚠️ Data connection issue. Please refresh.")
    st.stop()

macro = fetch_macro_indicators()
news_score, news_vol, headlines = get_sentiment()
tech_ind = calculate_technical_indicators(hist_max)
features = extract_features(hist_max, macro, (news_score, news_vol, headlines), tech_ind)

# Train model
if not st.session_state['model_trained'] and len(hist_max) >= 60:
    with st.spinner("🧠 Training AI Models..."):
        X, y = st.session_state['ensemble_model'].prepare_training_data(hist_max, macro, news_score, news_vol)
        if X is not None:
            st.session_state['model_trained'] = st.session_state['ensemble_model'].train(X, y)

# Predictions
mc_path, mc_target, mc_conf = model_monte_carlo(metrics['price'], metrics['volatility'])
ml_prediction, ml_models_pred, ml_weights = None, None, None

if st.session_state['model_trained']:
    ml_result = st.session_state['ensemble_model'].predict(features)
    if ml_result:
        ml_pred_pct, ml_models_pred, ml_weights = ml_result
        ml_prediction = metrics['price'] * (1 + ml_pred_pct / 100)

consensus_target = (mc_target + ml_prediction) / 2 if ml_prediction else mc_target

# === DASHBOARD ===
st.title(f"{selected_index} AI Prediction Terminal")
st.caption(f"Status: {status_msg} | AI: {'✅ Active' if st.session_state['model_trained'] else '⚙️ Training'}")

m = st.columns(5)
m[0].metric("Price", f"₹{metrics['price']:,.2f}", f"{((metrics['price']-metrics['prev'])/metrics['prev'])*100:.2f}%")
m[1].metric("VIX", f"{macro['vix']:.1f}", "Fear Index")
m[2].metric("USD/INR", f"₹{macro['usdinr']:.2f}", f"{macro['usdinr_change']*100:.2f}%")
m[3].metric("Oil", f"${macro['oil_price']:.1f}", f"{macro['oil_change']*100:.2f}%")
c_col = "normal" if consensus_target > metrics['price'] else "inverse"
m[4].metric("AI Target", f"₹{consensus_target:,.2f}", f"{((consensus_target/metrics['price']-1)*100):.2f}%", delta_color=c_col)

st.markdown("---")

g_col, s_col = st.columns([3, 1])

with g_col:
    with st.expander("🧠 AI Model Predictions", expanded=True):
        pred_cols = st.columns(3)
        pred_cols[0].metric("Monte Carlo", f"₹{mc_target:,.2f}", f"{((mc_target/metrics['price']-1)*100):.2f}%")
        
        if ml_prediction:
            pred_cols[1].metric("ML Ensemble", f"₹{ml_prediction:,.2f}", f"{((ml_prediction/metrics['price']-1)*100):.2f}%")
            pred_cols[2].metric("Consensus", f"₹{consensus_target:,.2f}", f"{((consensus_target/metrics['price']-1)*100):.2f}%")
            if ml_weights:
                st.caption(f"**Weights:** RF: {ml_weights['rf']:.1%} | GB: {ml_weights['gb']:.1%}")
        else:
            pred_cols[1].info("ML training... Using MC")
    
    st.subheader("📈 Price Chart")
    time_range = st.radio("Range", ["1D", "1M", "6M", "YTD", "1Y", "MAX"], horizontal=True, label_visibility="collapsed")
    
    fig = go.Figure()
    
    if time_range == "1D":
        if not hist_intra.empty:
            fig.add_trace(go.Scatter(x=hist_intra.index, y=hist_intra['Close'], mode='lines', name='Price', line=dict(color='#00F0FF', width=2)))
            last_t = hist_intra.index[-1]
            fig.add_trace(go.Scatter(x=[last_t, last_t+timedelta(hours=1)], y=[metrics['price'], consensus_target], mode='lines+markers', name='Forecast', line=dict(color='#FFA500', dash='dot')))
        else:
            st.warning("Intraday hidden (Market Closed)")
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
        fig.add_trace(go.Scatter(x=fdates, y=mc_path[1:], mode='lines+markers', name='AI Forecast', line=dict(color='#FFA500', dash='dot')))
        
        # Confidence bands
        fig.add_trace(go.Scatter(x=fdates, y=mc_conf[1][1:], fill=None, mode='lines', line=dict(color='rgba(255,165,0,0.2)'), showlegend=False))
        fig.add_trace(go.Scatter(x=fdates, y=mc_conf[0][1:], fill='tonexty', mode='lines', line=dict(color='rgba(255,165,0,0.2)'), name='95% CI'))
    
    fig.update_layout(template="plotly_dark", height=500, xaxis_rangeslider_visible=False, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

with s_col:
    st.subheader("📰 Sentiment")
    st.metric("News Score", f"{news_score:.2f}", "-1 (Bear) to +1 (Bull)")
    
    with st.container(border=True):
        st.write("**Headlines:**")
        for h in headlines[:5]:
            st.caption(f"• {h}")
    
    st.subheader("📊 Tech Indicators")
    with st.container(border=True):
        st.caption(f"RSI: {tech_ind.get('rsi', 0):.1f}")
        st.caption(f"MACD: {tech_ind.get('macd', 0):.2f}")
        st.caption(f"BB Position: {tech_ind.get('bb_position', 0):.2%}")

if is_open:
    time_module.sleep(refresh_rate)
    st.rerun()
