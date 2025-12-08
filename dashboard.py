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

# Suppress warnings
warnings.filterwarnings('ignore')

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="Pro Market Terminal",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
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
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #1A1C24;
        border-radius: 8px;
        padding: 10px 20px;
        color: white;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FF4B4B !important;
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

# Download VADER
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

# --- 3. ROBUST DATA ENGINE ---

def get_market_status():
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    if now.weekday() < 5 and (time(9,15) <= now.time() <= time(15,30)):
        return True, "🟢 LIVE", 60 
    return False, "🔴 CLOSED", 300

@st.cache_data(ttl=600)
def fetch_main_data(ticker):
    stock = yf.Ticker(ticker)
    try:
        hist_max = stock.history(period="2y", interval="1d")
        hist_intra = stock.history(period="1d", interval="5m")
        if hist_max.empty: return pd.DataFrame(), pd.DataFrame(), {}
        
        # Safe Metric Extraction
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
    # Batch fetch with strict deduplication
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

# --- 4. ADVANCED ML ENGINE ---

def calculate_technical_indicators(df):
    if len(df) < 50: return {}
    close = df['Close'].values
    return {
        'sma_20': np.mean(close[-20:]),
        'rsi': 50.0 # Simplified for speed/safety
    }

def extract_features(hist_df, macro, sentiment_score, tech_ind):
    features = {}
    if not hist_df.empty and len(hist_df) >= 22:
        close = hist_df['Close'].values
        # --- SAFE VOLATILITY CALCULATION ---
        # Take last 21 prices -> calculate 20 returns
        price_slice = close[-21:]
        returns = np.diff(price_slice) / price_slice[:-1]
        features['volatility_20d'] = np.std(returns) * 100
        
        features['return_1d'] = (close[-1] / close[-2] - 1) * 100
    else:
        features['volatility_20d'] = 0.0
        features['return_1d'] = 0.0
        
    features['vix'] = macro['vix']
    features['news'] = sentiment_score
    return features

class AdaptiveEnsemble:
    def __init__(self):
        self.models = {
            'rf': RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42),
            'gb': GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)
        }
        self.scaler = StandardScaler()
        self.is_trained = False
        self.weights = {'rf': 0.5, 'gb': 0.5}

    def train(self, X, y):
        try:
            if len(X) < 20: return False
            X_scaled = self.scaler.fit_transform(X)
            for name, model in self.models.items():
                model.fit(X_scaled, y)
            self.is_trained = True
            return True
        except: return False

    def predict(self, features_dict):
        if not self.is_trained: return None
        try:
            # Create feature array in consistent order
            row = [
                features_dict.get('volatility_20d', 0),
                features_dict.get('return_1d', 0),
                features_dict.get('vix', 15),
                features_dict.get('news', 0)
            ]
            X = np.array([row])
            X_scaled = self.scaler.transform(X)
            
            preds = {name: m.predict(X_scaled)[0] for name, m in self.models.items()}
            final_pred = sum(preds[k] * self.weights[k] for k in self.models)
            return final_pred
        except: return None

    def prepare_data(self, hist_df, macro, sentiment):
        X, y = [], []
        if len(hist_df) < 60: return None, None
        
        # Create training set from history
        for i in range(30, len(hist_df)-5):
            window = hist_df.iloc[i-30:i]
            # Mock features for training history
            f = {
                'volatility_20d': window['Close'].pct_change().std()*100,
                'return_1d': (window['Close'].iloc[-1]/window['Close'].iloc[-2]-1)*100,
                'vix': macro['vix'], # Assuming constant for hist simplicity
                'news': sentiment
            }
            target = (hist_df['Close'].iloc[i+1] / hist_df['Close'].iloc[i] - 1) * 100
            
            row = [f['volatility_20d'], f['return_1d'], f['vix'], f['news']]
            X.append(row)
            y.append(target)
            
        return np.array(X), np.array(y)

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

# --- 5. APP EXECUTION ---

if 'last_run' not in st.session_state: st.session_state['last_run'] = 0
if 'model' not in st.session_state: st.session_state['model'] = AdaptiveEnsemble()
if 'model_ready' not in st.session_state: st.session_state['model_ready'] = False

is_open, status_msg, refresh_rate = get_market_status()

# Sidebar
st.sidebar.title("🦅 Market Watch")
selected_index = st.sidebar.selectbox("Select Index", list(INDICES.keys()))
ticker = INDICES[selected_index]

# Fetch Data
hist_max, hist_intra, metrics = fetch_main_data(ticker)
if hist_max.empty:
    st.error("⚠️ Data connection failed. Reloading..."); st.stop()

macro = fetch_macro_indicators()
news_score, headlines = get_sentiment()

# --- TRAIN ML MODEL (Once) ---
if not st.session_state['model_ready']:
    with st.spinner("🧠 Initializing AI Brain..."):
        X, y = st.session_state['model'].prepare_data(hist_max, macro, news_score)
        if X is not None:
            success = st.session_state['model'].train(X, y)
            st.session_state['model_ready'] = success

# --- GENERATE PREDICTIONS ---
# 1. Monte Carlo
mc_path = model_monte_carlo(metrics['price'], metrics['volatility'])
mc_target = mc_path[-1]

# 2. ML Ensemble
ml_target = metrics['price'] # Default
if st.session_state['model_ready']:
    # Current features
    curr_feats = extract_features(hist_max, macro, news_score, {})
    pred_return = st.session_state['model'].predict(curr_feats)
    if pred_return:
        ml_target = metrics['price'] * (1 + pred_return/100)

consensus_target = (mc_target + ml_target) / 2

# --- VISUAL DASHBOARD ---

st.title(f"{selected_index} Command Center")
st.caption(f"Status: {status_msg} | Update Rate: {refresh_rate}s")

m = st.columns(5)
m[0].metric("Price", f"₹{metrics['price']:,.2f}", f"{((metrics['price']-metrics['prev'])/metrics['prev'])*100:.2f}%")
m[1].metric("Open", f"₹{metrics['open']:,.2f}", delta_color="off")
m[2].metric("High", f"₹{metrics['high']:,.2f}", delta_color="off")
m[3].metric("Low", f"₹{metrics['low']:,.2f}", delta_color="off")
c_col = "normal" if consensus_target > metrics['price'] else "inverse"
m[4].metric("AI Target (5D)", f"₹{consensus_target:,.2f}", f"{((consensus_target/metrics['price']-1)*100):.2f}%", delta_color=c_col)

st.markdown("---")

g_col, s_col = st.columns([3, 1])

with g_col:
    with st.expander("🧠 AI Model Breakdown", expanded=True):
        c1, c2 = st.columns(2)
        c1.metric("Monte Carlo (Statistical)", f"₹{mc_target:,.2f}", "Trend Projection")
        c2.metric("ML Ensemble (Learning)", f"₹{ml_target:,.2f}", "Pattern Recognition")

    st.subheader("📈 Trend Analysis")
    time_range = st.radio("Range", ["1D", "1M", "6M", "YTD", "1Y", "MAX"], horizontal=True, label_visibility="collapsed")
    
    fig = go.Figure()
    
    if time_range == "1D":
        if not hist_intra.empty:
            fig.add_trace(go.Scatter(x=hist_intra.index, y=hist_intra['Close'], mode='lines', name='Price', line=dict(color='#00F0FF', width=2)))
            last_t = hist_intra.index[-1]
            fig.add_trace(go.Scatter(x=[last_t, last_t+timedelta(hours=1)], y=[metrics['price'], consensus_target], mode='lines+markers', name='Forecast', line=dict(color='#FFA500', dash='dot')))
        else: st.warning("Intraday hidden (Market Closed)")
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
        fig.add_trace(go.Scatter(x=fdates, y=mc_path[1:], mode='lines+markers', name='AI Projection', line=dict(color='#FFA500', dash='dot')))

    fig.update_layout(template="plotly_dark", height=500, xaxis_rangeslider_visible=False, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

with s_col:
    st.subheader("📰 Sentiment")
    st.metric("News Score", f"{news_score:.2f}", "-1 (Bear) to +1 (Bull)")
    st.metric("Market VIX", f"{macro['vix']:.2f}", "Volatility Index")
    
    with st.container(border=True):
        st.write("**Headlines:**")
        for h in headlines: st.caption(f"• {h}")

st.markdown("---")

# --- MOVERS (SESSION STATE FIX) ---
st.subheader("📊 Market Components")

if 'movers_data' not in st.session_state:
    st.session_state['movers_data'] = pd.DataFrame()

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

time_module.sleep(refresh_rate)
st.rerun()
