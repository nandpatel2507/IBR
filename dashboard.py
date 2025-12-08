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

# --- 1. PRO UI CONFIGURATION ---
st.set_page_config(
    page_title="Pro Market Terminal",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for "Glassmorphism" Look
st.markdown("""
    <style>
    .stApp { background-color: #0E1117; }
    
    /* Cards/Containers */
    div[data-testid="stMetric"] {
        background-color: #1A1C24;
        border: 1px solid #333;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        transition: transform 0.2s;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        border-color: #00F0FF;
    }
    
    /* Chart Container */
    .stPlotlyChart {
        background-color: #1A1C24;
        border: 1px solid #333;
        border-radius: 12px;
        padding: 15px;
    }
    
    /* Tab Styling */
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

# Download VADER Lexicon
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

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

# --- 3. CORE LOGIC (MODELS & DATA) ---

def get_market_status():
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    if now.weekday() < 5 and (time(9,15) <= now.time() <= time(15,30)):
        return True, "🟢 LIVE", 60 
    return False, "🔴 CLOSED", 300

@st.cache_data(ttl=600)
def fetch_main_data(ticker):
    """Fetches Chart Data (Cached 10 mins)"""
    stock = yf.Ticker(ticker)
    try:
        hist_max = stock.history(period="max", interval="1d")
        hist_intra = stock.history(period="1d", interval="5m")
        if hist_max.empty: return pd.DataFrame(), pd.DataFrame(), {}
        
        # Calculate Volatility
        volatility = hist_max['Close'].pct_change().std() * 100
        
        metrics = {
            "price": float(hist_max['Close'].iloc[-1]),
            "prev": float(hist_max['Close'].iloc[-2]),
            "open": float(hist_max['Open'].iloc[-1]),
            "high": float(hist_max['High'].iloc[-1]),
            "low": float(hist_max['Low'].iloc[-1]),
            "volatility": volatility
        }
        return hist_max, hist_intra, metrics
    except: return pd.DataFrame(), pd.DataFrame(), {}

@st.cache_data(ttl=3600)
def fetch_movers_batch(const_tickers):
    """Fetches Movers in Chunks to avoid Bans"""
    chunk_size = 10
    chunks = [const_tickers[i:i + chunk_size] for i in range(0, len(const_tickers), chunk_size)]
    all_data = []
    
    prog = st.progress(0, "Scanning Market Components...")
    
    for i, chunk in enumerate(chunks):
        try:
            # Threading off for stability
            data = yf.download(chunk, period="2d", group_by='ticker', progress=False, threads=False)
            for t in chunk:
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
                except: continue
            time_module.sleep(0.5) # Be nice to API
            prog.progress((i + 1) / len(chunks))
        except: continue
            
    prog.empty()
    df = pd.DataFrame(all_data)
    
    # NUCLEAR DUPLICATE KILLER
    if not df.empty:
        df = df.drop_duplicates(subset=['Company'])
        
    return df

def get_sentiment():
    sia = SentimentIntensityAnalyzer()
    articles = []
    for feed in NEWS_FEEDS:
        try:
            parsed = feedparser.parse(feed)
            for entry in parsed.entries[:5]: articles.append(entry.title) # Top 5 headlines
        except: continue
    news_score = np.mean([sia.polarity_scores(a)['compound'] for a in articles]) if articles else 0
    return news_score, articles

# --- PREDICTION MODELS ---

def model_monte_carlo(price, vol, days=5):
    """Model 1: Monte Carlo Simulation (Statistical)"""
    dt = 1
    mu = 0.0005 # Assumed daily drift (long term positive)
    sigma = vol / 100 
    
    paths = []
    for _ in range(500): # 500 Simulations
        p = price
        path = [p]
        for _ in range(days):
            shock = np.random.normal(0, 1)
            p = p * np.exp((mu - 0.5 * sigma**2)*dt + sigma * np.sqrt(dt) * shock)
            path.append(p)
        paths.append(path)
    
    mean_path = np.mean(paths, axis=0)
    return mean_path, mean_path[-1]

def model_technical(price, change_pct, news_score):
    """Model 2: Technical + Sentiment Voting (Momentum)"""
    # Simple Momentum Logic
    tech_drift = change_pct * 0.5 # Momentum continues half the time
    sent_drift = news_score * 0.01 # News impact
    
    total_drift = tech_drift + sent_drift
    target = price * (1 + total_drift)
    return target

# --- 4. MAIN APP EXECUTION ---

if 'last_run' not in st.session_state: st.session_state['last_run'] = 0
is_open, status_msg, refresh_rate = get_market_status()

# Sidebar
st.sidebar.title("🦅 Market Watch")
selected_index = st.sidebar.selectbox("Select Index", list(INDICES.keys()))
ticker = INDICES[selected_index]

# Main Fetch
hist_max, hist_intra, metrics = fetch_main_data(ticker)

if hist_max.empty:
    st.error("⚠️ Data connection issue. Please refresh.")
    st.stop()

# Run Models
news_score, headlines = get_sentiment()
mc_path, mc_target = model_monte_carlo(metrics['price'], metrics['volatility'])
tech_target = model_technical(metrics['price'], (metrics['price']-metrics['prev'])/metrics['prev'], news_score)
consensus_target = (mc_target + tech_target) / 2

# --- 5. VISUAL DASHBOARD ---

st.title(f"{selected_index} Command Center")
st.caption(f"Status: {status_msg} | Updates: {refresh_rate}s")

# Heads Up Metrics
m = st.columns(5)
m[0].metric("Price", f"₹{metrics['price']:,.2f}", f"{((metrics['price']-metrics['prev'])/metrics['prev'])*100:.2f}%")
m[1].metric("Open", f"₹{metrics['open']:,.2f}", delta_color="off")
m[2].metric("High", f"₹{metrics['high']:,.2f}", delta_color="off")
m[3].metric("Low", f"₹{metrics['low']:,.2f}", delta_color="off")
# Consensus Metric
c_col = "normal" if consensus_target > metrics['price'] else "inverse"
m[4].metric("AI Consensus", f"₹{consensus_target:,.2f}", "Avg of 2 Models", delta_color=c_col)

st.markdown("---")

# PREDICTION & GRAPH AREA
g_col, s_col = st.columns([3, 1])

with g_col:
    # 1. Prediction Breakdown
    with st.expander("🧠 View Prediction Models (Monte Carlo vs Technical)", expanded=False):
        c1, c2 = st.columns(2)
        c1.metric("Monte Carlo Model", f"₹{mc_target:,.2f}", "Statistical Sim")
        c2.metric("Technical Model", f"₹{tech_target:,.2f}", "Momentum + News")
    
    # 2. Main Chart
    st.subheader("📈 Market Trends")
    time_range = st.radio("Range", ["1D", "1M", "6M", "YTD", "1Y", "MAX"], horizontal=True, label_visibility="collapsed")
    
    fig = go.Figure()
    
    if time_range == "1D":
        if not hist_intra.empty:
            fig.add_trace(go.Scatter(x=hist_intra.index, y=hist_intra['Close'], mode='lines', name='Price', line=dict(color='#00F0FF', width=2)))
            # Intraday Forecast Line
            last_t = hist_intra.index[-1]
            fig.add_trace(go.Scatter(x=[last_t, last_t+timedelta(hours=1)], y=[metrics['price'], consensus_target], mode='lines+markers', name='Forecast', line=dict(color='#FFA500', dash='dot')))
        else: st.warning("Intraday data hidden (Market Closed)")
    else:
        # Historical Filter
        df_p = hist_max.copy()
        end_d = df_p.index[-1]
        if time_range == "1M": start_d = end_d - timedelta(days=30)
        elif time_range == "6M": start_d = end_d - timedelta(days=180)
        elif time_range == "1Y": start_d = end_d - timedelta(days=365)
        elif time_range == "YTD": start_d = datetime(end_d.year, 1, 1).replace(tzinfo=end_d.tzinfo)
        else: start_d = df_p.index[0]
        
        df_p = df_p[df_p.index >= start_d]
        fig.add_trace(go.Scatter(x=df_p.index, y=df_p['Close'], mode='lines', name='Price', line=dict(color='#00F0FF')))
        
        # 5-Day Forecast Line
        fdates = [df_p.index[-1] + timedelta(days=i) for i in range(1, 6)]
        fig.add_trace(go.Scatter(x=fdates, y=mc_path[1:], mode='lines+markers', name='AI Projection', line=dict(color='#FFA500', dash='dot')))

    fig.update_layout(template="plotly_dark", height=500, xaxis_rangeslider_visible=False, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

with s_col:
    # 3. Restored News Section
    st.subheader("📰 Sentiment")
    st.metric("News Score", f"{news_score:.2f}", "-1 (Bear) to +1 (Bull)")
    
    with st.container(border=True):
        st.write("**Top Headlines:**")
        for h in headlines:
            st.caption(f"• {h}")

st.markdown("---")

# --- MOVERS SECTION (Using Session State to fix Duplication) ---
st.subheader("📊 Market Components")

# Initialize Session State for Movers
if 'movers_data' not in st.session_state:
    st.session_state['movers_data'] = pd.DataFrame()

# Load Button
if st.button("🚀 Load Market Movers (Click Once)"):
    st.session_state['movers_data'] = fetch_movers_batch(CONSTITUENTS[selected_index])

# Display Table from Session State (Persistent)
if not st.session_state['movers_data'].empty:
    df_m = st.session_state['movers_data']
    
    t_all, t_gain, t_loss = st.tabs(["📋 Full List", "🟢 Top Gainers", "🔴 Top Losers"])
    
    # Beautiful Column Config
    cfg = {
        "Price": st.column_config.NumberColumn(format="₹%.2f"),
        "Change %": st.column_config.ProgressColumn("Change %", format="%.2f%%", min_value=-5, max_value=5),
    }
    
    with t_all: st.dataframe(df_m.sort_values("Company"), column_config=cfg, use_container_width=True, hide_index=True)
    with t_gain: st.dataframe(df_m.sort_values("Change %", ascending=False).head(10), column_config=cfg, use_container_width=True, hide_index=True)
    with t_loss: st.dataframe(df_m.sort_values("Change %", ascending=True).head(10), column_config=cfg, use_container_width=True, hide_index=True)
else:
    st.info("Click the button above to scan constituent stocks.")

# Refresh
time_module.sleep(refresh_rate)
st.rerun()
