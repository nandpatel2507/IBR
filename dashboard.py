import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import feedparser
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from datetime import datetime, timedelta, time
import time as time_module # Rename to avoid conflict with datetime.time
import pytz

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="AI Market Command Center",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="collapsed" # Collapsed for cleaner look
)

# Custom CSS for "Sleep Mode" and Metrics
st.markdown("""
    <style>
    .metric-container {
        background-color: #1e1e1e;
        padding: 10px;
        border-radius: 8px;
        border: 1px solid #333;
    }
    .market-status-open {
        color: #00FF00;
        font-weight: bold;
        border: 1px solid #00FF00;
        padding: 5px 10px;
        border-radius: 5px;
    }
    .market-status-closed {
        color: #FF4B4B;
        font-weight: bold;
        border: 1px solid #FF4B4B;
        padding: 5px 10px;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Download VADER lexicon (Run once)
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
    "NIFTY 50": ["RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFOSYS.NS", "ITC.NS", "TCS.NS", "L&T.NS"],
    "NIFTY BANK": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "AXISBANK.NS", "KOTAKBANK.NS", "INDUSINDBK.NS"],
    "NIFTY IT": ["TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS", "LTIM.NS"],
    "SENSEX": ["RELIANCE.BO", "HDFCBANK.BO", "ICICIBANK.BO", "INFY.BO", "ITC.BO", "TCS.BO"]
}

NEWS_FEEDS = [
    "https://www.livemint.com/rss/markets",
    "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms"
]

# --- 3. MARKET STATUS LOGIC (IST) ---
def get_market_status():
    """
    Checks if Indian Market is Open (09:15 - 15:30 IST, Mon-Fri).
    Returns: (is_open (bool), status_text (str), sleep_time (int))
    """
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    
    # Market Hours
    market_open = time(9, 15)
    market_close = time(15, 30)
    
    is_weekday = now.weekday() < 5 # 0-4 is Mon-Fri
    is_trading_hours = market_open <= now.time() <= market_close
    
    if is_weekday and is_trading_hours:
        return True, "🟢 MARKET LIVE", 60 # Refresh every 60s
    else:
        return False, "🔴 MARKET CLOSED", 300 # Refresh every 5 mins (Rest Mode)

# --- 4. CORE FUNCTIONS ---

@st.cache_data(ttl=300)
def fetch_data_package(ticker):
    """Fetches History + Real-time info in one go"""
    stock = yf.Ticker(ticker)
    
    # 1. Get History (for charts)
    hist = stock.history(period="1y")
    
    # 2. Get Real-time Info (for Open/Close metrics)
    # yfinance 'info' is sometimes slow, so we fallback to history if needed
    try:
        info = stock.info
        todays_open = info.get('open', hist['Open'].iloc[-1])
        prev_close = info.get('previousClose', hist['Close'].iloc[-2])
        day_high = info.get('dayHigh', hist['High'].iloc[-1])
        day_low = info.get('dayLow', hist['Low'].iloc[-1])
    except:
        todays_open = hist['Open'].iloc[-1]
        prev_close = hist['Close'].iloc[-2]
        day_high = hist['High'].iloc[-1]
        day_low = hist['Low'].iloc[-1]
        
    return hist, todays_open, prev_close, day_high, day_low

def get_sentiment():
    sia = SentimentIntensityAnalyzer()
    articles = []
    for feed in NEWS_FEEDS:
        try:
            parsed = feedparser.parse(feed)
            for entry in parsed.entries[:3]:
                articles.append(entry.title)
        except: continue
    
    if not articles: return 0, []
    
    scores = [sia.polarity_scores(a)['compound'] for a in articles]
    return np.mean(scores), articles[:3]

# --- 5. MAIN APP ---

# Initialize Session State
if 'last_run' not in st.session_state: st.session_state['last_run'] = 0

# Market Status Check
is_open, status_msg, refresh_rate = get_market_status()

# Sidebar
st.sidebar.title("🦅 Market Watch")
selected_index = st.sidebar.selectbox("Index", list(INDICES.keys()))
ticker = INDICES[selected_index]
st.sidebar.markdown(f"**Status:** {status_msg}")
st.sidebar.caption(f"Update Rate: {refresh_rate}s")

# FETCH DATA
hist, open_p, prev_close, high_p, low_p = fetch_data_package(ticker)
current_price = hist['Close'].iloc[-1]
sentiment_score, headlines = get_sentiment()

# PREDICTION LOGIC
# If Market Open -> Predict Close
# If Market Closed -> Predict Open (Gap Up/Down)
change_pct = ((current_price - prev_close) / prev_close) 
total_bias = change_pct + (sentiment_score * 0.01) # Simple model

if is_open:
    prediction_label = "Predicted Close"
    predicted_value = current_price * (1 + (sentiment_score * 0.005)) # Intraday momentum
else:
    prediction_label = "Predicted Open (Tom)"
    # Overnight gap prediction based on Sentiment
    gap = sentiment_score * 0.015 # 1.5% max gap based on news
    predicted_value = current_price * (1 + gap)

# --- LAYOUT ---

# 1. Header & Status
c1, c2 = st.columns([3, 1])
with c1:
    st.title(f"{selected_index} Command Center")
    st.caption(f"Last Updated: {datetime.now().strftime('%H:%M:%S')} • {status_msg}")

# 2. KEY METRICS ROW (Open, Close, Prediction)
m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    st.metric("Current Price", f"₹{current_price:,.2f}", delta=f"{((current_price-prev_close)/prev_close)*100:.2f}%")
with m2:
    st.metric("Today's Open", f"₹{open_p:,.2f}", delta=f"{((open_p-prev_close)/prev_close)*100:.2f}%", delta_color="off")
with m3:
    st.metric("Day High", f"₹{high_p:,.2f}")
with m4:
    st.metric("Day Low", f"₹{low_p:,.2f}")
with m5:
    # THE AI PREDICTION
    color = "normal" if predicted_value > current_price else "inverse"
    st.metric(prediction_label, f"₹{predicted_value:,.2f}", delta=f"AI Bias: {sentiment_score:.2f}", delta_color=color)

st.divider()

# 3. CHART & NEWS
g1, g2 = st.columns([3, 1])

with g1:
    st.subheader("Technical View")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=hist.index,
                    open=hist['Open'], high=hist['High'],
                    low=hist['Low'], close=hist['Close'], name='Market'))
    
    # Add Prediction Marker
    if not is_open:
        # Show gap for tomorrow
        next_day = hist.index[-1] + timedelta(days=1)
        fig.add_trace(go.Scatter(x=[next_day], y=[predicted_value], mode='markers+text', 
                                 marker=dict(color='orange', size=12),
                                 text=["Exp. Open"], textposition="top center", name='AI Forecast'))

    fig.update_layout(height=450, template="plotly_dark", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

with g2:
    st.subheader("AI Sentiment")
    s_label = "🟢 BULLISH" if sentiment_score > 0.1 else "🔴 BEARISH" if sentiment_score < -0.1 else "⚪ NEUTRAL"
    st.info(f"Market Mood: {s_label}")
    st.write("---")
    st.caption("Key Drivers:")
    for h in headlines:
        st.write(f"• {h}")

# 4. CONSTITUENTS (Weights)
st.subheader("Index Movers")
# (Simplified for speed in this version)
const_data = yf.download(CONSTITUENTS[selected_index], period="1d", progress=False)['Close']
if not const_data.empty:
    latest = const_data.iloc[-1]
    # Simple table
    st.dataframe(latest.sort_values(ascending=False).head(5), use_container_width=True)

# --- AUTO-REFRESH LOGIC ---
# This script re-runs itself based on the sleep timer
time_module.sleep(refresh_rate)
st.rerun()
