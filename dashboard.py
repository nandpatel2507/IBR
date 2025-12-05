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

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="AI Market Command Center",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="collapsed"
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
    .stPlotlyChart {
        background-color: #0e1117;
        border-radius: 5px;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
    }
    </style>
""", unsafe_allow_html=True)

# Download VADER lexicon (Run once)
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

# --- 2. EXPANDED DATA UNIVERSE (FULL LISTS) ---
INDICES = {
    "NIFTY 50": "^NSEI",
    "NIFTY BANK": "^NSEBANK",
    "NIFTY IT": "^CNXIT",
    "SENSEX": "^BSESN"
}

# FULL CONSTITUENTS (As requested)
CONSTITUENTS = {
    "NIFTY 50": [
        "ADANIENT.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS", "AXISBANK.NS",
        "BAJAJ-AUTO.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS", "BPCL.NS", "BHARTIARTL.NS",
        "BRITANNIA.NS", "CIPLA.NS", "COALINDIA.NS", "DIVISLAB.NS", "DRREDDY.NS",
        "EICHERMOT.NS", "GRASIM.NS", "HCLTECH.NS", "HDFCBANK.NS", "HDFCLIFE.NS",
        "HEROMOTOCO.NS", "HINDALCO.NS", "HINDUNILVR.NS", "ICICIBANK.NS", "ITC.NS",
        "INDUSINDBK.NS", "INFY.NS", "JSWSTEEL.NS", "KOTAKBANK.NS", "LTIM.NS",
        "LT.NS", "M&M.NS", "MARUTI.NS", "NTPC.NS", "NESTLEIND.NS",
        "ONGC.NS", "POWERGRID.NS", "RELIANCE.NS", "SBILIFE.NS", "SBIN.NS",
        "SUNPHARMA.NS", "TCS.NS", "TATACONSUM.NS", "TATAMOTORS.NS", "TATASTEEL.NS",
        "TECHM.NS", "TITAN.NS", "ULTRACEMCO.NS", "UPL.NS", "WIPRO.NS"
    ],
    "NIFTY BANK": [
        "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "AXISBANK.NS", "KOTAKBANK.NS",
        "INDUSINDBK.NS", "BANKBARODA.NS", "PNB.NS", "IDFCFIRSTB.NS", "AUBANK.NS",
        "FEDERALBNK.NS", "BANDHANBNK.NS"
    ],
    "NIFTY IT": [
        "TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS",
        "LTIM.NS", "PERSISTENT.NS", "COFORGE.NS", "MPHASIS.NS", "OFSS.NS"
    ],
    "SENSEX": [
        "RELIANCE.BO", "HDFCBANK.BO", "ICICIBANK.BO", "INFY.BO", "ITC.BO",
        "TCS.BO", "L&T.BO", "AXISBANK.BO", "KOTAKBANK.BO", "HINDUNILVR.BO",
        "SBIN.BO", "BHARTIARTL.BO", "BAJFINANCE.BO", "TATASTEEL.BO", "M&M.BO",
        "MARUTI.BO", "TITAN.BO", "SUNPHARMA.BO", "NESTLEIND.BO", "ADANIENT.BO",
        "ULTRACEMCO.BO", "JSWSTEEL.BO", "POWERGRID.BO", "TATASTR.BO", "INDUSINDBK.BO",
        "NTPC.BO", "HCLTECH.BO", "TECHM.BO", "WIPRO.BO", "ASIANPAINT.BO"
    ]
}

NEWS_FEEDS = [
    "https://www.livemint.com/rss/markets",
    "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms"
]

# --- 3. MARKET STATUS LOGIC (IST) ---
def get_market_status():
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    if now.weekday() < 5 and (time(9,15) <= now.time() <= time(15,30)):
        return True, "🟢 MARKET LIVE", 60 # Refresh every 60s
    return False, "🔴 MARKET CLOSED", 300 # Refresh every 5 mins

# --- 4. CORE FUNCTIONS ---

@st.cache_data(ttl=300)
def fetch_data_package(ticker):
    """Fetches History + Real-time info (Restored from your preferred layout)"""
    stock = yf.Ticker(ticker)
    
    # Get 2y History (Needed for 1Y/YTD buttons)
    hist = stock.history(period="2y")
    
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

def get_hybrid_sentiment():
    """Combines News + VIX (The 'Pro' Feature)"""
    sia = SentimentIntensityAnalyzer()
    articles = []
    for feed in NEWS_FEEDS:
        try:
            parsed = feedparser.parse(feed)
            for entry in parsed.entries[:3]:
                articles.append(entry.title)
        except: continue
    
    news_score = 0
    if articles:
        scores = [sia.polarity_scores(a)['compound'] for a in articles]
        news_score = np.mean(scores)
    
    # VIX Impact
    try:
        vix_data = yf.download("^INDIAVIX", period="5d", progress=False)['Close']
        current_vix = vix_data.iloc[-1]
        if isinstance(current_vix, pd.Series): current_vix = current_vix.iloc[0]
        # VIX > 20 is Fear (-ve score)
        vix_impact = -1 * ((current_vix - 15) / 10) 
        vix_impact = max(min(vix_impact, 1.0), -1.0) 
    except:
        vix_impact = 0
        current_vix = 15.0

    final_score = (news_score * 0.5) + (vix_impact * 0.5)
    return final_score, articles[:3], current_vix

# --- 5. MAIN APP ---

if 'last_run' not in st.session_state: st.session_state['last_run'] = 0

is_open, status_msg, refresh_rate = get_market_status()

st.sidebar.title("🦅 Market Watch")
selected_index = st.sidebar.selectbox("Index", list(INDICES.keys()))
ticker = INDICES[selected_index]
st.sidebar.markdown(f"**Status:** {status_msg}")
st.sidebar.caption(f"Update Rate: {refresh_rate}s")

# FETCH DATA
hist, open_p, prev_close, high_p, low_p = fetch_data_package(ticker)
current_price = hist['Close'].iloc[-1]
sentiment_score, headlines, current_vix = get_hybrid_sentiment()

# PREDICTION LOGIC
daily_volatility = hist['Close'].pct_change().std()
predicted_change = (sentiment_score * 0.015)

if is_open:
    prediction_label = "Predicted Close"
    predicted_value = current_price * (1 + (sentiment_score * 0.005))
else:
    prediction_label = "Predicted Open (Tom)"
    gap = sentiment_score * 0.015 
    predicted_value = current_price * (1 + gap)

# --- LAYOUT (THE ONE YOU LIKED) ---

# 1. Header
c1, c2 = st.columns([3, 1])
with c1:
    st.title(f"{selected_index} Command Center")
    st.caption(f"Last Updated: {datetime.now().strftime('%H:%M:%S')} • {status_msg}")

# 2. KEY METRICS ROW (5 Columns - Restored)
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
    color = "normal" if predicted_value > current_price else "inverse"
    st.metric(prediction_label, f"₹{predicted_value:,.2f}", delta=f"AI Bias: {sentiment_score:.2f}", delta_color=color)

st.divider()

# 3. CHART & NEWS (3:1 Split - Restored)
g1, g2 = st.columns([3, 1])

with g1:
    st.subheader("Technical View & Uncertainty")
    
    # --- UPGRADED GRAPH (Cloud + Buttons) ---
    future_dates = [hist.index[-1] + timedelta(days=i) for i in range(1, 6)]
    future_prices = [current_price]
    for _ in range(5):
        drift = predicted_change / 5
        future_prices.append(future_prices[-1] * (1 + drift))
    future_prices.pop(0)
    
    # Cloud Calculation
    std_dev_band = [daily_volatility * price * 2 * np.sqrt(i+1) for i, price in enumerate(future_prices)]
    upper_band = [p + sd for p, sd in zip(future_prices, std_dev_band)]
    lower_band = [p - sd for p, sd in zip(future_prices, std_dev_band)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], mode='lines', name='History', line=dict(color='#00F0FF')))
    fig.add_trace(go.Scatter(x=future_dates, y=future_prices, mode='lines+markers', name='AI Forecast', line=dict(color='#FFA500', dash='dot')))
    
    # The Cloud
    fig.add_trace(go.Scatter(
        x=future_dates + future_dates[::-1],
        y=upper_band + lower_band[::-1],
        fill='toself',
        fillcolor='rgba(255, 165, 0, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Risk Range (2σ)'
