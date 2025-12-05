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

# Custom CSS
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

# Download VADER (Run once)
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

# --- 3. MARKET STATUS LOGIC ---
def get_market_status():
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    # Simple Logic: Mon-Fri, 9:15 to 15:30
    if now.weekday() < 5 and (time(9,15) <= now.time() <= time(15,30)):
        return True, "🟢 MARKET LIVE", 60 
    return False, "🔴 MARKET CLOSED", 300

# --- 4. CORE FUNCTIONS ---
@st.cache_data(ttl=300)
def fetch_data_package(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="2y") # 2y for graph buttons
    try:
        info = stock.info
        # Use info if available, else fallback to history
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
    sia = SentimentIntensityAnalyzer()
    articles = []
    # Fetch News
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
    
    # Fetch VIX
    try:
        vix_data = yf.download("^INDIAVIX", period="5d", progress=False)['Close']
        current_vix = vix_data.iloc[-1]
        if isinstance(current_vix, pd.Series): current_vix = current_vix.iloc[0]
        # VIX > 20 = Fear (-1.0), VIX < 12 = Greed (+1.0)
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

# PREDICTION
predicted_change = (sentiment_score * 0.015)
if is_open:
    prediction_label = "Predicted Close"
    predicted_value = current_price * (1 + (sentiment_score * 0.005))
else:
    prediction_label = "Predicted Open (Tom)"
    gap = sentiment_score * 0.015 
    predicted_value = current_price * (1 + gap)

# --- LAYOUT ---

# 1. HEADER
c1, c2 = st.columns([3, 1])
with c1:
    st.title(f"{selected_index} Command Center")
    st.caption(f"Last Updated: {datetime.now().strftime('%H:%M:%S')} • {status_msg}")

# 2. KEY METRICS (5 Cols)
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

# 3. CHART & NEWS
g1, g2 = st.columns([3, 1])

with g1:
    st.subheader("Technical View & Uncertainty")
    
    # Calculate Forecast
    future_dates = [hist.index[-1] + timedelta(days=i) for i in range(1, 6)]
    future_prices = [current_price]
    for _ in range(5):
        drift = predicted_change / 5
        future_prices.append(future_prices[-1] * (1 + drift))
    future_prices.pop(0)
    
    # Calculate Cloud (Standard Deviation)
    daily_volatility = hist['Close'].pct_change().std()
    std_dev_band = [daily_volatility * price * 2 * np.sqrt(i+1) for i, price in enumerate(future_prices)]
    upper_band = [p + sd for p, sd in zip(future_prices, std_dev_band)]
    lower_band = [p - sd for p, sd in zip(future_prices, std_dev_band)]

    fig = go.Figure()
    
    # Trace 1: History
    fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], mode='lines', name='History', line=dict(color='#00F0FF')))
    
    # Trace 2: Prediction
    fig.add_trace(go.Scatter(x=future_dates, y=future_prices, mode='lines+markers', name='AI Forecast', line=dict(color='#FFA500', dash='dot')))
    
    # Trace 3: Cloud (Fixed Syntax)
    fig.add_trace(go.Scatter(
        x=future_dates + future_dates[::-1],
        y=upper_band + lower_band[::-1],
        fill='toself',
        fillcolor='rgba(255, 165, 0, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Risk Range (2σ)'
    ))

    # Yahoo Buttons
    fig.update_xaxes(
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1M", step="month", stepmode="backward"),
                dict(count=6, label="6M", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1Y", step="year", stepmode="backward"),
                dict(step="all", label="MAX")
            ]),
            bgcolor="#262730"
        )
    )
    fig.update_layout(height=450, template="plotly_dark", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

with g2:
    st.subheader("AI Sentiment")
    st.metric("India VIX (Fear)", f"{current_vix:.2f}", delta="Volatility Impact", delta_color="inverse")
    st.write("---")
    
    s_label = "🟢 BULLISH" if sentiment_score > 0.1 else "🔴 BEARISH" if sentiment_score < -0.1 else "⚪ NEUTRAL"
    st.info(f"Market Mood: {s_label}")
    st.caption("Key Drivers:")
    for h in headlines:
        st.write(f"• {h}")

st.divider()

# 4. CONSTITUENTS (Tabs)
st.subheader(f"🏗️ {selected_index} Movers (Live)")
st.caption("Tracking all index constituents. Prices delayed by ~1-2 mins.")

const_tickers = CONSTITUENTS[selected_index]
# Batch Fetch
data = yf.download(const_tickers, period="2d", group_by='ticker', progress=False)

table_data = []
for t in const_tickers:
    try:
        # Check if we have data for this ticker
        if t in data.columns.levels[0]:
            df_t = data[t]
            latest = df_t['Close'].iloc[-1]
            prev = df_t['Close'].iloc[-2]
            chg_pct = ((latest - prev) / prev) * 100
            
            table_data.append({
                "Company": t.replace(".NS", "").replace(".BO", ""),
                "Price": latest,
                "Change %": chg_pct,
                "Trend": "🟢" if chg_pct > 0 else "🔴"
            })
    except: continue

if table_data:
    df_movers = pd.DataFrame(table_data)
    
    t1, t2, t3 = st.tabs(["📋 Full List", "🚀 Top Gainers", "📉 Top Losers"])
    
    col_conf = {
        "Price": st.column_config.NumberColumn(format="₹%.2f"),
        "Change %": st.column_config.NumberColumn(format="%.2f%%"),
    }
    
    with t1:
        st.dataframe(df_movers.sort_values("Company"), column_config=col_conf, use_container_width=True, hide_index=True)
    with t2:
        st.dataframe(df_movers.sort_values("Change %", ascending=False).head(10), column_config=col_conf, use_container_width=True, hide_index=True)
    with t3:
        st.dataframe(df_movers.sort_values("Change %", ascending=True).head(10), column_config=col_conf, use_container_width=True, hide_index=True)

# Auto Refresh
time_module.sleep(refresh_rate)
st.rerun()
