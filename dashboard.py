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

# Custom CSS for Pro Look
st.markdown("""
    <style>
    .metric-container {
        background-color: #1e1e1e;
        padding: 10px;
        border-radius: 8px;
        border: 1px solid #333;
    }
    /* Make the graph container pop */
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

# --- 2. EXPANDED DATA UNIVERSE ---
INDICES = {
    "NIFTY 50": "^NSEI",
    "NIFTY BANK": "^NSEBANK",
    "NIFTY IT": "^CNXIT",
    "SENSEX": "^BSESN"
}

# FULL CONSTITUENT LISTS
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

# --- 3. ANALYTICS LOGIC ---

def get_market_status():
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    if now.weekday() < 5 and (time(9,15) <= now.time() <= time(15,30)):
        return True, "🟢 MARKET LIVE", 60
    return False, "🔴 MARKET CLOSED", 300

@st.cache_data(ttl=300)
def fetch_main_data(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="2y") # Fetched 2y for graph buttons
    return hist

def get_hybrid_sentiment():
    """
    Combines News Sentiment (RSS) + Market Fear (VIX)
    """
    # 1. RSS Sentiment
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
    
    # 2. VIX (Fear) Sentiment
    try:
        vix_data = yf.download("^INDIAVIX", period="5d", progress=False)['Close']
        current_vix = vix_data.iloc[-1]
        if isinstance(current_vix, pd.Series): current_vix = current_vix.iloc[0]
        
        # Logic: VIX > 20 is Fear (-ve), VIX < 12 is Complacency/Greed (+ve)
        # Normalizing VIX impact inverted (High VIX = Low Score)
        vix_impact = -1 * ((current_vix - 15) / 10) # 15 is baseline
        vix_impact = max(min(vix_impact, 1.0), -1.0) # Clamp between -1 and 1
    except:
        vix_impact = 0
        current_vix = 15.0

    # Composite Score (50% News, 50% VIX)
    final_score = (news_score * 0.5) + (vix_impact * 0.5)
    
    return final_score, articles[:3], current_vix

# --- 4. MAIN APP ---

if 'last_run' not in st.session_state: st.session_state['last_run'] = 0

is_open, status_msg, refresh_rate = get_market_status()

st.sidebar.title("🦅 Market Watch")
selected_index = st.sidebar.selectbox("Select Index", list(INDICES.keys()))
ticker = INDICES[selected_index]
st.sidebar.markdown(f"**Status:** {status_msg}")

# Fetch Data
hist = fetch_main_data(ticker)
current_price = hist['Close'].iloc[-1]
prev_close = hist['Close'].iloc[-2]
sentiment_score, headlines, current_vix = get_hybrid_sentiment()

# Prediction & Volatility Logic
daily_volatility = hist['Close'].pct_change().std()
# Adjust prediction based on Sentiment
predicted_change = (sentiment_score * 0.015) # 1.5% swing based on mood
predicted_price = current_price * (1 + predicted_change)

# --- LAYOUT ---
st.title(f"{selected_index} Pro Terminal")
st.caption(f"Last Updated: {datetime.now().strftime('%H:%M:%S')} • {status_msg}")

# METRICS ROW
c1, c2, c3, c4 = st.columns(4)
change_val = current_price - prev_close
change_pct = (change_val / prev_close) * 100

c1.metric("Current Level", f"₹{current_price:,.2f}", delta=f"{change_pct:.2f}%")
c2.metric("AI Forecast (5D)", f"₹{predicted_price:,.2f}", delta=f"Composite Bias: {sentiment_score:.2f}")
c3.metric("India VIX (Fear)", f"{current_vix:.2f}", delta="Volatility Index", delta_color="off")
c4.metric("Risk Level", "HIGH ⚡" if current_vix > 20 else "LOW 🛡️")

st.divider()

# --- PRO GRAPH SECTION (Yahoo Style + Hazy Cloud) ---
g1, g2 = st.columns([3, 1])

with g1:
    st.subheader("📈 Technical Forecast & Uncertainty")
    
    # 1. Prepare Prediction Data (5 Days)
    future_dates = [hist.index[-1] + timedelta(days=i) for i in range(1, 6)]
    # Random walk drift based on sentiment
    future_prices = [current_price]
    for _ in range(5):
        drift = predicted_change / 5 # distribute impact over 5 days
        future_prices.append(future_prices[-1] * (1 + drift))
    future_prices.pop(0)
    
    # 2. Prepare Cloud (Standard Deviation)
    # We project 2 Standard Deviations (95% confidence interval)
    std_dev_band = [daily_volatility * price * 2 * np.sqrt(i+1) for i, price in enumerate(future_prices)]
    upper_band = [p + sd for p, sd in zip(future_prices, std_dev_band)]
    lower_band = [p - sd for p, sd in zip(future_prices, std_dev_band)]

    fig = go.Figure()
    
    # Historical Line
    fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], mode='lines', name='History', line=dict(color='#00F0FF', width=2)))
    
    # Prediction Line
    fig.add_trace(go.Scatter(x=future_dates, y=future_prices, mode='lines+markers', name='AI Forecast', line=dict(color='#FFA500', dash='dot')))
    
    # The Hazy Cloud (Uncertainty)
    fig.add_trace(go.Scatter(
        x=future_dates + future_dates[::-1],
        y=upper_band + lower_band[::-1],
        fill='toself',
        fillcolor='rgba(255, 165, 0, 0.2)', # Hazy Orange
        line=dict(color='rgba(255,255,255,0)'),
        name='Risk Range (2σ)'
    ))

    # Yahoo-Style Range Selectors
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
    
    fig.update_layout(height=500, template="plotly_dark", hovermode="x unified", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig, use_container_width=True)

with g2:
    st.subheader("📰 Sentiment Drivers")
    st.caption("Live Headlines:")
    for h in headlines:
        st.info(f"• {h}")
    st.write("---")
    st.caption("Market mood is calculated using a weighted average of RSS News Sentiment and India VIX (Fear Index).")

st.divider()

# --- CONSTITUENTS TABLE (Full List + Icons) ---
st.subheader(f"🏗️ {selected_index} Movers")
st.caption("Tracking all index constituents. Prices delayed by ~1 min.")

const_tickers = CONSTITUENTS[selected_index]
# Batch Fetch
data = yf.download(const_tickers, period="2d", group_by='ticker', progress=False)

table_data = []
for t in const_tickers:
    try:
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
    
    # Tabs for sorting
    t1, t2, t3 = st.tabs(["📋 Full List", "🚀 Top Gainers", "wv Top Losers"])
    
    # Common column config
    col_conf = {
        "Price": st.column_config.NumberColumn(format="₹%.2f"),
        "Change %": st.column_config.NumberColumn(format="%.2f%%"),
    }
    
    with t1:
        st.dataframe(df_movers.sort_values("Company"), column_config=col_conf, use_container_width=True, hide_index=True, height=400)
    with t2:
        st.dataframe(df_movers.sort_values("Change %", ascending=False).head(10), column_config=col_conf, use_container_width=True, hide_index=True)
    with t3:
        st.dataframe(df_movers.sort_values("Change %", ascending=True).head(10), column_config=col_conf, use_container_width=True, hide_index=True)

# Auto Refresh
time_module.sleep(refresh_rate)
st.rerun()
