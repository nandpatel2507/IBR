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
    page_title="Market Command Center",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Professional Dark Theme CSS
st.markdown("""
    <style>
    .stApp { background-color: #0E1117; }
    div[data-testid="stMetric"] {
        background-color: #1A1C24;
        border: 1px solid #333;
        padding: 15px;
        border-radius: 8px;
    }
    .stPlotlyChart { background-color: #1A1C24; border-radius: 8px; padding: 10px; }
    .stTabs [aria-selected="true"] { background-color: #FF4B4B !important; color: white !important; }
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

# --- 3. CORE LOGIC ---
def get_market_status():
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    if now.weekday() < 5 and (time(9,15) <= now.time() <= time(15,30)):
        return True, "🟢 LIVE", 60 
    return False, "🔴 CLOSED", 300

@st.cache_data(ttl=300)
def fetch_data_package(ticker):
    stock = yf.Ticker(ticker)
    hist_daily = stock.history(period="2y", interval="1d")
    hist_intraday = stock.history(period="1d", interval="5m")
    if not hist_daily.empty:
        hist_daily['SMA20'] = hist_daily['Close'].rolling(window=20).mean()
        hist_daily['STD20'] = hist_daily['Close'].rolling(window=20).std()
        hist_daily['Upper'] = hist_daily['SMA20'] + (hist_daily['STD20'] * 2)
        hist_daily['Lower'] = hist_daily['SMA20'] - (hist_daily['STD20'] * 2)
    if not hist_intraday.empty:
        hist_intraday['SMA20'] = hist_intraday['Close'].rolling(window=20).mean()
        hist_intraday['STD20'] = hist_intraday['Close'].rolling(window=20).std()
        hist_intraday['Upper'] = hist_intraday['SMA20'] + (hist_intraday['STD20'] * 2)
        hist_intraday['Lower'] = hist_intraday['SMA20'] - (hist_intraday['STD20'] * 2)
    try:
        info = stock.info
        todays_open = info.get('open', hist_daily['Open'].iloc[-1])
        prev_close = info.get('previousClose', hist_daily['Close'].iloc[-2])
        day_high = info.get('dayHigh', hist_daily['High'].iloc[-1])
        day_low = info.get('dayLow', hist_daily['Low'].iloc[-1])
    except:
        todays_open = hist_daily['Open'].iloc[-1]
        prev_close = hist_daily['Close'].iloc[-2]
        day_high = hist_daily['High'].iloc[-1]
        day_low = hist_daily['Low'].iloc[-1]
    return hist_daily, hist_intraday, todays_open, prev_close, day_high, day_low

@st.cache_data(ttl=3600)
def fetch_movers_data(const_tickers):
    try:
        data = yf.download(const_tickers, period="2d", group_by='ticker', progress=False, threads=False)
        table_data = []
        for t in const_tickers:
            try:
                if len(const_tickers) == 1: df_t = data
                else: 
                    if t not in data.columns.levels[0]: continue
                    df_t = data[t]
                if len(df_t) < 2: continue
                latest = df_t['Close'].iloc[-1]; prev = df_t['Close'].iloc[-2]
                if pd.isna(latest) or prev == 0: continue
                table_data.append({"Company": t.replace(".NS","").replace(".BO",""), "Price": latest, "Change %": ((latest - prev) / prev) * 100})
            except: continue
        return pd.DataFrame(table_data)
    except: return pd.DataFrame()

# --- 4. MAIN APP EXECUTION ---
is_open, status_msg, refresh_rate = get_market_status()
selected_index = st.sidebar.selectbox("Select Index", list(INDICES.keys()))
ticker = INDICES[selected_index]

hist_daily, hist_intraday, open_p, prev_close, high_p, low_p = fetch_data_package(ticker)
current_price = hist_daily['Close'].iloc[-1]

sia = SentimentIntensityAnalyzer()
articles = []
for feed in NEWS_FEEDS:
    try:
        parsed = feedparser.parse(feed)
        for entry in parsed.entries[:3]: articles.append(entry.title)
    except: continue
news_score = np.mean([sia.polarity_scores(a)['compound'] for a in articles]) if articles else 0

try:
    vix_data = yf.download("^INDIAVIX", period="5d", progress=False)['Close']
    current_vix = vix_data.iloc[-1]
except: current_vix = 15.0

pred_change = (news_score * 0.015) + (((current_price - prev_close)/prev_close)*0.5)
pred_price = current_price * (1 + pred_change)

# --- 5. RENDER UI ---
st.title(f"{selected_index} Command Center")
st.caption(f"Status: {status_msg} | Updates: {refresh_rate}s")

m = st.columns(5)
m[0].metric("Price", f"₹{current_price:,.2f}", f"{((current_price-prev_close)/prev_close)*100:.2f}%")
m[1].metric("Open", f"₹{open_p:,.2f}")
m[2].metric("High", f"₹{high_p:,.2f}")
m[3].metric("Low", f"₹{low_p:,.2f}")
m[4].metric("Forecast", f"₹{pred_price:,.2f}", f"AI:{news_score:.2f}")

g1, g2 = st.columns([3, 1])
with g1:
    t1, t2 = st.tabs(["Intraday", "Historical"])
    with t1:
        fig = go.Figure(go.Candlestick(x=hist_intraday.index, open=hist_intraday['Open'], high=hist_intraday['High'], low=hist_intraday['Low'], close=hist_intraday['Close']))
        fig.update_layout(template="plotly_dark", height=450, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
    with t2:
        fig_h = go.Figure(go.Scatter(x=hist_daily.index, y=hist_daily['Close'], line=dict(color='#00F0FF')))
        fig_h.update_layout(template="plotly_dark", height=450)
        st.plotly_chart(fig_h, use_container_width=True)

with g2:
    st.subheader("Sentiment")
    st.metric("VIX", f"{current_vix:.2f}")
    for h in articles: st.caption(f"• {h}")

st.subheader("Index Movers")
df_movers = fetch_movers_data(CONSTITUENTS[selected_index])
if not df_movers.empty:
    st.dataframe(df_movers.sort_values("Change %", ascending=False), use_container_width=True, hide_index=True)

time_module.sleep(refresh_rate)
st.rerun()
