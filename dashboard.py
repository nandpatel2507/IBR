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
    /* Main Background */
    .stApp {
        background-color: #0E1117;
    }
    /* Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #1A1C24;
        border: 1px solid #333;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.3);
    }
    /* Chart Container */
    .stPlotlyChart {
        background-color: #1A1C24;
        border-radius: 8px;
        padding: 10px;
    }
    /* Header Styling */
    h1, h2, h3 {
        color: #E0E0E0;
        font-family: 'Helvetica Neue', sans-serif;
    }
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1A1C24;
        border-radius: 4px;
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
    "NIFTY 50": [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "ITC.NS", 
        "BHARTIARTL.NS", "LT.NS", "SBIN.NS", "HINDUNILVR.NS", "BAJFINANCE.NS", 
        "MARUTI.NS", "AXISBANK.NS", "HCLTECH.NS", "TITAN.NS", "ASIANPAINT.NS", 
        "SUNPHARMA.NS", "ULTRACEMCO.NS", "TATASTEEL.NS", "NTPC.NS", "POWERGRID.NS", 
        "M&M.NS", "TATAMOTORS.NS", "ADANIENT.NS", "ADANIPORTS.NS", "BAJAJFINSV.NS", 
        "COALINDIA.NS", "ONGC.NS", "GRASIM.NS", "JSWSTEEL.NS", "TECHM.NS", 
        "HINDALCO.NS", "WIPRO.NS", "DIVISLAB.NS", "CIPLA.NS", "APOLLOHOSP.NS", 
        "DRREDDY.NS", "EICHERMOT.NS", "NESTLEIND.NS", "TATACONSUM.NS", "BRITANNIA.NS", 
        "SBILIFE.NS", "HDFCLIFE.NS", "HEROMOTOCO.NS", "KOTAKBANK.NS", "INDUSINDBK.NS", 
        "BPCL.NS", "SHRIRAMFIN.NS", "LTIM.NS"
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

# --- 3. LOGIC & CACHING ---
def get_market_status():
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    if now.weekday() < 5 and (time(9,15) <= now.time() <= time(15,30)):
        return True, "🟢 LIVE", 60 
    return False, "🔴 CLOSED", 300

@st.cache_data(ttl=300)
def fetch_data_package(ticker):
    stock = yf.Ticker(ticker)
    
    # Historical (Daily)
    hist_daily = stock.history(period="2y", interval="1d")
    
    # Intraday (5m)
    hist_intraday = stock.history(period="1d", interval="5m")
    
    # --- CALC BOLLINGER BANDS (STD DEV) FOR BOTH TIMEFRAMES ---
    
    # 1. Daily Bands
    if not hist_daily.empty:
        hist_daily['SMA20'] = hist_daily['Close'].rolling(window=20).mean()
        hist_daily['STD20'] = hist_daily['Close'].rolling(window=20).std()
        hist_daily['Upper'] = hist_daily['SMA20'] + (hist_daily['STD20'] * 2)
        hist_daily['Lower'] = hist_daily['SMA20'] - (hist_daily['STD20'] * 2)

    # 2. Intraday Bands (For "Daily Section" Standard Deviation)
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
    """Batch fetch movers (Cached 1hr)"""
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
                latest = df_t['Close'].iloc[-1]
                prev = df_t['Close'].iloc[-2]
                if pd.isna(latest) or prev == 0: continue
                chg_pct = ((latest - prev) / prev) * 100
                table_data.append({
                    "Company": t.replace(".NS","").replace(".BO",""),
                    "Price": latest,
                    "Change %": chg_pct,
                    "Trend": "🟢" if chg_pct > 0 else "🔴"
                })
            except: continue
        return pd.DataFrame(table_data)
    except: return pd.DataFrame()

@st.cache_data(ttl=1800)
def get_option_chain_pcr(ticker):
    try:
        stock = yf.Ticker(ticker)
        exps = stock.options
        if not exps: return 1.0
        opt = stock.option_chain(exps[0])
        calls_vol = opt.calls['volume'].sum()
        puts_vol = opt.puts['volume'].sum()
        return puts_vol / calls_vol if calls_vol > 0 else 1.0
    except: return 1.0

def get_fii_proxy():
    try:
        data = yf.download("INR=X", period="5d", progress=False)['Close']
        if isinstance(data, pd.DataFrame): data = data.iloc[:, 0]
        trend = (data.iloc[-1] - data.iloc[0])
        return "Selling 🔻" if trend > 0 else "Buying 🟢"
    except: return "Neutral ⚪"

def get_hybrid_sentiment():
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
    
    try:
        vix_data = yf.download("^INDIAVIX", period="5d", progress=False)['Close']
        if isinstance(vix_data, pd.DataFrame): vix_data = vix_data.iloc[:, 0]
        current_vix = vix_data.iloc[-1]
        vix_impact = -1 * ((current_vix - 15) / 10) 
        vix_impact = max(min(vix_impact, 1.0), -1.0) 
    except:
        vix_impact = 0
        current_vix = 15.0

    final_score = (news_score * 0.5) + (vix_impact * 0.5)
    return final_score, articles[:3], current_vix

# --- 4. MAIN APPLICATION ---
if 'last_run' not in st.session_state: st.session_state['last_run'] = 0
is_open, status_msg, refresh_rate = get_market_status()

# Sidebar
st.sidebar.title("🦅 Market Watch")
selected_index = st.sidebar.selectbox("Select Index", list(INDICES.keys()))
ticker = INDICES[selected_index]
st.sidebar.divider()
st.sidebar.info(f
