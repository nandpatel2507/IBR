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
    
    # --- CALC BOLLINGER BANDS (STD DEV) ---
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
st.sidebar.info(f"Status: {status_msg}")
st.sidebar.caption(f"Next update in {refresh_rate}s")

# Fetch Data
hist_daily, hist_intraday, open_p, prev_close, high_p, low_p = fetch_data_package(ticker)
current_price = hist_daily['Close'].iloc[-1]
sentiment_score, headlines, current_vix = get_hybrid_sentiment()
fii_status = get_fii_proxy()
pcr_value = get_option_chain_pcr(ticker)

# Prediction Logic
pcr_bias = (pcr_value - 1) * 0.005
predicted_change = (sentiment_score * 0.015) + pcr_bias
if is_open:
    prediction_label = "Exp. Close"
    predicted_value = current_price * (1 + (sentiment_score * 0.005))
else:
    prediction_label = "Exp. Open (Tom)"
    gap = sentiment_score * 0.015 
    predicted_value = current_price * (1 + gap)

# --- 5. DASHBOARD UI ---

# Header Section
c1, c2 = st.columns([3, 1])
with c1:
    st.title(f"{selected_index} Command Center")
    st.markdown(f"**Live Analysis & AI Predictions** • *Last Update: {datetime.now().strftime('%H:%M:%S')}*")
with c2:
    if is_open:
        st.success(f"Market is {status_msg}")
    else:
        st.error(f"Market is {status_msg}")

st.markdown("---")

# METRICS
m1, m2, m3, m4, m5 = st.columns(5)
with m1: st.metric("Current Price", f"₹{current_price:,.2f}", delta=f"{((current_price-prev_close)/prev_close)*100:.2f}%")
with m2: st.metric("Today's Open", f"₹{open_p:,.2f}", delta=f"{((open_p-prev_close)/prev_close)*100:.2f}%", delta_color="off")
with m3: st.metric("Day High", f"₹{high_p:,.2f}")
with m4: st.metric("Day Low", f"₹{low_p:,.2f}")
with m5: 
    col = "normal" if predicted_value > current_price else "inverse"
    st.metric(prediction_label, f"₹{predicted_value:,.2f}", delta=f"AI Bias: {sentiment_score:.2f}", delta_color=col)

st.markdown("---")

# CHARTS
g1, g2 = st.columns([3, 1])

with g1:
    st.subheader("📊 Market Trends")
    tab_intra, tab_hist = st.tabs(["⏱️ Today (Intraday)", "📅 Historical Trends"])
    
    # 1. INTRADAY
    with tab_intra:
        if not hist_intraday.empty:
            show_intra_bb = st.toggle("Show Intraday Volatility (Std Dev)", value=True)
            
            fig_intra = go.Figure()
            
            # Candles
            fig_intra.add_trace(go.Candlestick(
                x=hist_intraday.index,
                open=hist_intraday['Open'], high=hist_intraday['High'],
                low=hist_intraday['Low'], close=hist_intraday['Close'],
                name='Live Price'
            ))
            
            # Intraday Bollinger Bands
            if show_intra_bb:
                fig_intra.add_trace(go.Scatter(x=hist_intraday.index, y=hist_intraday['Upper'], line=dict(color='rgba(255, 255, 0, 0.3)', width=1), name='Upper StdDev'))
                fig_intra.add_trace(go.Scatter(x=hist_intraday.index, y=hist_intraday['Lower'], line=dict(color='rgba(255, 255, 0, 0.3)', width=1), name='Lower StdDev', fill='tonexty', fillcolor='rgba(255, 255, 0, 0.05)'))

            # AI Prediction Line
            last_time = hist_intraday.index[-1]
            fig_intra.add_trace(go.Scatter(
                x=[last_time, last_time + timedelta(hours=1)],
                y=[current_price, predicted_value],
                mode='lines+markers', name='AI Trajectory',
                line=dict(color='#FFA500', dash='dot')
            ))
            
            fig_intra.update_layout(height=500, template="plotly_dark", xaxis_rangeslider_visible=False, title="Real-Time Price Action")
            st.plotly_chart(fig_intra, use_container_width=True)
        else:
            st.info("Waiting for market open data...")

    # 2. HISTORICAL
    with tab_hist:
        show_bb = st.toggle("Show Historical Bollinger Bands", value=False)
        
        future_dates = [hist_daily.index[-1] + timedelta(days=i) for i in range(1, 6)]
        future_prices = [current_price]
        for _ in range(5):
            drift = predicted_change / 5
            future_prices.append(future_prices[-1] * (1 + drift))
        future_prices.pop(0)
        
        # Cloud Logic
        daily_vol = hist_daily['Close'].pct_change().std()
        std_band = [daily_vol * price * 2 * np.sqrt(i+1) for i, price in enumerate(future_prices)]
        upper_b = [p + sd for p, sd in zip(future_prices, std_band)]
        lower_b = [p - sd for p, sd in zip(future_prices, std_band)]

        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(x=hist_daily.index, y=hist_daily['Close'], mode='lines', name='Price', line=dict(color='#00F0FF')))
        
        if show_bb:
            fig_hist.add_trace(go.Scatter(x=hist_daily.index, y=hist_daily['Upper'], line=dict(color='rgba(200,200,200,0.5)', width=1), name='Upper BB'))
            fig_hist.add_trace(go.Scatter(x=hist_daily.index, y=hist_daily['Lower'], line=dict(color='rgba(200,200,200,0.5)', width=1), name='Lower BB', fill='tonexty', fillcolor='rgba(200,200,200,0.05)'))

        fig_hist.add_trace(go.Scatter(x=future_dates, y=future_prices, mode='lines+markers', name='AI Forecast', line=dict(color='#FFA500', dash='dot')))
        fig_hist.add_trace(go.Scatter(x=future_dates + future_dates[::-1], y=upper_b + lower_b[::-1], fill='toself', fillcolor='rgba(255, 165, 0, 0.2)', line=dict(color='rgba(255,255,255,0)'), name='Uncertainty Cloud'))

        fig_hist.update_xaxes(rangeselector=dict(buttons=list([
            dict(count=1, label="1M", step="month", stepmode="backward"),
            dict(count=6, label="6M", step="month", stepmode="backward"),
            dict(count=1, label="YTD", step="year", stepmode="todate"),
            dict(count=1, label="1Y", step="year", stepmode="backward"),
            dict(step="all", label="MAX")
        ]), bgcolor="#262730"))
        
        fig_hist.update_layout(height=500, template="plotly_dark", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig_hist, use_container_width=True)

with g2:
    st.subheader("📡 Signals")
    with st.container(border=True):
        st.metric("FII Proxy (USD/INR)", fii_status, delta="Foreign Flow")
        st.write("---")
        pcr_c = "normal" if pcr_value > 1 else "inverse"
        st.metric("Put-Call Ratio", f"{pcr_value:.2f}", delta=">1 Bullish", delta_color=pcr_c)
        st.write("---")
        st.metric("India VIX", f"{current_vix:.2f}", delta="Fear Index", delta_color="inverse")
    
    st.subheader("📰 AI News")
    with st.container(border=True):
        for h in headlines:
            st.caption(f"• {h}")

st.markdown("---")

# MOVERS
st.subheader(f"🏗️ {selected_index} Movers & Shakers")
const_tickers = CONSTITUENTS[selected_index]
df_movers = fetch_movers_data(const_tickers)

if not df_movers.empty:
    t1, t2, t3 = st.tabs(["📋 Full List", "🚀 Top Gainers", "📉 Top Losers"])
    col_conf = {"Price": st.column_config.NumberColumn(format="₹%.2f"), "Change %": st.column_config.NumberColumn(format="%.2f%%")}
    with t1: st.dataframe(df_movers.sort_values("Company"), column_config=col_conf, use_container_width=True, hide_index=True)
    with t2: st.dataframe(df_movers.sort_values("Change %", ascending=False).head(10), column_config=col_conf, use_container_width=True, hide_index=True)
    with t3: st.dataframe(df_movers.sort_values("Change %", ascending=True).head(10), column_config=col_conf, use_container_width=True, hide_index=True)
else:
    st.info("Index constituent data is syncing...")

time_module.sleep(refresh_rate)
st.rerun()
