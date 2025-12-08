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
from yfinance.exceptions import YFRateLimitError

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="Market Command Center",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Professional Dark CSS
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
    </style>
""", unsafe_allow_html=True)

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

# --- 3. ROBUST DATA LOGIC ---
def get_market_status():
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    if now.weekday() < 5 and (time(9,15) <= now.time() <= time(15,30)):
        return True, "🟢 LIVE", 60 
    return False, "🔴 CLOSED", 300

@st.cache_data(ttl=600)
def fetch_data_with_retry(ticker):
    stock = yf.Ticker(ticker)
    backoff = 1
    for attempt in range(3): 
        try:
            # CHANGED TO 'max' FOR FULL HISTORY
            hist_daily = stock.history(period="max", interval="1d")
            hist_intraday = stock.history(period="1d", interval="5m")
            if hist_daily.empty: raise ValueError("Empty Data")
            return hist_daily, hist_intraday, stock.info
        except Exception:
            time_module.sleep(backoff)
            backoff *= 2
    return pd.DataFrame(), pd.DataFrame(), {}

@st.cache_data(ttl=3600)
def fetch_chunked_movers(const_tickers):
    chunk_size = 10
    chunks = [const_tickers[i:i + chunk_size] for i in range(0, len(const_tickers), chunk_size)]
    all_data = []
    progress_bar = st.progress(0, text="Scanning Market Data...")
    
    for i, chunk in enumerate(chunks):
        try:
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
                    chg_pct = ((latest - prev) / prev) * 100
                    all_data.append({
                        "Company": t.replace(".NS","").replace(".BO",""),
                        "Price": latest,
                        "Change %": chg_pct,
                        "Trend": "🟢" if chg_pct > 0 else "🔴"
                    })
                except: continue
            time_module.sleep(1.0)
            progress_bar.progress((i + 1) / len(chunks), text=f"Scanning Batch {i+1}/{len(chunks)}...")
        except Exception: continue
            
    progress_bar.empty()
    return pd.DataFrame(all_data)

# --- 4. MAIN APP ---
if 'last_run' not in st.session_state: st.session_state['last_run'] = 0
is_open, status_msg, refresh_rate = get_market_status()

# Sidebar
selected_index = st.sidebar.selectbox("Select Index", list(INDICES.keys()))
ticker = INDICES[selected_index]

# MAIN FETCH
hist_daily, hist_intraday, info = fetch_data_with_retry(ticker)

if hist_daily.empty:
    st.error("⚠️ Data Feed Disrupted. Yahoo Finance is rate-limiting requests. Please wait 30s and refresh.")
    st.stop()

# Extract Metrics SAFELY
try:
    current_price = float(hist_daily['Close'].iloc[-1])
    prev_close = float(hist_daily['Close'].iloc[-2])
    open_p = float(info.get('open', hist_daily['Open'].iloc[-1]))
    high_p = float(info.get('dayHigh', hist_daily['High'].iloc[-1]))
    low_p = float(info.get('dayLow', hist_daily['Low'].iloc[-1]))
except:
    current_price = 0.0; prev_close = 1.0; open_p = 0.0; high_p = 0.0; low_p = 0.0

# Sentiment & VIX
sia = SentimentIntensityAnalyzer()
articles = []
for feed in NEWS_FEEDS:
    try:
        parsed = feedparser.parse(feed)
        for entry in parsed.entries[:3]: articles.append(entry.title)
    except: continue
news_score = np.mean([sia.polarity_scores(a)['compound'] for a in articles]) if articles else 0

try:
    vix_df = yf.download("^INDIAVIX", period="5d", progress=False)
    vix_series = vix_df['Close'] if 'Close' in vix_df.columns else vix_df.iloc[:, 0]
    current_vix = float(vix_series.iloc[-1]) if not vix_series.empty else 15.0
except: current_vix = 15.0

# Prediction Logic
pred_change = (news_score * 0.015) + (((current_price - prev_close)/prev_close)*0.5)
pred_price = current_price * (1 + pred_change)

# --- 5. RENDER UI ---
st.title(f"{selected_index} Command Center")
st.caption(f"Status: {status_msg} | Update Rate: {refresh_rate}s")

m = st.columns(5)
m[0].metric("Price", f"₹{current_price:,.2f}", f"{((current_price-prev_close)/prev_close)*100:.2f}%")
m[1].metric("Open", f"₹{open_p:,.2f}")
m[2].metric("High", f"₹{high_p:,.2f}")
m[3].metric("Low", f"₹{low_p:,.2f}")
m[4].metric("AI Forecast", f"₹{pred_price:,.2f}", f"Sentiment:{news_score:.2f}")

st.markdown("---")

g1, g2 = st.columns([3, 1])
with g1:
    t1, t2 = st.tabs(["Intraday", "Historical (Max)"])
    
    # INTRADAY TAB
    with t1:
        if not hist_intraday.empty:
            fig = go.Figure(go.Candlestick(x=hist_intraday.index, open=hist_intraday['Open'], high=hist_intraday['High'], low=hist_intraday['Low'], close=hist_intraday['Close'], name='Price'))
            
            # PREDICTION TRAJECTORY (Visualizing the forecast)
            last_time = hist_intraday.index[-1]
            fig.add_trace(go.Scatter(
                x=[last_time, last_time + timedelta(hours=1)],
                y=[current_price, pred_price],
                mode='lines+markers', name='AI Prediction',
                line=dict(color='#FFA500', dash='dot', width=3)
            ))
            
            fig.update_layout(template="plotly_dark", height=450, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Intraday data unavailable")
    
    # HISTORICAL TAB
    with t2:
        # Prediction for next 5 days
        future_dates = [hist_daily.index[-1] + timedelta(days=i) for i in range(1, 6)]
        future_prices = [current_price]
        for _ in range(5): future_prices.append(future_prices[-1] * (1 + (pred_change/5)))
        future_prices.pop(0)

        fig_h = go.Figure()
        fig_h.add_trace(go.Scatter(x=hist_daily.index, y=hist_daily['Close'], line=dict(color='#00F0FF'), name='History'))
        
        # PREDICTION TRACE (Bright Orange)
        fig_h.add_trace(go.Scatter(
            x=future_dates, y=future_prices, 
            mode='lines+markers', name='AI Forecast (5D)',
            line=dict(color='#FFA500', dash='dot', width=3)
        ))
        
        # RESTORED YAHOO BUTTONS (Range Selector)
        fig_h.update_xaxes(
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
        fig_h.update_layout(template="plotly_dark", height=450)
        st.plotly_chart(fig_h, use_container_width=True)

with g2:
    st.subheader("Sentiment")
    st.metric("VIX Index", f"{current_vix:.2f}")
    for h in articles: st.caption(f"• {h}")

st.markdown("---")

# MOVERS
st.subheader(f"🏗️ {selected_index} Movers")
st.info("ℹ️ To prevent data crashes, Movers are loaded on demand.")

if st.button("🚀 Scan Market Movers (Safe Mode)"):
    df_movers = fetch_chunked_movers(CONSTITUENTS[selected_index])
    if not df_movers.empty:
        t_list, t_gain, t_loss = st.tabs(["Full List", "Top Gainers", "Top Losers"])
        with t_list: st.dataframe(df_movers, use_container_width=True)
        with t_gain: st.dataframe(df_movers.sort_values("Change %", ascending=False).head(10), use_container_width=True)
        with t_loss: st.dataframe(df_movers.sort_values("Change %", ascending=True).head(10), use_container_width=True)
    else:
        st.error("Could not fetch movers. Try again in 1 minute.")

time_module.sleep(refresh_rate)
st.rerun()
