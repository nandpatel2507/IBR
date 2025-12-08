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

# --- 1. APP CONFIGURATION ---
st.set_page_config(
    page_title="Market Command Center",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Professional Dark CSS
st.markdown("""
    <style>
    /* Main Background */
    .stApp { background-color: #0E1117; }
    
    /* Metrics Styling */
    div[data-testid="stMetric"] {
        background-color: #1A1C24;
        border: 1px solid #333;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    
    /* Chart Container */
    .stPlotlyChart { 
        background-color: #1A1C24; 
        border-radius: 8px; 
        padding: 10px;
        border: 1px solid #333;
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] { gap: 20px; }
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

# Download NLTK Data
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

# FULL CONSTITUENTS LIST
CONSTITUENTS = {
    "NIFTY 50": ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "ITC.NS", "BHARTIARTL.NS", "LT.NS", "SBIN.NS", "HINDUNILVR.NS", "BAJFINANCE.NS", "MARUTI.NS", "AXISBANK.NS", "HCLTECH.NS", "TITAN.NS", "ASIANPAINT.NS", "SUNPHARMA.NS", "ULTRACEMCO.NS", "TATASTEEL.NS", "NTPC.NS", "POWERGRID.NS", "M&M.NS", "TATAMOTORS.NS", "ADANIENT.NS", "ADANIPORTS.NS", "BAJAJFINSV.NS", "COALINDIA.NS", "ONGC.NS", "GRASIM.NS", "JSWSTEEL.NS", "TECHM.NS", "HINDALCO.NS", "WIPRO.NS", "DIVISLAB.NS", "CIPLA.NS", "APOLLOHOSP.NS", "DRREDDY.NS", "EICHERMOT.NS", "NESTLEIND.NS", "TATACONSUM.NS", "BRITANNIA.NS", "SBILIFE.NS", "HDFCLIFE.NS", "HEROMOTOCO.NS", "KOTAKBANK.NS", "INDUSINDBK.NS", "BPCL.NS", "SHRIRAMFIN.NS", "LTIM.NS"],
    "NIFTY BANK": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "AXISBANK.NS", "KOTAKBANK.NS", "INDUSINDBK.NS", "BANKBARODA.NS", "PNB.NS", "IDFCFIRSTB.NS", "AUBANK.NS", "FEDERALBNK.NS", "BANDHANBNK.NS"],
    "NIFTY IT": ["TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS", "LTIM.NS", "PERSISTENT.NS", "COFORGE.NS", "MPHASIS.NS", "OFSS.NS"],
    "SENSEX": ["RELIANCE.BO", "HDFCBANK.BO", "ICICIBANK.BO", "INFY.BO", "ITC.BO", "TCS.BO", "L&T.BO", "AXISBANK.BO", "KOTAKBANK.BO", "HINDUNILVR.BO", "SBIN.BO", "BHARTIARTL.BO", "BAJFINANCE.BO", "TATASTEEL.BO", "M&M.BO", "MARUTI.BO", "TITAN.BO", "SUNPHARMA.BO", "NESTLEIND.BO", "ADANIENT.BO", "ULTRACEMCO.BO", "JSWSTEEL.BO", "POWERGRID.BO", "TATASTR.BO", "INDUSINDBK.BO", "NTPC.BO", "HCLTECH.BO", "TECHM.BO", "WIPRO.BO", "ASIANPAINT.BO"]
}

NEWS_FEEDS = ["https://www.livemint.com/rss/markets", "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms"]

# --- 3. ROBUST DATA ENGINE (NO CRASH LOGIC) ---

def get_market_status():
    """Checks if Indian Market is Open (09:15 - 15:30 IST)"""
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    if now.weekday() < 5 and (time(9,15) <= now.time() <= time(15,30)):
        return True, "🟢 LIVE", 60 
    return False, "🔴 CLOSED", 300

@st.cache_data(ttl=600)
def fetch_main_data(ticker):
    """
    Fetches Max History & Intraday data.
    CRITICAL FIX: Does NOT fetch 'stock.info' to avoid Rate Limits.
    Calculates Open/High/Low directly from history.
    """
    stock = yf.Ticker(ticker)
    
    try:
        # 1. Fetch MAX history for the long-term chart
        hist_max = stock.history(period="max", interval="1d")
        
        # 2. Fetch Intraday for the 'Today' chart
        hist_intra = stock.history(period="1d", interval="5m")
        
        if hist_max.empty:
            return pd.DataFrame(), pd.DataFrame(), {}
            
        # 3. Extract Metrics Manually (Faster & Safer than API calls)
        latest_row = hist_max.iloc[-1]
        prev_row = hist_max.iloc[-2]
        
        metrics = {
            "current_price": latest_row['Close'],
            "prev_close": prev_row['Close'],
            "open": latest_row['Open'],
            "high": latest_row['High'],
            "low": latest_row['Low'],
            "volatility": hist_max['Close'].pct_change().std() * 100 # Standard Deviation
        }
        
        return hist_max, hist_intra, metrics
        
    except Exception as e:
        # Fail gracefully
        return pd.DataFrame(), pd.DataFrame(), {}

@st.cache_data(ttl=3600)
def fetch_chunked_movers(const_tickers):
    """
    Fetches constituents in chunks of 10 to avoid bans.
    """
    chunk_size = 10
    chunks = [const_tickers[i:i + chunk_size] for i in range(0, len(const_tickers), chunk_size)]
    all_data = []
    
    # Progress Bar UI
    progress_text = "Scanning Index Components..."
    my_bar = st.progress(0, text=progress_text)
    
    for i, chunk in enumerate(chunks):
        try:
            # Batch download without threads (safer for cloud)
            data = yf.download(chunk, period="2d", group_by='ticker', progress=False, threads=False)
            
            for t in chunk:
                try:
                    # Robust DataFrame handling
                    if len(chunk) == 1: df_t = data
                    else: 
                        if t not in data.columns.levels[0]: continue
                        df_t = data[t]
                    
                    if len(df_t) < 2: continue
                    
                    # Force Float Conversion
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
            
            # Throttling
            time_module.sleep(0.5)
            my_bar.progress((i + 1) / len(chunks), text=f"Scanning Batch {i+1}/{len(chunks)}")
            
        except Exception: continue
            
    my_bar.empty()
    return pd.DataFrame(all_data)

def get_sentiment():
    sia = SentimentIntensityAnalyzer()
    articles = []
    for feed in NEWS_FEEDS:
        try:
            parsed = feedparser.parse(feed)
            for entry in parsed.entries[:3]: articles.append(entry.title)
        except: continue
    
    if not articles: return 0, []
    scores = [sia.polarity_scores(a)['compound'] for a in articles]
    return np.mean(scores), articles

# --- 4. APP EXECUTION START ---

# Initialize Session State
if 'last_run' not in st.session_state: st.session_state['last_run'] = 0

# Check Market Status
is_open, status_msg, refresh_rate = get_market_status()

# Sidebar Setup
st.sidebar.title("🦅 Market Watch")
selected_index = st.sidebar.selectbox("Select Index", list(INDICES.keys()))
ticker = INDICES[selected_index]

# --- MAIN DATA FETCH ---
hist_max, hist_intra, metrics = fetch_main_data(ticker)

# Safety Check: If data failed to load
if hist_max.empty or not metrics:
    st.error("⚠️ Unable to load market data. The API might be busy. Please refresh in 1 minute.")
    st.stop()

# Extract Metrics safely (Force floats)
price = float(metrics['current_price'])
prev_c = float(metrics['prev_close'])
open_p = float(metrics['open'])
high_p = float(metrics['high'])
low_p = float(metrics['low'])
std_dev = float(metrics['volatility']) # Standard Deviation

# Sentiment & Prediction
news_score, headlines = get_sentiment()

# Prediction Algo
pred_change = (news_score * 0.015) + (((price - prev_c)/prev_c)*0.5)
pred_price = price * (1 + pred_change)

# Label Logic
if is_open:
    pred_label = "Exp. Close"
else:
    pred_label = "Exp. Open (Tom)"

# --- 5. VISUAL DASHBOARD ---

st.title(f"{selected_index} Command Center")
st.caption(f"Status: {status_msg} | Std Dev (Volatility): {std_dev:.2f}%")

# METRICS ROW
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Current Price", f"₹{price:,.2f}", f"{((price-prev_c)/prev_c)*100:.2f}%")
col2.metric("Open", f"₹{open_p:,.2f}", delta_color="off")
col3.metric("High", f"₹{high_p:,.2f}", delta_color="off")
col4.metric("Low", f"₹{low_p:,.2f}", delta_color="off")
# Prediction Metric
p_col = "normal" if pred_price > price else "inverse"
col5.metric(pred_label, f"₹{pred_price:,.2f}", f"AI: {news_score:.2f}", delta_color=p_col)

st.markdown("---")

# GRAPH SECTION
g_col1, g_col2 = st.columns([3, 1])

with g_col1:
    st.subheader("📈 Market Trends & Forecast")
    
    # TABS
    tab_live, tab_hist = st.tabs(["⏱️ Today (Live)", "📅 Historical (Max)"])
    
    # 1. LIVE CHART
    with tab_live:
        if not hist_intra.empty:
            fig_live = go.Figure(go.Candlestick(
                x=hist_intra.index,
                open=hist_intra['Open'], high=hist_intra['High'],
                low=hist_intra['Low'], close=hist_intra['Close'],
                name='Live Price'
            ))
            # Live Prediction Line
            last_time = hist_intra.index[-1]
            fig_live.add_trace(go.Scatter(
                x=[last_time, last_time + timedelta(hours=1)],
                y=[price, pred_price],
                mode='lines+markers',
                name='AI Forecast',
                line=dict(color='#FFA500', dash='dot', width=3)
            ))
            fig_live.update_layout(template="plotly_dark", height=500, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig_live, use_container_width=True)
        else:
            st.info("Intraday data currently unavailable.")

    # 2. HISTORICAL CHART
    with tab_hist:
        # Prepare Prediction Data (5 Days into future)
        future_dates = [hist_max.index[-1] + timedelta(days=i) for i in range(1, 6)]
        future_prices = [price]
        for _ in range(5):
            future_prices.append(future_prices[-1] * (1 + (pred_change/5)))
        future_prices.pop(0) # Remove start point

        fig_hist = go.Figure()
        
        # Main Price Line
        fig_hist.add_trace(go.Scatter(
            x=hist_max.index, 
            y=hist_max['Close'], 
            mode='lines', 
            name='Price History', 
            line=dict(color='#00F0FF', width=1.5)
        ))
        
        # Prediction Line (Orange Dotted)
        fig_hist.add_trace(go.Scatter(
            x=future_dates, 
            y=future_prices, 
            mode='lines+markers', 
            name='AI Forecast (5D)',
            line=dict(color='#FFA500', dash='dot', width=2)
        ))

        # YAHOO STYLE BUTTONS
        fig_hist.update_xaxes(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(count=5, label="5Y", step="year", stepmode="backward"),
                    dict(step="all", label="MAX")
                ]),
                bgcolor="#262730",
                font=dict(color="white")
            )
        )
        fig_hist.update_layout(template="plotly_dark", height=500, xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig_hist, use_container_width=True)

# NEWS SECTION
with g_col2:
    st.subheader("📰 AI Analysis")
    with st.container(border=True):
        st.caption(f"Sentiment Score: {news_score:.2f}")
        for h in headlines:
            st.markdown(f"• {h}")
    
    st.write("")
    st.subheader("⚙️ Volatility")
    st.metric("Std Deviation", f"{std_dev:.2f}%", help="Annualized Standard Deviation based on historical data")

st.markdown("---")

# MOVERS SECTION (Lazy Load for Stability)
st.subheader(f"🏗️ {selected_index} Components")
st.info("ℹ️ Click below to scan all 30/50 stocks. This loads live data batch-by-batch.")

if st.button("🚀 Scan Market Movers"):
    df_movers = fetch_chunked_movers(CONSTITUENTS[selected_index])
    
    if not df_movers.empty:
        t1, t2, t3 = st.tabs(["📋 Full List", "🚀 Gainers", "📉 Losers"])
        
        # Formatting Config
        cfg = {
            "Price": st.column_config.NumberColumn(format="₹%.2f"),
            "Change %": st.column_config.NumberColumn(format="%.2f%%")
        }
        
        with t1: st.dataframe(df_movers.sort_values("Company"), column_config=cfg, use_container_width=True, hide_index=True)
        with t2: st.dataframe(df_movers.sort_values("Change %", ascending=False).head(10), column_config=cfg, use_container_width=True, hide_index=True)
        with t3: st.dataframe(df_movers.sort_values("Change %", ascending=True).head(10), column_config=cfg, use_container_width=True, hide_index=True)
    else:
        st.warning("⚠️ Could not fetch constituents. Try again later.")

# Auto-Refresh Logic
time_module.sleep(refresh_rate)
st.rerun()
