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

# Custom CSS for "Glassmorphism" look
st.markdown("""
    <style>
    .stApp { background-color: #0E1117; }
    
    /* Metrics Styling */
    div[data-testid="stMetric"] {
        background-color: #1A1C24;
        border: 1px solid #2D2D2D;
        padding: 15px;
        border-radius: 10px;
        transition: transform 0.2s;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        border-color: #4A4A4A;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #1A1C24;
        border-radius: 4px;
        color: #A0A0A0;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FF4B4B !important;
        color: white !important;
    }
    
    /* Chart Border */
    .stPlotlyChart {
        background-color: #1A1C24;
        border: 1px solid #2D2D2D;
        border-radius: 10px;
        padding: 15px;
    }
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

# --- 3. ROBUST DATA ENGINE ---

def get_market_status():
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    if now.weekday() < 5 and (time(9,15) <= now.time() <= time(15,30)):
        return True, "🟢 MARKET LIVE", 60 
    return False, "🔴 MARKET CLOSED", 300

@st.cache_data(ttl=600)
def fetch_main_data(ticker):
    stock = yf.Ticker(ticker)
    try:
        hist_max = stock.history(period="max", interval="1d")
        hist_intra = stock.history(period="1d", interval="5m")
        if hist_max.empty: return pd.DataFrame(), pd.DataFrame(), {}
        
        metrics = {
            "price": float(hist_max['Close'].iloc[-1]),
            "prev": float(hist_max['Close'].iloc[-2]),
            "open": float(hist_max['Open'].iloc[-1]),
            "high": float(hist_max['High'].iloc[-1]),
            "low": float(hist_max['Low'].iloc[-1]),
            "volatility": float(hist_max['Close'].pct_change().std() * 100)
        }
        return hist_max, hist_intra, metrics
    except: return pd.DataFrame(), pd.DataFrame(), {}

@st.cache_data(ttl=3600)
def fetch_movers_batch(const_tickers):
    """
    Chunked fetching + Strict Deduplication
    """
    chunk_size = 10
    chunks = [const_tickers[i:i + chunk_size] for i in range(0, len(const_tickers), chunk_size)]
    all_data = []
    
    prog = st.progress(0, "Scanning Market Depth...")
    
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
                    
                    all_data.append({
                        "Company": t.replace(".NS","").replace(".BO",""),
                        "Price": latest,
                        "Change %": ((latest - prev) / prev) * 100,
                        "Volume": float(df_t['Volume'].iloc[-1])
                    })
                except: continue
            time_module.sleep(0.5)
            prog.progress((i + 1) / len(chunks))
        except: continue
            
    prog.empty()
    df = pd.DataFrame(all_data)
    
    # STRICT DEDUPLICATION
    if not df.empty:
        df = df.drop_duplicates(subset=['Company'])
        
    return df

def get_sentiment():
    sia = SentimentIntensityAnalyzer()
    articles = []
    for feed in NEWS_FEEDS:
        try:
            parsed = feedparser.parse(feed)
            for entry in parsed.entries[:3]: articles.append(entry.title)
        except: continue
    news_score = np.mean([sia.polarity_scores(a)['compound'] for a in articles]) if articles else 0
    return news_score, articles

# --- 4. APP EXECUTION ---
if 'last_run' not in st.session_state: st.session_state['last_run'] = 0
is_open, status_msg, refresh_rate = get_market_status()

# Sidebar
selected_index = st.sidebar.selectbox("Select Index", list(INDICES.keys()))
ticker = INDICES[selected_index]

# Main Fetch
hist_max, hist_intra, metrics = fetch_main_data(ticker)

if hist_max.empty:
    st.error("⚠️ Data unavailable. API Limit Reached. Please wait 1 min.")
    st.stop()

# Calculations
news_score, headlines = get_sentiment()
pred_change = (news_score * 0.015) + (((metrics['price'] - metrics['prev'])/metrics['prev'])*0.5)
pred_price = metrics['price'] * (1 + pred_change)

# --- 5. VISUAL DASHBOARD ---

c1, c2 = st.columns([3, 1])
with c1:
    st.title(f"{selected_index}")
    st.caption(f"Real-Time Data Feed • {status_msg}")
with c2:
    if is_open: st.success(f"Status: {status_msg}")
    else: st.error(f"Status: {status_msg}")

# Heads Up Display
m = st.columns(5)
m[0].metric("Price", f"₹{metrics['price']:,.2f}", f"{((metrics['price']-metrics['prev'])/metrics['prev'])*100:.2f}%")
m[1].metric("Open", f"₹{metrics['open']:,.2f}", delta_color="off")
m[2].metric("High", f"₹{metrics['high']:,.2f}", delta_color="off")
m[3].metric("Low", f"₹{metrics['low']:,.2f}", delta_color="off")
m[4].metric("AI Target", f"₹{pred_price:,.2f}", f"Sent: {news_score:.2f}", delta_color="normal")

st.markdown("---")

# --- MAIN GRAPH AREA ---
g_col, s_col = st.columns([3, 1])

with g_col:
    # Time Range Selector
    time_range = st.radio("Time Range", ["1D", "1M", "6M", "YTD", "1Y", "MAX"], horizontal=True, label_visibility="collapsed")
    
    fig = go.Figure()
    
    # Logic to switch between Intraday and Historical
    if time_range == "1D":
        if not hist_intra.empty:
            fig.add_trace(go.Scatter(x=hist_intra.index, y=hist_intra['Close'], mode='lines', name='Price', line=dict(color='#00F0FF', width=2)))
            # Forecast Line
            last_time = hist_intra.index[-1]
            fig.add_trace(go.Scatter(
                x=[last_time, last_time + timedelta(hours=1)],
                y=[metrics['price'], pred_price],
                mode='lines+markers', name='Forecast',
                line=dict(color='#FFA500', dash='dot', width=2)
            ))
            fig.update_layout(title=f"Intraday Action")
        else:
            st.warning("Intraday data unavailable (Market Closed). Switch to 1M.")
    
    else:
        df_plot = hist_max.copy()
        end_date = df_plot.index[-1]
        if time_range == "1M": start_date = end_date - timedelta(days=30)
        elif time_range == "6M": start_date = end_date - timedelta(days=180)
        elif time_range == "1Y": start_date = end_date - timedelta(days=365)
        elif time_range == "YTD": start_date = datetime(end_date.year, 1, 1).replace(tzinfo=end_date.tzinfo)
        else: start_date = df_plot.index[0] # MAX
        
        df_plot = df_plot[df_plot.index >= start_date]
        
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['Close'], mode='lines', name='Price', line=dict(color='#00F0FF')))
        
        # Forecast 5D
        future_dates = [df_plot.index[-1] + timedelta(days=i) for i in range(1, 6)]
        future_prices = [metrics['price'] * (1 + (pred_change/5)*i) for i in range(1, 6)]
        
        fig.add_trace(go.Scatter(x=future_dates, y=future_prices, mode='lines+markers', name='AI Forecast', line=dict(color='#FFA500', dash='dot')))
        fig.update_layout(title=f"{time_range} Trend Analysis")

    fig.update_layout(template="plotly_dark", height=500, xaxis_rangeslider_visible=False, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig, use_container_width=True)

with s_col:
    st.subheader("📰 AI Insights")
    with st.container(border=True):
        st.info(f"Model predicts a {pred_change*100:.2f}% move based on current volatility of {metrics['volatility']:.2f}%.")
        for h in headlines:
            st.caption(f"• {h}")

st.markdown("---")

# --- MOVERS SECTION (VISUALLY ENHANCED) ---
st.subheader("📊 Market Depth & Components")

# Using a container for the button keeps it clean
with st.container(border=True):
    col_btn, col_info = st.columns([1, 4])
    with col_btn:
        load_btn = st.button("🚀 Load Movers Table", use_container_width=True)
    with col_info:
        st.caption("Click to scan all constituent stocks. Data is fetched in batches to prevent API blocks.")

if load_btn:
    df_movers = fetch_movers_batch(CONSTITUENTS[selected_index])
    
    if not df_movers.empty:
        t_all, t_gain, t_loss = st.tabs(["📋 Full List", "🟢 Top Gainers", "🔴 Top Losers"])
        
        # PRO COLUMN CONFIGURATION
        column_cfg = {
            "Price": st.column_config.NumberColumn(format="₹%.2f"),
            "Change %": st.column_config.ProgressColumn(
                "Change %", 
                format="%.2f%%", 
                min_value=-5, 
                max_value=5
            ),
            "Volume": st.column_config.NumberColumn(format="%d")
        }
        
        with t_all:
            st.dataframe(df_movers.sort_values("Company"), column_config=column_cfg, use_container_width=True, hide_index=True)
        with t_gain:
            st.dataframe(df_movers.sort_values("Change %", ascending=False).head(10), column_config=column_cfg, use_container_width=True, hide_index=True)
        with t_loss:
            st.dataframe(df_movers.sort_values("Change %", ascending=True).head(10), column_config=column_cfg, use_container_width=True, hide_index=True)
    else:
        st.error("Data fetch failed. Try again in 30 seconds.")

# Auto Refresh
time_module.sleep(refresh_rate)
st.rerun()
