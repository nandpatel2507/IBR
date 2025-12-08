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
    page_title="Market Command Center",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

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

# --- 3. ADVANCED QUANT LOGIC ---

def calculate_rsi(series, period=14):
    """Calculates Relative Strength Index (Technical Indicator)"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def get_smart_prediction(hist_df, news_score, volatility):
    """
    CORPORATE FINANCE & TECHNICAL COMPOSITE MODEL
    ---------------------------------------------
    1. Historical Drift (CAGR): What is the asset's natural growth rate?
    2. RSI Mean Reversion: Is it overbought (>70) or oversold (<30)?
    3. Momentum: Is Price > 20-Day Moving Average?
    4. Sentiment Shock: Adjust drift based on News & VIX.
    """
    # 1. Historical Drift (Log Returns Mean)
    # Using last 30 days to capture recent regime
    daily_returns = hist_df['Close'].pct_change().tail(30)
    base_drift = daily_returns.mean() # Daily Expected Return
    
    # 2. RSI Factor (Mean Reversion)
    current_rsi = hist_df['RSI'].iloc[-1]
    rsi_signal = 0
    if current_rsi > 70: rsi_signal = -0.002 # Overbought -> Downward pressure
    elif current_rsi < 30: rsi_signal = 0.003 # Oversold -> Upward pressure
    
    # 3. Momentum Factor (Trend)
    current_price = hist_df['Close'].iloc[-1]
    sma_20 = hist_df['SMA20'].iloc[-1]
    momentum_signal = 0.001 if current_price > sma_20 else -0.001
    
    # 4. Sentiment & Volatility Shock
    # High VIX reduces confidence (dampens positive drift)
    vix_factor = 1.0
    if volatility > 20: vix_factor = 0.5 
    
    sentiment_impact = (news_score * 0.005) * vix_factor
    
    # COMPOSITE FORMULA
    # Predicted Daily Move = (Base Drift + RSI Correction + Momentum + News)
    predicted_daily_move = base_drift + rsi_signal + momentum_signal + sentiment_impact
    
    # Explanation Text for UI
    logic_text = f"""
    **Logic Breakdown:**
    • Base Drift: {base_drift*100:.3f}% (Hist. Return)
    • RSI ({current_rsi:.1f}): {'Bearish (Overbought)' if current_rsi > 70 else 'Bullish (Oversold)' if current_rsi < 30 else 'Neutral'}
    • Trend: {'Bullish (>SMA20)' if current_price > sma_20 else 'Bearish (<SMA20)'}
    • Sentiment: {sentiment_impact*100:.3f}% Impact
    """
    
    return predicted_daily_move, logic_text

# --- 4. DATA ENGINE ---

def get_market_status():
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    if now.weekday() < 5 and (time(9,15) <= now.time() <= time(15,30)):
        return True, "🟢 LIVE", 60 
    return False, "🔴 CLOSED", 300

@st.cache_data(ttl=600)
def fetch_main_data(ticker):
    stock = yf.Ticker(ticker)
    try:
        # Fetch Max History for robust stats
        hist_max = stock.history(period="2y", interval="1d")
        hist_intra = stock.history(period="1d", interval="5m")
        
        if hist_max.empty: return pd.DataFrame(), pd.DataFrame(), {}
        
        # Calculate Technicals (Pandas Vectorized)
        hist_max['SMA20'] = hist_max['Close'].rolling(window=20).mean()
        hist_max['RSI'] = calculate_rsi(hist_max['Close'])
        
        # Extract Safe Metrics
        metrics = {
            "price": float(hist_max['Close'].iloc[-1]),
            "prev": float(hist_max['Close'].iloc[-2]),
            "open": float(hist_max['Open'].iloc[-1]),
            "high": float(hist_max['High'].iloc[-1]),
            "low": float(hist_max['Low'].iloc[-1]),
            "volatility": float(hist_max['Close'].pct_change().std() * 100)
        }
        return hist_max, hist_intra, metrics
        
    except Exception: return pd.DataFrame(), pd.DataFrame(), {}

@st.cache_data(ttl=3600)
def fetch_chunked_movers(const_tickers):
    # Batch downloader
    chunk_size = 10
    chunks = [const_tickers[i:i + chunk_size] for i in range(0, len(const_tickers), chunk_size)]
    all_data = []
    
    progress_bar = st.progress(0, text="Analyzing Market...")
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
                        "Change %": ((latest - prev) / prev) * 100
                    })
                except: continue
            time_module.sleep(0.5)
            progress_bar.progress((i + 1) / len(chunks))
        except: continue
    progress_bar.empty()
    return pd.DataFrame(all_data)

# --- 5. APP EXECUTION ---
if 'last_run' not in st.session_state: st.session_state['last_run'] = 0
is_open, status_msg, refresh_rate = get_market_status()

selected_index = st.sidebar.selectbox("Select Index", list(INDICES.keys()))
ticker = INDICES[selected_index]

# Fetch
hist_max, hist_intra, metrics = fetch_main_data(ticker)

if hist_max.empty:
    st.error("⚠️ Market Data Connection Failed. Retrying...")
    st.stop()

# Sentiment
sia = SentimentIntensityAnalyzer()
articles = []
for feed in NEWS_FEEDS:
    try:
        parsed = feedparser.parse(feed)
        for entry in parsed.entries[:3]: articles.append(entry.title)
    except: continue
news_score = np.mean([sia.polarity_scores(a)['compound'] for a in articles]) if articles else 0

# RUN THE LOGICAL PREDICTION MODEL
pred_move_pct, logic_explanation = get_smart_prediction(hist_max, news_score, metrics['volatility'])
pred_price = metrics['price'] * (1 + pred_move_pct)

# --- UI LAYOUT ---
st.title(f"{selected_index} Command Center")
st.caption(f"Status: {status_msg}")

# Heads Up
m = st.columns(5)
m[0].metric("Price", f"₹{metrics['price']:,.2f}", f"{((metrics['price']-metrics['prev'])/metrics['prev'])*100:.2f}%")
m[1].metric("Open", f"₹{metrics['open']:,.2f}")
m[2].metric("High", f"₹{metrics['high']:,.2f}")
m[3].metric("Low", f"₹{metrics['low']:,.2f}")
m[4].metric("AI Target (1D)", f"₹{pred_price:,.2f}", f"{pred_move_pct*100:.2f}% exp.", delta_color="normal")

st.markdown("---")

# MAIN CHART
g1, g2 = st.columns([3, 1])

with g1:
    st.subheader("📈 Analysis")
    t1, t2 = st.tabs(["Intraday", "Historical (Max)"])
    
    with t1:
        if not hist_intra.empty:
            fig = go.Figure(go.Candlestick(x=hist_intra.index, open=hist_intra['Open'], high=hist_intra['High'], low=hist_intra['Low'], close=hist_intra['Close'], name='Price'))
            # Visualizing the Prediction
            last_time = hist_intra.index[-1]
            fig.add_trace(go.Scatter(
                x=[last_time, last_time + timedelta(hours=2)],
                y=[metrics['price'], pred_price],
                mode='lines+markers', name='Target Path',
                line=dict(color='#FFA500', dash='dot', width=2)
            ))
            fig.update_layout(template="plotly_dark", height=450, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
        else: st.info("Intraday unavailable")
        
    with t2:
        # Forecast 5 Days based on calculated drift
        dates = [hist_max.index[-1] + timedelta(days=i) for i in range(1, 6)]
        prices = [metrics['price'] * (1 + (pred_move_pct * i)) for i in range(1, 6)]
        
        fig_h = go.Figure()
        fig_h.add_trace(go.Scatter(x=hist_max.index, y=hist_max['Close'], line=dict(color='#00F0FF'), name='History'))
        fig_h.add_trace(go.Scatter(x=dates, y=prices, mode='lines+markers', line=dict(color='#FFA500', dash='dot'), name='AI Projection'))
        
        fig_h.update_xaxes(rangeselector=dict(buttons=list([
            dict(count=1, label="1M", step="month", stepmode="backward"),
            dict(count=6, label="6M", step="month", stepmode="backward"),
            dict(count=1, label="YTD", step="year", stepmode="todate"),
            dict(count=1, label="1Y", step="year", stepmode="backward"),
            dict(step="all", label="MAX")
        ]), bgcolor="#262730"))
        fig_h.update_layout(template="plotly_dark", height=450)
        st.plotly_chart(fig_h, use_container_width=True)

with g2:
    st.subheader("🧠 Model Logic")
    st.info(logic_explanation)
    st.metric("Volatility (Risk)", f"{metrics['volatility']:.2f}%")
    st.caption("Recent Headlines:")
    for h in articles: st.text(f"• {h[:40]}...")

st.markdown("---")

# MOVERS
st.subheader(f"🏗️ {selected_index} Components")
if st.button("🚀 Scan Market Movers"):
    df_movers = fetch_chunked_movers(CONSTITUENTS[selected_index])
    if not df_movers.empty:
        t_l, t_g, t_lo = st.tabs(["List", "Gainers", "Losers"])
        with t_l: st.dataframe(df_movers, use_container_width=True)
        with t_g: st.dataframe(df_movers.sort_values("Change %", ascending=False).head(10), use_container_width=True)
        with t_lo: st.dataframe(df_movers.sort_values("Change %", ascending=True).head(10), use_container_width=True)

time_module.sleep(refresh_rate)
st.rerun()
