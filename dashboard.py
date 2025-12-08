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

st.markdown("""
    <style>
    .stApp { background-color: #0E1117; }
    
    /* Metrics Styling */
    div[data-testid="stMetric"] {
        background-color: #1A1C24;
        border: 1px solid #2D2D2D;
        padding: 15px;
        border-radius: 10px;
    }
    
    /* Chart Container */
    .stPlotlyChart {
        background-color: #1A1C24;
        border: 1px solid #2D2D2D;
        border-radius: 10px;
        padding: 15px;
    }
    
    /* Progress Bar Color */
    .stProgress > div > div > div > div {
        background-color: #FFA500;
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
        return True, "🟢 LIVE", 60 
    return False, "🔴 CLOSED", 300

@st.cache_data(ttl=600)
def fetch_main_data(ticker):
    stock = yf.Ticker(ticker)
    try:
        # Max history for robust volatility calculations
        hist_max = stock.history(period="2y", interval="1d")
        hist_intra = stock.history(period="1d", interval="5m")
        if hist_max.empty: return pd.DataFrame(), pd.DataFrame(), {}
        
        metrics = {
            "price": float(hist_max['Close'].iloc[-1]),
            "prev": float(hist_max['Close'].iloc[-2]),
            "open": float(hist_max['Open'].iloc[-1]),
            "high": float(hist_max['High'].iloc[-1]),
            "low": float(hist_max['Low'].iloc[-1]),
            "volatility": float(hist_max['Close'].pct_change().std()) # Daily Volatility
        }
        return hist_max, hist_intra, metrics
    except: return pd.DataFrame(), pd.DataFrame(), {}

@st.cache_data(ttl=3600)
def fetch_movers_batch(const_tickers):
    """Chunked fetching + Strict Deduplication"""
    chunk_size = 10
    chunks = [const_tickers[i:i + chunk_size] for i in range(0, len(const_tickers), chunk_size)]
    all_data = []
    seen = set()
    
    prog = st.progress(0, "Scanning Market Depth...")
    
    for i, chunk in enumerate(chunks):
        try:
            data = yf.download(chunk, period="2d", group_by='ticker', progress=False, threads=False)
            for t in chunk:
                if t in seen: continue
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
                    seen.add(t)
                except: continue
            time_module.sleep(0.2)
            prog.progress((i + 1) / len(chunks))
        except: continue
            
    prog.empty()
    df = pd.DataFrame(all_data)
    if not df.empty: df = df.drop_duplicates(subset=['Company'])
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

# --- 4. ADVANCED FINANCIAL MODELING ---

def calculate_technical_signals(df):
    """
    Computes Quant Technical Indicators:
    1. RSI (Momentum)
    2. MACD (Trend)
    3. EMA 20 (Support/Resistance)
    """
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # EMA 20
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    
    # Latest Signals
    latest = df.iloc[-1]
    
    score = 0
    reasons = []
    
    # RSI Logic
    if latest['RSI'] < 30: 
        score += 1; reasons.append("RSI Oversold (Bullish)")
    elif latest['RSI'] > 70: 
        score -= 1; reasons.append("RSI Overbought (Bearish)")
    
    # MACD Logic
    if latest['MACD'] > latest['Signal_Line']: 
        score += 1; reasons.append("MACD Bullish Crossover")
    else: 
        score -= 1; reasons.append("MACD Bearish Trend")
        
    # Trend Logic
    if latest['Close'] > latest['EMA20']:
        score += 0.5; reasons.append("Price > 20 EMA (Uptrend)")
    else:
        score -= 0.5; reasons.append("Price < 20 EMA (Downtrend)")
        
    return score, reasons

def monte_carlo_simulation(start_price, mu, sigma, days=5, simulations=500):
    """
    Performs Geometric Brownian Motion Simulation.
    S_t = S_0 * exp((mu - 0.5*sigma^2)t + sigma*W_t)
    """
    dt = 1 # daily steps
    simulation_results = np.zeros((simulations, days))
    
    for i in range(simulations):
        price_path = [start_price]
        for d in range(days):
            # Random Shock
            shock = np.random.normal(0, 1)
            # Drift + Shock
            drift = (mu - 0.5 * sigma**2) * dt
            diffusion = sigma * np.sqrt(dt) * shock
            price = price_path[-1] * np.exp(drift + diffusion)
            price_path.append(price)
        simulation_results[i, :] = price_path[1:]
        
    # Calculate Mean Path (Expected) and Confidence Intervals
    mean_path = np.mean(simulation_results, axis=0)
    upper_bound = np.percentile(simulation_results, 95, axis=0) # 95% Confidence
    lower_bound = np.percentile(simulation_results, 5, axis=0)  # 5% Confidence
    
    return mean_path, upper_bound, lower_bound

# --- 5. APP EXECUTION ---
if 'last_run' not in st.session_state: st.session_state['last_run'] = 0
is_open, status_msg, refresh_rate = get_market_status()

# Sidebar
selected_index = st.sidebar.selectbox("Select Index", list(INDICES.keys()))
ticker = INDICES[selected_index]

# Main Fetch
hist_max, hist_intra, metrics = fetch_main_data(ticker)

if hist_max.empty:
    st.error("⚠️ Market Data unavailable. Please refresh.")
    st.stop()

# --- RUNNING THE QUANT ENGINE ---

# 1. Technical Voting
tech_score, tech_reasons = calculate_technical_signals(hist_max)
sentiment_score, headlines = get_sentiment()

# 2. Monte Carlo Setup
# Calculate daily log returns mean (Drift) and std (Volatility)
log_returns = np.log(hist_max['Close'] / hist_max['Close'].shift(1)).dropna()
mu = log_returns.mean()
sigma = log_returns.std()

# 3. Run Simulation (5 Days)
mc_mean, mc_upper, mc_lower = monte_carlo_simulation(metrics['price'], mu, sigma, days=5)
pred_price_5d = mc_mean[-1]

# --- 6. VISUAL DASHBOARD ---

c1, c2 = st.columns([3, 1])
with c1:
    st.title(f"{selected_index}")
    st.caption(f"Real-Time Feed • {status_msg}")
with c2:
    if is_open: st.success(f"Status: {status_msg}")
    else: st.error(f"Status: {status_msg}")

# Heads Up
m = st.columns(5)
m[0].metric("Price", f"₹{metrics['price']:,.2f}", f"{((metrics['price']-metrics['prev'])/metrics['prev'])*100:.2f}%")
m[1].metric("Open", f"₹{metrics['open']:,.2f}")
m[2].metric("High", f"₹{metrics['high']:,.2f}")
m[3].metric("Low", f"₹{metrics['low']:,.2f}")
m[4].metric("Exp. Target (5D)", f"₹{pred_price_5d:,.2f}", f"Monte Carlo Mean", delta_color="normal")

st.markdown("---")

# GRAPH AREA
g_col, s_col = st.columns([3, 1])

with g_col:
    time_range = st.radio("Time Range", ["1D", "1M", "6M", "YTD", "1Y", "MAX"], horizontal=True, label_visibility="collapsed")
    fig = go.Figure()
    
    if time_range == "1D":
        if not hist_intra.empty:
            fig.add_trace(go.Scatter(x=hist_intra.index, y=hist_intra['Close'], mode='lines', name='Price', line=dict(color='#00F0FF', width=2)))
            fig.update_layout(title="Intraday Action")
        else: st.warning("Intraday data hidden (Market Closed). Switch to 1M.")
    else:
        df_plot = hist_max.copy()
        end_date = df_plot.index[-1]
        if time_range == "1M": start_date = end_date - timedelta(days=30)
        elif time_range == "6M": start_date = end_date - timedelta(days=180)
        elif time_range == "1Y": start_date = end_date - timedelta(days=365)
        elif time_range == "YTD": start_date = datetime(end_date.year, 1, 1).replace(tzinfo=end_date.tzinfo)
        else: start_date = df_plot.index[0]
        
        df_plot = df_plot[df_plot.index >= start_date]
        
        # Historical Line
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['Close'], mode='lines', name='History', line=dict(color='#00F0FF')))
        
        # MONTE CARLO FORECAST (The "Cone")
        future_dates = [df_plot.index[-1] + timedelta(days=i) for i in range(1, 6)]
        
        # Most Likely Path
        fig.add_trace(go.Scatter(x=future_dates, y=mc_mean, mode='lines+markers', name='Expected Path', line=dict(color='#FFA500', dash='dot', width=3)))
        
        # Risk Cone (Upper/Lower Bounds)
        fig.add_trace(go.Scatter(
            x=future_dates + future_dates[::-1],
            y=np.concatenate([mc_upper, mc_lower[::-1]]),
            fill='toself', fillcolor='rgba(255, 165, 0, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='95% Confidence Interval'
        ))
        
        fig.update_layout(title=f"{time_range} Trend + Monte Carlo Forecast")

    fig.update_layout(template="plotly_dark", height=500, xaxis_rangeslider_visible=False, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig, use_container_width=True)

with s_col:
    st.subheader("🧠 Quant Logic")
    with st.container(border=True):
        st.write("**Technical Vote:**")
        for r in tech_reasons:
            icon = "✅" if "Bullish" in r or ">" in r else "🔻"
            st.caption(f"{icon} {r}")
        
        st.write("---")
        st.write("**Sentiment:**")
        sent_label = "Bullish" if sentiment_score > 0.1 else "Bearish" if sentiment_score < -0.1 else "Neutral"
        st.info(f"News Sentiment: {sent_label} ({sentiment_score:.2f})")

st.markdown("---")

# MOVERS TAB
st.subheader("📊 Market Depth")
tab_movers, tab_info = st.tabs(["🚀 Top Movers", "ℹ️ Methodology"])

with tab_movers:
    if st.button("Load Movers Table"):
        df_movers = fetch_movers_batch(CONSTITUENTS[selected_index])
        if not df_movers.empty:
            c1, c2 = st.columns(2)
            
            # Use improved column config
            col_cfg = {
                "Price": st.column_config.NumberColumn(format="₹%.2f"),
                "Change %": st.column_config.ProgressColumn("Change %", format="%.2f%%", min_value=-5, max_value=5),
                "Volume": st.column_config.NumberColumn(format="%d")
            }
            
            with c1:
                st.write("### 🟢 Gainers")
                st.dataframe(df_movers.sort_values("Change %", ascending=False).head(10), column_config=col_cfg, use_container_width=True, hide_index=True)
            with c2:
                st.write("### 🔴 Losers")
                st.dataframe(df_movers.sort_values("Change %", ascending=True).head(10), column_config=col_cfg, use_container_width=True, hide_index=True)
        else:
            st.error("Data fetch failed. Try again.")

with tab_info:
    st.markdown("""
    ### Prediction Methodology
    1.  **Monte Carlo Simulation (5D):** Uses Geometric Brownian Motion with historical drift and volatility to simulate 500 future price paths. The orange line is the average of all paths.
    2.  **Technical Voting:** Aggregates signals from RSI (14), MACD (12,26,9), and EMA (20) to determine current momentum.
    3.  **Sentiment Analysis:** VADER NLP processes live RSS feeds to adjust short-term bias.
    """)

time_module.sleep(refresh_rate)
st.rerun()
