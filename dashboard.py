import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import time
from datetime import datetime
import pytz

# --- PAGE CONFIGURATION (Must be first) ---
st.set_page_config(
    page_title="Market Command Center [Live]",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR "STEALTH" UPDATES & STYLING ---
st.markdown("""
    <style>
    .metric-card {
        background-color: #1e1e1e;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #333;
        text-align: center;
    }
    .stProgress > div > div > div > div {
        background-color: #00cc00;
    }
    /* Hide Streamlit default menu for cleaner look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- 1. DEFINING THE UNIVERSE (INDICES & CONSTITUENTS) ---
# NOTE: Real-world indices have 30-50 stocks. Hardcoding the heavyweights here for performance.
# In a production app, you would fetch these lists dynamically from a CSV or API.

INDICES = {
    "NIFTY 50": "^NSEI",
    "NIFTY BANK": "^NSEBANK",
    "NIFTY IT": "^CNXIT",
    "SENSEX": "^BSESN"
}

# Top Heavyweights for each index (These drive 80% of the movement)
# We use these to calculate "Live Weightage"
CONSTITUENTS = {
    "NIFTY 50": [
        "RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFOSYS.NS", "ITC.NS", 
        "TCS.NS", "L&T.NS", "AXISBANK.NS", "KOTAKBANK.NS", "HINDUNILVR.NS",
        "SBIN.NS", "BHARTIARTL.NS", "BAJFINANCE.NS", "ASIANPAINT.NS", "MARUTI.NS"
    ],
    "NIFTY BANK": [
        "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "AXISBANK.NS", "KOTAKBANK.NS",
        "INDUSINDBK.NS", "BANKBARODA.NS", "PNB.NS", "IDFCFIRSTB.NS", "AUBANK.NS"
    ],
    "NIFTY IT": [
        "TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS",
        "LTIM.NS", "PERSISTENT.NS", "COFORGE.NS", "MPHASIS.NS", "OFSS.NS"
    ],
    "SENSEX": [
        "RELIANCE.BO", "HDFCBANK.BO", "ICICIBANK.BO", "INFY.BO", "ITC.BO",
        "TCS.BO", "L&T.BO", "AXISBANK.BO", "KOTAKBANK.BO", "HINDUNILVR.BO",
        "SBIN.BO", "BHARTIARTL.BO", "BAJFINANCE.BO", "TATASTEEL.BO", "M&M.BO"
    ]
}

# --- 2. GLOBAL MACRO FACTORS (For the Prediction Model) ---
MACRO_TICKERS = {
    "US 10Y Yield": "^TNX",
    "Brent Crude Oil": "BZ=F",
    "Gold Futures": "GC=F",
    "USD/INR": "INR=X",
    "S&P 500 (US Market)": "^GSPC",
    "India VIX": "^INDIAVIX"
}

# --- HELPER FUNCTIONS ---

@st.cache_data(ttl=300)  # Cache historical data for 5 mins
def get_historical_data(tickers):
    """Fetches 6 months of daily data for training the model"""
    try:
        data = yf.download(tickers, period="6mo", interval="1d", progress=False)['Close']
        return data.fillna(method='ffill').fillna(method='bfill')
    except Exception as e:
        return pd.DataFrame()

def get_live_data(tickers):
    """Fetches real-time snapshot (1-minute latency)"""
    try:
        # Fetching strictly the last valid price
        data = yf.download(tickers, period="1d", interval="1m", progress=False)['Close'].iloc[-1]
        return data
    except Exception:
        # Fallback to daily if minute data fails
        return yf.download(tickers, period="1d", progress=False)['Close'].iloc[-1]

def run_prediction_model(target_symbol, macro_df):
    """
    Runs a Multivariate Regression on the fly.
    Target: Index Price (e.g., Nifty 50)
    Features: Oil, Gold, USD/INR, US Yields
    """
    # Align data
    df = macro_df.copy()
    if target_symbol not in df.columns:
        return 0, 0, {} # Error handling
        
    # Features (X) and Target (y)
    features = ["US 10Y Yield", "Brent Crude Oil", "USD/INR", "S&P 500 (US Market)"]
    # Ensure these cols exist
    available_features = [f for f in features if f in df.columns]
    
    X = df[available_features].pct_change().dropna()
    y = df[target_symbol].pct_change().dropna()
    
    # Align indices
    common_idx = X.index.intersection(y.index)
    X = X.loc[common_idx]
    y = y.loc[common_idx]
    
    if len(X) < 10: return 0, 0, {} # Not enough data

    # Train Model
    model = LinearRegression()
    model.fit(X, y)
    
    # Make Prediction (based on latest macro values)
    latest_macro = df[available_features].iloc[-1].values.reshape(1, -1)
    # This is a 'return' prediction, so we convert to price later
    pred_return = model.predict(X.iloc[-1].values.reshape(1, -1))[0] 
    
    # Extract Coefficients (Contribution)
    contributions = dict(zip(available_features, model.coef_))
    
    return pred_return, model.score(X, y) * 100, contributions

# --- MAIN LAYOUT ---

# Sidebar: Index Selection
st.sidebar.title("📡 Market Radar")
selected_index = st.sidebar.selectbox("Select Target Index", list(INDICES.keys()))
selected_ticker = INDICES[selected_index]
constituents_list = CONSTITUENTS[selected_index]

# Sidebar: Live Status
st.sidebar.divider()
status_placeholder = st.sidebar.empty()

# --- MAIN DASHBOARD CONTAINER ---
# We use st.empty() containers for parts that need to refresh "unnoticed"

# 1. Header & Live Price
header_col1, header_col2, header_col3 = st.columns([2, 1, 1])
with header_col1:
    st.title(f"📊 {selected_index} Analytics")
with header_col2:
    live_price_placeholder = st.empty()
with header_col3:
    prediction_placeholder = st.empty()

st.divider()

# 2. The "Smart" Prediction Section
st.subheader("🤖 AI-Driven Market Forecast")
col_pred1, col_pred2 = st.columns([3, 2])

with col_pred1:
    chart_placeholder = st.empty()

with col_pred2:
    st.write("###### Market Drivers (Multivariate Regression)")
    factors_placeholder = st.empty()

st.divider()

# 3. Constituents & Weightage (The "X-Ray" View)
st.subheader(f"🏗️ {selected_index} Composition & Live Weights")
weightage_placeholder = st.empty()

# --- THE STEALTH LOOP (Background Process) ---
# This loop runs forever inside the script once loaded.
# It updates the 'placeholders' defined above.

if "run_loop" not in st.session_state:
    st.session_state.run_loop = True

def update_dashboard():
    # 1. FETCH ALL DATA (Batch Request for Speed)
    all_tickers = list(INDICES.values()) + list(MACRO_TICKERS.values()) + constituents_list
    
    # Get History for Model
    hist_data = get_historical_data(all_tickers)
    
    # Get Live Snapshot
    live_prices = get_live_data(all_tickers)
    
    # --- RENDER 1: LIVE HEADER PRICE ---
    current_val = live_prices[selected_ticker]
    prev_close = hist_data[selected_ticker].iloc[-2]
    change = current_val - prev_close
    pct_change = (change / prev_close) * 100
    
    live_price_placeholder.metric(
        label="Live Index Level",
        value=f"{current_val:,.2f}",
        delta=f"{change:+.2f} ({pct_change:+.2f}%)"
    )
    
    # --- RENDER 2: PREDICTION MODEL ---
    # Rename columns in hist_data to match readable names
    inv_macro_map = {v: k for k, v in MACRO_TICKERS.items()}
    model_data = hist_data.rename(columns=inv_macro_map)
    
    pred_return, accuracy, contributions = run_prediction_model(selected_ticker, model_data)
    
    # Calculate Predicted Price
    predicted_price = current_val * (1 + pred_return)
    direction = "BULLISH 🟢" if pred_return > 0 else "BEARISH 🔴"
    
    prediction_placeholder.metric(
        label="Next Session Forecast",
        value=f"{predicted_price:,.2f}",
        delta=direction,
        delta_color="off"
    )
    
    # Factor Importance Bar Chart
    if contributions:
        coeffs = pd.Series(contributions).sort_values()
        fig_factors = px.bar(
            x=coeffs.values, 
            y=coeffs.index, 
            orientation='h',
            labels={'x': 'Impact Strength', 'y': 'Macro Factor'},
            title="What is moving the market NOW?",
            color=coeffs.values,
            color_continuous_scale="RdBu"
        )
        fig_factors.update_layout(height=300, showlegend=False)
        factors_placeholder.plotly_chart(fig_factors, use_container_width=True)

    # Main Chart (Live vs Predicted Trend) - Simplified for demo
    # In a real app, this would append live points.
    chart_data = hist_data[selected_ticker].tail(50)
    fig_main = px.line(chart_data, title=f"Real-Time Trend: {selected_index}")
    chart_placeholder.plotly_chart(fig_main, use_container_width=True)

    # --- RENDER 3: WEIGHTAGE CALCULATION ---
    # Logic: Market Cap = Price * Shares. 
    # Since we don't have share count in free API, we use Price * Approx Weight Factor or just Price Relative Strength
    # BETTER PROXY: We assume the 'Price' change reflects the weight impact for this demo.
    # To do real Market Cap weights, we'd need a paid API (e.g., Bloomberg).
    # Here, we show 'Performance Contribution' which is effectively what the user cares about.
    
    const_prices = live_prices[constituents_list]
    const_prev = hist_data[constituents_list].iloc[-2]
    const_changes = ((const_prices - const_prev) / const_prev) * 100
    
    # Create DataFrame
    df_weights = pd.DataFrame({
        "Company": constituents_list,
        "Price": const_prices.values,
        "Change (%)": const_changes.values
    }).sort_values(by="Change (%)", ascending=False)
    
    # Visualizing as a TreeMap (Green=Up, Red=Down)
    fig_tree = px.treemap(
        df_weights,
        path=['Company'],
        values=abs(df_weights['Change (%)']), # Size = Volatility
        color='Change (%)',
        color_continuous_scale='RdGy', # Red to Green
        title=f"{selected_index} Heatmap: Size = Volatility, Color = Trend"
    )
    weightage_placeholder.plotly_chart(fig_tree, use_container_width=True)
    
    # Sidebar Clock
    tz = pytz.timezone('Asia/Kolkata')
    curr_time = datetime.now(tz).strftime("%H:%M:%S")
    status_placeholder.success(f"Last Update: {curr_time} IST")

# --- EXECUTION LOOP ---
# Check if we are in a 'live' context.
# We create a button to start the loop so it doesn't freeze on initial load.
if st.button('🚀 ACTIVATE LIVE FEED'):
    with st.spinner("Connecting to Global Exchanges..."):
        while True:
            update_dashboard()
            time.sleep(60) # Wait 60 seconds
else:
    # Run once on load
    update_dashboard()
    st.info("Click 'ACTIVATE LIVE FEED' to enable auto-refresh (60s interval).")
