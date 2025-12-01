import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import statsmodels.api as sm

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Monetary Policy Impact Dashboard",
    page_icon="📈",
    layout="wide"
)

# Title and Description
st.title("📊 Global Monetary Policy & Stock Market Dashboard")
st.markdown("""
This live dashboard tracks the relationship between **Interest Rate Expectations (Bond Yields)** and **Stock Market Performance** in India and the USA.
It serves as a dynamic extension of the research: *"The Impact of Interest Rate Changes on Stock Market Performance: Evidence from India and the USA"*.

*Data Source: Yahoo Finance (Live Daily Updates)*
""")

# ==========================================
# 2. DATA FETCHING FUNCTION
# ==========================================
@st.cache_data(ttl=3600)  # Cache data for 1 hour
def get_data():
    # Tickers mapping
    # Note: We use US 10Y Bond Yield (^TNX) as a high-frequency proxy for interest rate expectations
    tickers = {
        '^NSEI': 'Nifty 50',
        '^NSEBANK': 'Nifty Bank',
        '^CNXIT': 'Nifty IT',
        '^CNXAUTO': 'Nifty Auto',
        '^CNXREALTY': 'Nifty Realty',
        '^GSPC': 'S&P 500',
        '^IXIC': 'Nasdaq',
        '^TNX': 'US 10Y Bond Yield', 
        'INR=X': 'USD/INR'
    }
    
    # Fetch data for the last 5 years
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)
    
    # Download data with error handling
    try:
        # auto_adjust=False ensures we get the raw columns we expect
        raw_data = yf.download(list(tickers.keys()), start=start_date, end=end_date, auto_adjust=False)
        
        # Robust column selection (Handle cases where Adj Close might be missing)
        if 'Adj Close' in raw_data.columns:
            data = raw_data['Adj Close']
        elif 'Close' in raw_data.columns:
            data = raw_data['Close']
        else:
            # If structure is different, try to return as is or return empty
            return pd.DataFrame()

        # Rename columns to friendly names
        data.rename(columns=tickers, inplace=True)
        
        # Forward fill missing data (for holidays)
        data = data.fillna(method='ffill')
        
        # Drop rows where all data is still NaN (start of dataframe)
        data = data.dropna(how='all')
        
        return data

    except Exception as e:
        st.error(f"Failed to download data: {e}")
        return pd.DataFrame()

# Load Data
try:
    with st.spinner('Fetching live market data...'):
        df = get_data()
    
    # --- CRITICAL FIX: Check if data is empty before accessing it ---
    if df is None or df.empty:
        st.error("⚠️ No data could be retrieved. This may be due to a connection issue with Yahoo Finance. Please refresh the page in a few moments.")
        st.stop()
        
    st.success(f"Data updated successfully! Last date: {df.index[-1].strftime('%Y-%m-%d')}")

except Exception as e:
    st.error(f"Error fetching data: {e}")
    st.stop()

# ==========================================
# 3. SIDEBAR CONTROLS
# ==========================================
st.sidebar.header("Dashboard Settings")
time_frame = st.sidebar.selectbox("Select Time Frame", ["1 Year", "3 Years", "5 Years"])

# Filter DataFrame based on selection
days_map = {"1 Year": 365, "3 Years": 365*3, "5 Years": 365*5}
df_filtered = df.tail(days_map[time_frame])

# ==========================================
# 4. KEY METRICS (HEADLINES)
# ==========================================
st.subheader("Market Snapshot (Latest Close)")
col1, col2, col3, col4 = st.columns(4)

def get_change(series):
    # Safe check for empty series
    clean_series = series.dropna()
    if len(clean_series) < 2:
        return 0, 0
        
    # Get last valid value
    latest = clean_series.iloc[-1]
    prev = clean_series.iloc[-2]
    change = ((latest - prev) / prev) * 100
    return latest, change

nifty_val, nifty_chg = get_change(df['Nifty 50'])
us_yield_val, us_yield_chg = get_change(df['US 10Y Bond Yield'])
usd_val, usd_chg = get_change(df['USD/INR'])
bank_val, bank_chg = get_change(df['Nifty Bank'])

col1.metric("Nifty 50", f"{nifty_val:,.0f}", f"{nifty_chg:.2f}%")
col2.metric("US 10Y Bond Yield", f"{us_yield_val:.2f}%", f"{us_yield_chg:.2f}%")
col3.metric("USD/INR", f"{usd_val:.2f}", f"{usd_chg:.2f}%")
col4.metric("Nifty Bank", f"{bank_val:,.0f}", f"{bank_chg:.2f}%")

# ==========================================
# 5. TABS FOR ANALYSIS
# ==========================================
tab1, tab2, tab3 = st.tabs(["📈 Trend Analysis", "🔗 Correlation Matrix", "🔮 Rate Impact Simulator"])

# --- TAB 1: TRENDS ---
with tab1:
    st.subheader("Interest Rates vs. Market Performance")
    
    # User selection
    sector_select = st.selectbox("Select Sector to Compare against US Yields:", 
                                 ['Nifty 50', 'Nifty Bank', 'Nifty IT', 'Nifty Auto', 'Nifty Realty'])
    
    # Dual Axis Chart
    fig = go.Figure()

    # Trace 1: Stock Index
    fig.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered[sector_select],
                             name=sector_select, line=dict(color='blue')))

    # Trace 2: US Yields (Secondary Axis)
    fig.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered['US 10Y Bond Yield'],
                             name="US 10Y Bond Yield", line=dict(color='red', dash='dot'), yaxis='y2'))

    # Layout
    fig.update_layout(
        title=f"{sector_select} vs. US Interest Rate Expectations",
        yaxis=dict(title=f"{sector_select} Price"),
        yaxis2=dict(title="US 10Y Yield (%)", overlaying='y', side='right'),
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.info("Observation: Look for periods where the Red Line (Rates) goes UP and the Blue Line (Stocks) goes DOWN. This visualizes the inverse relationship.")

# --- TAB 2: CORRELATIONS ---
with tab2:
    st.subheader("Live Correlation Matrix")
    st.markdown("This heatmap shows how strongly different assets move together. **Red** = Inverse Relationship (Rates up, Stocks down).")
    
    # Calculate daily returns for correlation
    returns_df = df_filtered.pct_change().dropna()
    
    # Filter only the columns we want in the matrix
    target_cols = ['US 10Y Bond Yield', 'USD/INR', 'Nifty 50', 'Nifty Bank', 'Nifty IT', 'S&P 500']
    # Ensure columns exist before correlation
    available_cols = [c for c in target_cols if c in returns_df.columns]
    
    corr_matrix = returns_df[available_cols].corr()

    fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale="RdBu", zmin=-1, zmax=1)
    st.plotly_chart(fig_corr, use_container_width=True)

# --- TAB 3: SIMULATOR ---
with tab3:
    st.subheader("🔮 Rate Impact Simulator")
    st.markdown("""
    Use this tool to simulate future scenarios. We use **Simple Linear Regression (OLS)** on the filtered historical data 
    to calculate the 'Beta' (Sensitivity) of each sector to changes in US Interest Rates.
    """)
    
    col_sim1, col_sim2 = st.columns(2)
    
    # Prepare data for regression
    returns_df = df_filtered.pct_change().dropna()
    
    with col_sim1:
        # User Input
        rate_shock = st.slider("Simulate Change in Interest Rates (%):", min_value=-2.0, max_value=2.0, value=0.50, step=0.05,
                               help="A +0.50% change means yields go from e.g., 4.00% to 4.50%")
        target_sector = st.selectbox("Select Target Sector:", ['Nifty Bank', 'Nifty IT', 'Nifty Realty', 'Nifty Auto', 'Nifty 50'])

    # Calculate Sensitivity (Beta) live
    # Y = Stock Returns, X = Bond Yield Changes
    Y = returns_df[target_sector]
    X = returns_df['US 10Y Bond Yield']
    X = sm.add_constant(X)
    
    model = sm.OLS(Y, X).fit()
    beta = model.params['US 10Y Bond Yield']
    r_squared = model.rsquared
    
    # Prediction Calculation
    impact_prediction = beta * rate_shock
    
    with col_sim2:
        st.markdown(f"### Historical Sensitivity (Beta): **{beta:.3f}**")
        st.caption(f"Based on data from the last {time_frame}")
        
        st.markdown("---")
        st.markdown(f"#### Predicted Impact on {target_sector}")
        
        if impact_prediction < 0:
            st.error(f"📉 Estimated Drop: {impact_prediction:.2f}%")
        else:
            st.success(f"📈 Estimated Rise: {impact_prediction:.2f}%")
            
        st.write(f"*Model Confidence (R²): {r_squared:.3f}*")
            
    st.warning("""
    **Disclaimer:** This prediction is based purely on historical linear correlations. 
    It assumes that the future relationship will mimic the past. Real markets are influenced by many other factors.
    """)

# Footer
st.markdown("---")
st.markdown("Dashboard created by Nand Patel for Research Project (MM25GF025)")
