import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import statsmodels.api as sm
import numpy as np

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Global Monetary Policy Dashboard",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Global Monetary Policy & Stock Market Dashboard")
st.markdown("""
This dashboard analyzes the dual impact of **US Federal Reserve** and **Reserve Bank of India (RBI)** policy shifts on Indian Sectoral Indices.
It uses **Multivariate Regression** to isolate the sensitivity of stocks to domestic vs. international rate shocks.
""")

# ==========================================
# 2. DATA FETCHING & PREPARATION
# ==========================================
@st.cache_data(ttl=3600)
def get_combined_data():
    # --- A. Fetch Market Data from Yahoo Finance ---
    tickers = {
        '^NSEI': 'Nifty 50',
        '^NSEBANK': 'Nifty Bank',
        '^CNXIT': 'Nifty IT',
        '^CNXAUTO': 'Nifty Auto',
        '^CNXREALTY': 'Nifty Realty',
        '^GSPC': 'S&P 500',
        '^TNX': 'US 10Y Bond Yield',  # Proxy for US Rates
        'INR=X': 'USD/INR'
    }
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=15*365) # 15 Years of data
    
    try:
        # Download data
        raw_data = yf.download(list(tickers.keys()), start=start_date, end=end_date, auto_adjust=False)
        
        # Robust column selection
        if 'Adj Close' in raw_data.columns:
            df = raw_data['Adj Close']
        elif 'Close' in raw_data.columns:
            df = raw_data['Close']
        else:
            return pd.DataFrame()

        df.rename(columns=tickers, inplace=True)
        df = df.fillna(method='ffill')
    except Exception as e:
        st.error(f"Error fetching Yahoo data: {e}")
        return pd.DataFrame()

    # --- B. Construct Historical RBI Repo Rate (Manual Data) ---
    # Since RBI rates aren't on Yahoo, we map the official change dates.
    rbi_history = [
        ('2010-01-01', 4.75), ('2010-03-19', 5.00), ('2010-04-20', 5.25), ('2010-07-02', 5.50),
        ('2010-07-27', 5.75), ('2010-09-16', 6.00), ('2010-11-02', 6.25), ('2011-01-25', 6.50),
        ('2011-03-17', 6.75), ('2011-05-03', 7.25), ('2011-06-16', 7.50), ('2011-07-26', 8.00),
        ('2011-10-25', 8.50), ('2012-04-17', 8.00), ('2012-06-18', 8.00), ('2013-01-29', 7.75),
        ('2013-03-19', 7.50), ('2013-05-03', 7.25), ('2013-07-15', 10.25), ('2013-09-20', 7.50),
        ('2013-10-29', 7.75), ('2014-01-28', 8.00), ('2015-01-15', 7.75), ('2015-03-04', 7.50),
        ('2015-06-02', 7.25), ('2015-09-29', 6.75), ('2016-04-05', 6.50), ('2016-10-04', 6.25),
        ('2017-08-02', 6.00), ('2018-06-06', 6.25), ('2018-08-01', 6.50), ('2019-02-07', 6.25),
        ('2019-04-04', 6.00), ('2019-06-06', 5.75), ('2019-08-07', 5.40), ('2019-10-04', 5.15),
        ('2020-03-27', 4.40), ('2020-05-22', 4.00), ('2022-05-04', 4.40), ('2022-06-08', 4.90),
        ('2022-08-05', 5.40), ('2022-09-30', 5.90), ('2022-12-07', 6.25), ('2023-02-08', 6.50)
    ]
    
    # Create daily series for RBI rate
    rbi_series = pd.Series(np.nan, index=df.index)
    for date_str, rate in rbi_history:
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        if date_obj in rbi_series.index:
            rbi_series.loc[date_obj] = rate
    
    # Forward fill the rate (it stays constant until changed)
    df['RBI Repo Rate'] = rbi_series.ffill()
    
    # Fill any remaining NaNs (start of data) with first available rate
    df['RBI Repo Rate'] = df['RBI Repo Rate'].bfill()
    
    return df

# Load Data
try:
    with st.spinner('Calculating multivariate regression models...'):
        df = get_combined_data()
        
    if df is None or df.empty:
        st.error("Data load failed. Please refresh.")
        st.stop()
        
    st.success(f"Models calibrated with data up to: {df.index[-1].strftime('%d-%b-%Y')}")

except Exception as e:
    st.error(f"Critical Error: {e}")
    st.stop()

# ==========================================
# 3. SIDEBAR
# ==========================================
st.sidebar.header("Analysis Settings")
time_frame = st.sidebar.selectbox("Analysis Window", ["1 Year", "5 Years", "10 Years", "Full History"])

days_map = {"1 Year": 365, "5 Years": 365*5, "10 Years": 365*10, "Full History": 10000}
df_analysis = df.tail(days_map[time_frame]).copy()

# ==========================================
# 4. TABS
# ==========================================
tab1, tab2, tab3 = st.tabs(["🔮 Dual-Rate Simulator", "📈 Historical Impact Graph", "📉 Correlation Check"])

# --- TAB 1: DUAL-RATE SIMULATOR ---
with tab1:
    st.subheader("Simulate Interest Rate Changes")
    st.markdown("Adjust the sliders below to see how a **Simultaneous Shock** from the Fed and RBI impacts Indian sectors.")
    
    col_input1, col_input2, col_input3 = st.columns([1,1,2])
    
    with col_input1:
        st.markdown("🇺🇸 **US Fed / Bond Yield**")
        us_shock = st.number_input("Change in US Rates (%)", min_value=-2.0, max_value=2.0, value=0.25, step=0.05)

    with col_input2:
        st.markdown("🇮🇳 **RBI Repo Rate**")
        rbi_shock = st.number_input("Change in RBI Rate (%)", min_value=-2.0, max_value=2.0, value=0.00, step=0.05)

    with col_input3:
        target_sector = st.selectbox("Select Target Sector", ['Nifty Bank', 'Nifty IT', 'Nifty Realty', 'Nifty Auto', 'Nifty 50'])

    # --- MULTIVARIATE REGRESSION ---
    # We calculate calculating % returns for regression
    returns = df_analysis.pct_change().dropna()
    
    # Define Variables
    Y = returns[target_sector]
    X = returns[['US 10Y Bond Yield', 'RBI Repo Rate']] # Two Independent Variables
    X = sm.add_constant(X)
    
    # Fit Model
    model = sm.OLS(Y, X).fit()
    
    beta_us = model.params['US 10Y Bond Yield']
    beta_rbi = model.params['RBI Repo Rate']
    r_sq = model.rsquared

    # Calculate Predicted Impact
    # Note: We scale beta slightly for intuitive "shock" visualization as daily beta is very small
    impact_us = beta_us * us_shock
    impact_rbi = beta_rbi * rbi_shock
    total_impact = impact_us + impact_rbi
    
    st.markdown("---")
    
    # Results Display
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        st.markdown(f"### 📊 Net Predicted Impact: :{'red' if total_impact < 0 else 'green'}[{total_impact:.2f}%]")
        
        # Gauge Chart for visual impact
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = total_impact,
            title = {'text': f"Impact on {target_sector}"},
            delta = {'reference': 0},
            gauge = {
                'axis': {'range': [-5, 5]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [-5, 0], 'color': "#ffcccc"},
                    {'range': [0, 5], 'color': "#ccffcc"}],
            }
        ))
        fig_gauge.update_layout(height=250, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with res_col2:
        st.markdown("#### Sensitivity Breakdown (Beta)")
        st.info(f"""
        **1. Sensitivity to US Rates:** `{beta_us:.3f}`
        *(If US rates go up 1%, this sector moves by {beta_us:.2f}%)*
        
        **2. Sensitivity to RBI Rates:** `{beta_rbi:.3f}`
        *(If RBI rates go up 1%, this sector moves by {beta_rbi:.2f}%)*
        
        **Model Accuracy (R²):** `{r_sq:.3f}`
        """)
        
        if abs(beta_us) > abs(beta_rbi):
            st.warning(f"💡 Insight: {target_sector} is currently more sensitive to **US Factors** than Domestic Rates.")
        else:
            st.success(f"💡 Insight: {target_sector} is dominated by **Domestic (RBI)** Policy.")

# --- TAB 2: HISTORICAL GRAPH ---
with tab2:
    st.subheader("Historical Rate Cycles vs. Market")
    
    # Normalize data for comparison
    norm_df = df_analysis.copy()
    
    # Scale Interest Rates to match Index levels roughly for visualization (Dual Axis is better)
    
    fig = go.Figure()

    # Market Index (Left Axis)
    fig.add_trace(go.Scatter(
        x=norm_df.index, y=norm_df[target_sector],
        name=f"{target_sector} Price", line=dict(color='blue', width=2)
    ))

    # RBI Rate (Right Axis)
    fig.add_trace(go.Scatter(
        x=norm_df.index, y=norm_df['RBI Repo Rate'],
        name="RBI Repo Rate", line=dict(color='orange', width=2, dash='solid'),
        yaxis='y2'
    ))

    # US Yield (Right Axis)
    fig.add_trace(go.Scatter(
        x=norm_df.index, y=norm_df['US 10Y Bond Yield'],
        name="US 10Y Yield", line=dict(color='red', width=2, dash='dot'),
        yaxis='y2'
    ))

    fig.update_layout(
        title=f"{target_sector} vs. Policy Rates",
        yaxis=dict(title="Index Value"),
        yaxis2=dict(title="Interest Rate (%)", overlaying='y', side='right'),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.1)
    )
    
    st.plotly_chart(fig, use_container_width=True)

# --- TAB 3: CORRELATION ---
with tab3:
    st.subheader("Correlation Matrix")
    
    # Show correlations
    cols_to_corr = ['RBI Repo Rate', 'US 10Y Bond Yield', 'USD/INR', 'Nifty 50', 'Nifty Bank', 'Nifty IT']
    corr = returns[cols_to_corr].corr()
    
    fig_corr = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    st.plotly_chart(fig_corr, use_container_width=True)
