import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import statsmodels.api as sm
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

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
It features an **Automated News Sentiment Engine** that scans live financial news to adjust market predictions in real-time.
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
        raw_data = yf.download(list(tickers.keys()), start=start_date, end=end_date, auto_adjust=False)
        
        if 'Adj Close' in raw_data.columns:
            df = raw_data['Adj Close']
        elif 'Close' in raw_data.columns:
            df = raw_data['Close']
        else:
            return pd.DataFrame()

        df.rename(columns=tickers, inplace=True)
        # Handle weekends/holidays in market data first
        df = df.fillna(method='ffill')
        
    except Exception as e:
        st.error(f"Error fetching Yahoo data: {e}")
        return pd.DataFrame()

    # --- B. Construct Historical RBI Repo Rate (Manual Data) ---
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
    
    # Improved Logic: Create a DataFrame for Rates and Reindex it to match Market Data
    # This handles non-trading days (Sat/Sun) where a rate change might have occurred
    rbi_dates = [datetime.strptime(x[0], '%Y-%m-%d') for x in rbi_history]
    rbi_rates = [x[1] for x in rbi_history]
    
    rbi_df = pd.DataFrame({'Rate': rbi_rates}, index=rbi_dates)
    rbi_df = rbi_df.sort_index()
    
    # Reindex RBI data to match the Stock Market dataframe index
    # method='ffill' propagates the last known rate forward to all trading days
    aligned_rbi = rbi_df.reindex(df.index, method='ffill')
    
    # Assign to main dataframe
    df['RBI Repo Rate'] = aligned_rbi['Rate']
    
    # Fill any initial gaps (before 2010 if any) with the first available rate
    df['RBI Repo Rate'] = df['RBI Repo Rate'].bfill()
    
    return df

# --- C. LIVE NEWS SENTIMENT FUNCTION ---
@st.cache_data(ttl=1800) 
def get_live_sentiment(sector_name):
    sector_map = {
        'Nifty 50': '^NSEI',
        'Nifty Bank': '^NSEBANK', 
        'Nifty IT': 'INFY.NS',
        'Nifty Auto': 'TATAMOTORS.NS',
        'Nifty Realty': 'DLF.NS'
    }
    
    ticker_symbol = sector_map.get(sector_name, '^NSEI')
    try:
        stock = yf.Ticker(ticker_symbol)
        news = stock.news
        
        if not news:
            return 0, "No recent news found."
            
        sentiment_scores = []
        headlines = []
        analyzer = SentimentIntensityAnalyzer()
        
        for item in news:
            title = item.get('title', '')
            if title:
                score = analyzer.polarity_scores(title)['compound']
                sentiment_scores.append(score)
                headlines.append(title)
        
        if not sentiment_scores:
            return 0, "No readable text."
            
        avg_score = sum(sentiment_scores) / len(sentiment_scores)
        scaled_score = avg_score * 5 
        scaled_score = max(min(scaled_score, 5), -5)
        
        return scaled_score, headlines[:3]
        
    except Exception as e:
        return 0, f"Error fetching news: {str(e)}"

# Load Data
try:
    with st.spinner('Updating market data...'):
        df = get_combined_data()
        
    if df is None or df.empty:
        st.error("Data load failed. Please refresh.")
        st.stop()
        
    st.success(f"Data updated: {df.index[-1].strftime('%d-%b-%Y')}")

except Exception as e:
    st.error(f"Critical Error: {e}")
    st.stop()

# ==========================================
# 3. TABS
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs(["🔮 Smart Prediction", "📉 Historical Impact", "📊 Sensitivity Analysis", "🔗 Correlation"])

# --- TAB 1: SMART PREDICTION MODULE ---
with tab1:
    st.subheader("🤖 AI-Assisted Market Prediction (Next 5 Days)")
    st.markdown("Predict trends using Live Sentiment + Rate Shocks + Momentum.")
    
    col_pred1, col_pred2 = st.columns([1, 2])
    
    with col_pred1:
        st.markdown("### 1. Select Sector")
        pred_sector = st.selectbox("Sector:", ['Nifty 50', 'Nifty Bank', 'Nifty IT', 'Nifty Auto', 'Nifty Realty'], key='pred_sector')
        
        st.markdown("---")
        st.markdown("### 2. Live Sentiment Analysis")
        
        with st.spinner(f"Scanning news for {pred_sector}..."):
            auto_sentiment, headlines = get_live_sentiment(pred_sector)
            
        st.metric("Live Sentiment Score", f"{auto_sentiment:.2f}", delta="Bullish" if auto_sentiment > 0 else "Bearish")
        
        with st.expander("See Recent Headlines Analyzed"):
            if isinstance(headlines, list):
                for h in headlines:
                    st.caption(f"• {h}")
            else:
                st.caption(headlines)
        
        use_manual = st.checkbox("Override with Manual Sentiment?")
        if use_manual:
            final_sentiment = st.slider("Manual Score:", -5.0, 5.0, 0.0)
        else:
            final_sentiment = auto_sentiment
            
        st.markdown("---")
        st.markdown("### 3. Interest Rate Scenarios")
        us_rate_change = st.number_input("🇺🇸 US Rate Change (%)", min_value=-2.0, max_value=2.0, value=0.00, step=0.05)
        rbi_rate_change = st.number_input("🇮🇳 RBI Rate Change (%)", min_value=-2.0, max_value=2.0, value=0.00, step=0.05)
        
    with col_pred2:
        # --- ROBUST MODEL LOGIC ---
        # 1. Prepare Data
        df_calc = df.tail(365*5).pct_change()
        
        # 2. Clean Data (Handle Infinite values from % change calculation)
        df_calc = df_calc.replace([np.inf, -np.inf], np.nan).dropna()
        
        # 3. Check for sufficient data
        if len(df_calc) < 30:
            st.error("Insufficient data for regression analysis. Please try again later.")
        else:
            Y_calc = df_calc[pred_sector]
            X_calc = df_calc[['US 10Y Bond Yield', 'RBI Repo Rate']]
            
            # Ensure strictly float types
            Y_calc = Y_calc.astype(float)
            X_calc = X_calc.astype(float)
            
            X_calc = sm.add_constant(X_calc)
            
            try:
                model = sm.OLS(Y_calc, X_calc).fit()
                beta_us = model.params['US 10Y Bond Yield']
                beta_rbi = model.params['RBI Repo Rate']
                
                # Trend Logic
                recent_data = df[pred_sector].tail(30).dropna()
                if len(recent_data) > 10:
                    X_hist = np.arange(len(recent_data))
                    Y_hist = recent_data.values.astype(float)
                    coeffs = np.polyfit(X_hist, Y_hist, 1)
                    slope = coeffs[0]
                    
                    last_close = Y_hist[-1]
                    last_date = recent_data.index[-1]
                    
                    sentiment_daily_impact = (last_close * 0.001) * final_sentiment 
                    
                    total_rate_shock_pct = (beta_us * us_rate_change) + (beta_rbi * rbi_rate_change)
                    total_rate_price_shock = last_close * (total_rate_shock_pct / 100)
                    rate_daily_impact = total_rate_price_shock / 5
                    
                    future_dates = [last_date + timedelta(days=i) for i in range(1, 8)]
                    future_dates = [d for d in future_dates if d.weekday() < 5][:5]
                    
                    pred_prices = []
                    current_pred = last_close
                    
                    for _ in range(5):
                        current_pred = current_pred + slope + sentiment_daily_impact + rate_daily_impact
                        pred_prices.append(current_pred)
                        
                    # Visualization
                    hist_df = pd.DataFrame({'Date': recent_data.index, 'Price': Y_hist, 'Type': 'Actual History (30 Days)'})
                    pred_df = pd.DataFrame({'Date': future_dates, 'Price': pred_prices, 'Type': 'Predicted (Next 5 Days)'})
                    connect_row = pd.DataFrame({'Date': [recent_data.index[-1]], 'Price': [Y_hist[-1]], 'Type': 'Predicted (Next 5 Days)'})
                    pred_df = pd.concat([connect_row, pred_df])
                    full_df = pd.concat([hist_df, pred_df])
                    
                    fig_pred = px.line(full_df, x='Date', y='Price', color='Type',
                                    color_discrete_map={'Actual History (30 Days)': 'blue', 'Predicted (Next 5 Days)': 'green' if pred_prices[-1] > last_close else 'red'})
                    
                    fig_pred.update_layout(title=f"Forecast: {pred_sector} (Sentiment: {final_sentiment:.2f})", hovermode="x unified")
                    fig_pred.update_traces(patch={"line": {"dash": "dot", "width": 3}}, selector={"legendgroup": "Predicted (Next 5 Days)"})
                    
                    st.plotly_chart(fig_pred, use_container_width=True)
                    
                    final_pred = pred_prices[-1]
                    pct_change = ((final_pred - last_close) / last_close) * 100
                    
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Current Price", f"{last_close:,.2f}")
                    m2.metric("Predicted Price", f"{final_pred:,.2f}", f"{pct_change:.2f}%")
                    m3.info(f"**Rate Sensitivity:**\nUS Beta: {beta_us:.2f} | RBI Beta: {beta_rbi:.2f}")
                else:
                    st.error("Not enough recent data to generate prediction.")
            except Exception as e:
                st.error(f"Modeling Error: {e}")

# --- TAB 2: HISTORICAL IMPACT ---
with tab2:
    st.subheader("Historical Rate Cycles vs. Market")
    sector_hist = st.selectbox("Select Sector:", ['Nifty 50', 'Nifty Bank', 'Nifty IT'], key='hist_sector')
    
    norm_df = df.tail(365*10).copy()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=norm_df.index, y=norm_df[sector_hist], name=f"{sector_hist} Price", line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=norm_df.index, y=norm_df['RBI Repo Rate'], name="RBI Repo Rate", line=dict(color='orange'), yaxis='y2'))
    fig.add_trace(go.Scatter(x=norm_df.index, y=norm_df['US 10Y Bond Yield'], name="US 10Y Yield", line=dict(color='red', dash='dot'), yaxis='y2'))

    fig.update_layout(
        title=f"{sector_hist} vs. Policy Rates (10 Years)",
        yaxis=dict(title="Index Value"),
        yaxis2=dict(title="Interest Rate (%)", overlaying='y', side='right'),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.1)
    )
    st.plotly_chart(fig, use_container_width=True)

# --- TAB 3: SENSITIVITY ANALYSIS ---
with tab3:
    st.subheader("Sectoral Sensitivity Gauge")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        target_sector_s = st.selectbox("Select Target Sector:", ['Nifty Bank', 'Nifty IT', 'Nifty Realty', 'Nifty Auto'], key='sens_sector')
    
    # Robust data cleaning here too
    df_analysis = df.tail(365*5).pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    
    if len(df_analysis) > 30:
        Y = df_analysis[target_sector_s]
        X = df_analysis[['US 10Y Bond Yield', 'RBI Repo Rate']]
        X = sm.add_constant(X)
        model = sm.OLS(Y, X).fit()
        beta_us_s = model.params['US 10Y Bond Yield']
        beta_rbi_s = model.params['RBI Repo Rate']
        
        with col_s2:
            st.write("") 
            if abs(beta_us_s) > abs(beta_rbi_s):
                st.warning(f"⚠️ {target_sector_s} is more sensitive to **US Rates** ({beta_us_s:.2f}) than RBI Rates.")
            else:
                st.success(f"🛡️ {target_sector_s} is driven more by **Domestic RBI Rates** ({beta_rbi_s:.2f}).")

        beta_df = pd.DataFrame({
            'Factor': ['US 10Y Yield', 'RBI Repo Rate'],
            'Sensitivity (Beta)': [beta_us_s, beta_rbi_s]
        })
        fig_bar = px.bar(beta_df, x='Factor', y='Sensitivity (Beta)', color='Sensitivity (Beta)', color_continuous_scale='RdBu_r')
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.error("Insufficient data for sensitivity analysis.")

# --- TAB 4: CORRELATION ---
with tab4:
    st.subheader("Correlation Matrix")
    returns_corr = df.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    cols_to_corr = ['RBI Repo Rate', 'US 10Y Bond Yield', 'USD/INR', 'Nifty 50', 'Nifty Bank', 'Nifty IT']
    
    # Ensure columns exist
    valid_cols = [c for c in cols_to_corr if c in returns_corr.columns]
    
    fig_corr = px.imshow(returns_corr[valid_cols].corr(), text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    st.plotly_chart(fig_corr, use_container_width=True)
