import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
import warnings

# Suppress harmless ARIMA warnings
warnings.filterwarnings('ignore')

# --- 1. Configuration and Constants ---
APP_TITLE = "The ARIMA Risk Navigator: NSE/BSE Forecast"
# Use .NS for NSE or .BO for BSE
TICKER_SYMBOLS = {
    'Reliance Industries (NSE)': 'RELIANCE.NS',
    'Tata Consultancy Services (NSE)': 'TCS.NS',
    'HDFC Bank (NSE)': 'HDFCBANK.NS',
    'ICICI Bank (NSE)': 'ICICIBANK.NS',
    'Bajaj Finance (NSE)': 'BAJFINANCE.NS',
    'State Bank of India (NSE)': 'SBIN.NS',
    'Tata Motors (BSE)': 'TATAMOTORS.BO',
}
DAYS_TO_FORECAST_MAX = 30
LAKH_CRORE = 1e12 # 1 Lakh Crore = 10^12 (1 Trillion)

# --- Custom CSS for Styling ---
def set_custom_styles():
    """Applies custom CSS for a dashboard look and background color."""
    
    st.markdown("""
        <style>
        /* Main background color - Light Gray */
        [data-testid="stAppViewContainer"] > .main {
            background-color: #f0f2f6; /* <--- MAIN APP BACKGROUND COLOR */
        }

        /* Metrics Container background - White/Off-White for contrast */
        .metrics-container {
            padding: 15px;
            border-radius: 10px;
            background-color: #ffffff; /* <--- METRICS BOX BACKGROUND COLOR */
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }

        /* Adjust the font size of the main title */
        .css-1av0ud8 { 
            font-size: 2.5em; 
        }
        </style>
        """, unsafe_allow_html=True)
        
# --- 2. Data Fetching and Metrics Calculation ---

@st.cache_data(ttl=3600)
def get_historical_data(ticker_symbol):
    """Fetches 5 years of historical data."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5 * 365)
    try:
        ticker = yf.Ticker(ticker_symbol)
        history_df = ticker.history(start=start_date, end=end_date)
        info = ticker.info
        
        if history_df.empty or not info:
            raise ValueError("Data fetching resulted in empty data.")
        
        history_df.index = history_df.index.tz_localize(None)
        
        return history_df.dropna(), info
    except Exception as e:
        st.error(f"Error fetching historical data for **{ticker_symbol}**. Error: {e}")
        return None, None

def calculate_financial_metrics(history_df, info):
    """Calculates Current Price, Change %, P/L, and Market Cap."""
    if history_df.empty:
        return None
    
    current_price = history_df['Close'].iloc[-1]
    last_close = history_df['Close'].iloc[-2]
    last_open = history_df['Open'].iloc[-1]
    
    change_percent = (current_price - last_close) / last_close * 100
    daily_pl = current_price - last_open
    
    market_cap = info.get('marketCap', 0)
    market_cap_lakh_cr = market_cap / LAKH_CRORE if market_cap else 0
    
    return {
        'current_price': current_price,
        'change_percent': change_percent,
        'daily_pl': daily_pl,
        'market_cap_lakh_cr': market_cap_lakh_cr,
    }

# --- 3. ARIMA Forecasting and Risk Assessment ---

@st.cache_data(ttl=3600) 
def run_arima_forecast(data_series, p, d, q, steps):
    """Runs the ARIMA model and returns the forecast and confidence intervals."""
    try:
        model = ARIMA(data_series.tail(365), order=(p, d, q))
        model_fit = model.fit()
        
        forecast = model_fit.get_forecast(steps=steps)
        forecast_df = forecast.summary_frame(alpha=0.05)
        
        # --- FIX FOR LENGTH MISMATCH ERROR ---
        columns_to_select = ['mean', 'mean_ci_lower', 'mean_ci_upper']
        existing_cols = [col for col in columns_to_select if col in forecast_df.columns]
        
        if len(existing_cols) < 3:
             raise ValueError(f"Forecast frame has insufficient columns: {forecast_df.columns.tolist()}")

        forecast_df = forecast_df[existing_cols]
        forecast_df.columns = ['Predicted', 'Lower_CI', 'Upper_CI'] 
        # -------------------------------------

        # Create future dates for the forecast
        last_date = data_series.index[-1]
        future_dates = pd.date_range(start=last_date, periods=steps + 1, inclusive='right')
        forecast_df.index = future_dates
        
        return forecast_df
        
    except Exception as e:
        st.error(f"Error running ARIMA model. Check ARIMA parameters (p, d, q): {e}")
        return None

def generate_risk_signal(forecast_df, days_to_predict):
    """Generates a GO/NO-GO recommendation based on forecast vs. volatility band."""
    
    if forecast_df is None or days_to_predict < 5:
        return "HOLD", "🟡", "Forecast steps too short for risk assessment (min 5 days required)."

    final_predicted_price = forecast_df['Predicted'].iloc[-1]
    
    try:
        ci_target_index = days_to_predict - 5 # 5 days before the end
        ci_lower_5d = forecast_df['Lower_CI'].iloc[ci_target_index]
        ci_upper_5d = forecast_df['Upper_CI'].iloc[ci_target_index]
    except IndexError:
        return "HOLD", "🟡", "Not enough forecast days to check the 5-day interval."

    # Logic Implementation:
    if final_predicted_price > ci_upper_5d:
        message = f"**GO (BUY/HOLD)**: Final forecast (**₹{final_predicted_price:,.2f}**) significantly exceeds the 5-day volatility upper bound (**₹{ci_upper_5d:,.2f}**). Volatility-adjusted prediction is strong."
        return "GO", "✅", message
    
    elif final_predicted_price < ci_lower_5d:
        message = f"**NO-GO (SELL/AVOID)**: Final forecast (**₹{final_predicted_price:,.2f}**) falls below the 5-day volatility lower bound (**₹{ci_lower_5d:,.2f}**). High risk; forecast is discounting recent volatility."
        return "NO-GO", "❌", message
    
    else:
        message = f"**HOLD**: Forecast (**₹{final_predicted_price:,.2f}**) is within the expected 5-day volatility band (**₹{ci_lower_5d:,.2f}** to **₹{ci_upper_5d:,.2f}**). Neutral risk profile."
        return "HOLD", "🟡", message

# --- 4. Charting Functions ---

def create_candlestick_chart(df, ticker_name):
    """Creates a Candlestick chart for the last 1 year."""
    df_1y = df.tail(252)
    
    fig = go.Figure(data=[go.Candlestick(
        x=df_1y.index,
        open=df_1y['Open'],
        high=df_1y['High'],
        low=df_1y['Low'],
        close=df_1y['Close'],
        name='Price'
    )])
    
    fig.update_layout(
        title='**1-Year Price Action (Candlestick)**',
        xaxis_rangeslider_visible=False,
        height=400,
        template="plotly_white",
        margin=dict(t=50, b=20, l=20, r=20)
    )
    return fig

def create_forecast_plot(history_df, forecast_df, days_to_predict, p, d, q):
    """Creates a plot of historical data, forecast, and CI band."""
    
    plot_df_actual = history_df['Close'].tail(90).reset_index() 
    plot_df_forecast = forecast_df.reset_index().rename(columns={'index': 'Date'})
    
    fig = go.Figure()

    # 1. Shaded Confidence Interval (Risk Band)
    fig.add_trace(go.Scatter(
        x=plot_df_forecast['Date'],
        y=plot_df_forecast['Upper_CI'],
        mode='lines',
        name='95% Upper CI',
        line=dict(width=0),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=plot_df_forecast['Date'],
        y=plot_df_forecast['Lower_CI'],
        fill='tonexty',
        fillcolor='rgba(255, 165, 0, 0.2)', # Orange/Yellow shading for risk
        mode='lines',
        name='95% Confidence Interval',
        line=dict(width=0)
    ))

    # 2. Historical Data
    fig.add_trace(go.Scatter(
        x=plot_df_actual['Date'], 
        y=plot_df_actual['Close'],
        mode='lines',
        name='Historical Close',
        line=dict(color='#0077b6', width=2)
    ))

    # 3. Predicted Price Line
    fig.add_trace(go.Scatter(
        x=plot_df_forecast['Date'],
        y=plot_df_forecast['Predicted'],
        mode='lines',
        name='Predicted Price',
        line=dict(color='#dc3545', width=3, dash='dash')
    ))

    fig.update_layout(
        title=f'**ARIMA ({p}, {d}, {q}) {days_to_predict}-Day Forecast & Risk Band**',
        height=400,
        template="plotly_white",
        yaxis_title="Price (₹)",
        margin=dict(t=50, b=20, l=20, r=20)
    )
    return fig

# --- 5. Streamlit UI (main function) ---

def main():
    # Apply custom styles
    set_custom_styles()
    
    st.set_page_config(
        page_title=APP_TITLE,
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title(APP_TITLE)
    st.markdown("---")
    
    # --- Sidebar for Inputs ---
    st.sidebar.header("⚙️ Configuration")
    
    selected_name = st.sidebar.selectbox(
        'Select Stock',
        list(TICKER_SYMBOLS.keys())
    )
    ticker_symbol = TICKER_SYMBOLS[selected_name]

    st.sidebar.markdown("---")
    
    # ARIMA Parameters
    st.sidebar.subheader("ARIMA Order (p, d, q)")
    p = st.sidebar.number_input('P (Autoregressive)', min_value=0, max_value=5, value=5, step=1)
    d = st.sidebar.number_input('D (Differencing)', min_value=0, max_value=2, value=1, step=1)
    q = st.sidebar.number_input('Q (Moving Average)', min_value=0, max_value=5, value=0, step=1)
    
    st.sidebar.markdown("---")
    
    # Days to Predict
    days_to_predict = st.sidebar.slider(
        'Days to Forecast (Max 30)',
        min_value=5, 
        max_value=DAYS_TO_FORECAST_MAX, 
        value=15, 
        step=1
    )

    # --- Data Fetching ---
    with st.spinner(f"Fetching data for {ticker_symbol}..."):
        history_df, info = get_historical_data(ticker_symbol)

    if history_df is None or info is None:
        st.stop()
        
    metrics = calculate_financial_metrics(history_df, info)

    # --- Dashboard Layout: Metrics Container ---
    st.header(f"💰📈 {selected_name} Performance Snapshot")
    
    # Use a container for the metrics and apply the custom CSS class
    with st.container(border=False):
        st.markdown('<div class="metrics-container">', unsafe_allow_html=True)
        metric_cols = st.columns(4)
        
        # Run ARIMA here so the signal is ready for the 4th metric column
        forecast_df = run_arima_forecast(history_df['Close'], p, d, q, days_to_predict)

        with metric_cols[0]:
            st.metric(
                label="Current Price (₹)",
                value=f"₹{metrics['current_price']:,.2f}",
                delta=f"{metrics['change_percent']:+.2f}%",
            )
        with metric_cols[1]:
            st.metric(
                label="Daily P/L (Today)",
                value=f"₹{metrics['daily_pl']:,.2f}",
                delta_color="off",
                help="Close Price - Open Price."
            )
        with metric_cols[2]:
            st.metric(
                label="Market Cap (Lakh Cr)",
                value=f"₹{metrics['market_cap_lakh_cr']:,.2f}",
                delta_color="off",
            )
        with metric_cols[3]:
            if forecast_df is not None:
                signal, emoji, _ = generate_risk_signal(forecast_df, days_to_predict)
                color = "green" if signal == "GO" else ("red" if signal == "NO-GO" else "orange")
                
                # Custom HTML for the prominent signal box
                st.markdown(f'''
                    <div style="text-align: center; border: 3px solid {color}; padding: 10px; border-radius: 8px; background-color: {color}1A;">
                    <span style="font-size: 0.9em; font-weight: bold; color: #333333;">RISK SIGNAL</span>
                    <h3 style="margin: 5px 0 0 0; color: {color};">{emoji} {signal}</h3>
                    </div>
                ''', unsafe_allow_html=True)
            else:
                st.warning("Prediction Failed")

        st.markdown('</div>', unsafe_allow_html=True) # Close metrics-container div

    st.markdown("---")

    # --- Dashboard Layout: Charts and Details Columns ---
    
    chart_col, detail_col = st.columns([1.7, 1])
    
    with chart_col:
        # 1. Candlestick Chart
        st.subheader("📈 Historical Price & Volatility")
        candlestick_fig = create_candlestick_chart(history_df, selected_name)
        st.plotly_chart(candlestick_fig, use_container_width=True)
        
        # 2. Forecast Plot with CI
        if forecast_df is not None:
            st.subheader("🔮 ARIMA Forecast & Risk Band")
            forecast_fig = create_forecast_plot(history_df, forecast_df, days_to_predict, p, d, q)
            st.plotly_chart(forecast_fig, use_container_width=True)
            

    with detail_col:
        st.subheader("📊 Prediction Details & Risk Assessment")
        
        if forecast_df is not None:
            # Display Signal Message
            _, _, message = generate_risk_signal(forecast_df, days_to_predict)
            st.info(message)
            
            # Display Prediction Summary Table
            st.markdown("##### Forecast Summary")
            final_pred = forecast_df['Predicted'].iloc[-1]
            lower_ci = forecast_df['Lower_CI'].iloc[-1]
            upper_ci = forecast_df['Upper_CI'].iloc[-1]
            
            summary_data = {
                'Metric': ['Forecast End Date', 'Predicted Close Price', '95% CI Lower Bound', '95% CI Upper Bound'],
                'Value': [(forecast_df.index[-1]).strftime('%b %d, %Y'), 
                          f"₹{final_pred:,.2f}", 
                          f"₹{lower_ci:,.2f}", 
                          f"₹{upper_ci:,.2f}"]
            }
            summary_df = pd.DataFrame(summary_data).set_index('Metric')
            st.table(summary_df)
            
            st.markdown("---")
            st.markdown("##### Full Forecast Data")
            st.dataframe(forecast_df, use_container_width=True, height=200)

    st.markdown("---")
    st.caption("""
        **Disclaimer:** This is an academic tool for **ARIMA forecasting** and **volatility assessment**. The 'GO/NO-GO' signal is simulated based on the prediction's deviation from its 95% confidence interval. **Do NOT use for actual trading or financial decisions.**
    """)

if __name__ == "__main__":
    main()
