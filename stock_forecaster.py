import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
from prophet import Prophet

st.set_page_config(page_title="📈 Prophet Stock Forecast", layout="wide")
st.title("📈 Stock Price Forecast with Pretrained Prophet Model")

ticker = st.text_input("Enter stock ticker (e.g., AAPL):", "AAPL")

if ticker:
    try:
        df = yf.Ticker(ticker).history(period="2y")
        if df.empty:
            st.error(f"❌ No data found for ticker '{ticker.upper()}'. Please check the symbol and try again.")
            st.stop()
    except Exception as e:
        st.error(f"⚠️ Error fetching data: {e}")
        st.stop()
    df = df.reset_index()[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
    df["ds"] = df["ds"].dt.tz_localize(None)  # Remove timezone info


    

    # Prophet forecasting
    model = Prophet(daily_seasonality=True)
    model.fit(df)

    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    # Plot
    st.subheader(f"🔮 {ticker} Forecast for Next 30 Days")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["ds"], y=df["y"], name="Actual"))
    fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], name="Forecast"))
    fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_lower"], name="Lower Bound", line=dict(dash='dot')))
    fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_upper"], name="Upper Bound", line=dict(dash='dot')))
    fig.update_layout(xaxis_title="Date", yaxis_title="Price (USD)")
    st.plotly_chart(fig, use_container_width=True)
