import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta
import os
from typing import Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@st.cache_data(ttl=300)  # 5-minute cache
def get_stock_data(symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """Fetch stock data from Snowflake."""
    from snowflake.snowpark import Session
    
    try:
        connection_parameters = {
            "account": os.getenv("SNOWFLAKE_ACCOUNT"),
            "user": os.getenv("SNOWFLAKE_USER"),
            "password": os.getenv("SNOWFLAKE_PASSWORD"),
            "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
            "database": os.getenv("SNOWFLAKE_DB"),
            "schema": os.getenv("SNOWFLAKE_SCHEMA"),
            "role": os.getenv("SNOWFLAKE_ROLE"),
        }
        session = Session.builder.configs(connection_parameters).create()
        
        query = f"""
        SELECT DATE, OPEN, HIGH, LOW, CLOSE, VOLUME
        FROM STOCK_PRICES
        WHERE SYMBOL = '{symbol.upper()}'
          AND DATE BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY DATE
        """
        df = session.sql(query).to_pandas()
        session.close()
        
        logger.info(f"Fetched {len(df)} rows for {symbol}")
        return df if not df.empty else None
        
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {e}")
        st.error(f"Could not load data for {symbol}")
        return None

def show_stock_viewer():
    st.header("Stock Price Viewer")
    
    symbol = st.text_input("Stock Symbol", value="AAPL").upper()
    col1, col2 = st.columns(2)
    
    with col1:
        end_date = st.date_input("End Date", value=datetime.today())
    with col2:
        start_date = st.date_input("Start Date", value=end_date - timedelta(days=30))
    
    if st.button("Load Data"):
        with st.spinner(f"Loading {symbol} data..."):
            df = get_stock_data(symbol, str(start_date), str(end_date))
            
            if df is not None:
                # Candlestick chart
                fig = go.Figure(data=[go.Candlestick(
                    x=df['DATE'],
                    open=df['OPEN'],
                    high=df['HIGH'],
                    low=df['LOW'],
                    close=df['CLOSE']
                )])
                fig.update_layout(title=f"{symbol} Price Chart", xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)
                
                # Volume chart
                fig_vol = go.Figure(data=[go.Bar(x=df['DATE'], y=df['VOLUME'])])
                fig_vol.update_layout(title="Volume")
                st.plotly_chart(fig_vol, use_container_width=True)
                
                # Key metrics
                latest = df.iloc[-1]
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Latest Price", f"${latest['CLOSE']:.2f}")
                col2.metric("High", f"${latest['HIGH']:.2f}")
                col3.metric("Low", f"${latest['LOW']:.2f}")
                col4.metric("Volume", f"{latest['VOLUME']:,}")
                
                # Export
                csv = df.to_csv(index=False)
                st.download_button("Download CSV", csv, f"{symbol}_data.csv", "text/csv")