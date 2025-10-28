import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os
import logging
from typing import Optional
from snowflake.snowpark import Session

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_snowpark_session() -> Session:
    """Create Snowpark session from environment variables."""
    connection_parameters = {
        "account": os.getenv("SNOWFLAKE_ACCOUNT"),
        "user": os.getenv("SNOWFLAKE_USER"),
        "password": os.getenv("SNOWFLAKE_PASSWORD"),
        "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
        "database": os.getenv("SNOWFLAKE_DB"),
        "schema": os.getenv("SNOWFLAKE_SCHEMA"),
        "role": os.getenv("SNOWFLAKE_ROLE"),
    }
    return Session.builder.configs(connection_parameters).create()

def log_api_call(session: Session, endpoint: str, params: str, status: int):
    """Log API call to audit table."""
    try:
        query = f"""
        INSERT INTO API_LOGS (ENDPOINT, PARAMS, STATUS, TIMESTAMP)
        VALUES ('{endpoint}', '{params}', {status}, CURRENT_TIMESTAMP())
        """
        session.sql(query).collect()
    except Exception as e:
        logger.warning(f"Failed to log API call: {e}")

def ensure_tables(session: Session):
    """Create required tables if they don't exist."""
    tables = [
        """
        CREATE TABLE IF NOT EXISTS STOCK_PRICES (
            SYMBOL STRING,
            DATE DATE,
            OPEN FLOAT,
            HIGH FLOAT,
            LOW FLOAT,
            CLOSE FLOAT,
            VOLUME BIGINT,
            PRIMARY KEY (SYMBOL, DATE)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS API_LOGS (
            ID BIGINT IDENTITY,
            ENDPOINT STRING,
            PARAMS STRING,
            STATUS INT,
            TIMESTAMP TIMESTAMP_NTZ
        )
        """
    ]
    
    for table_sql in tables:
        session.sql(table_sql).collect()

def ingest_stock_data(symbol: str, start_date: Optional[str] = None, end_date: Optional[str] = None):
    """Ingest yfinance data into Snowflake."""
    session = create_snowpark_session()
    
    try:
        ensure_tables(session)
        
        # Default to last 30 days
        if not end_date:
            end_date = datetime.today().strftime('%Y-%m-%d')
        if not start_date:
            start_date = (datetime.today() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        logger.info(f"Fetching {symbol} data from {start_date} to {end_date}")
        
        # Fetch data
        ticker = yf.Ticker(symbol)
        hist = ticker.history(start=start_date, end=end_date)
        
        if hist.empty:
            logger.warning(f"No data returned for {symbol}")
            log_api_call(session, "yfinance", f"{symbol}:{start_date}:{end_date}", 404)
            return
        
        # Prepare DataFrame
        df = hist.reset_index()
        df['Symbol'] = symbol.upper()
        df = df.rename(columns={
            'Date': 'DATE',
            'Open': 'OPEN',
            'High': 'HIGH',
            'Low': 'LOW',
            'Close': 'CLOSE',
            'Volume': 'VOLUME'
        })
        df = df[['Symbol', 'DATE', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']]
        
        # Log API call
        log_api_call(session, "yfinance", f"{symbol}:{start_date}:{end_date}", 200)
        
        # Write to Snowflake with MERGE
        snowpark_df = session.create_dataframe(df)
        snowpark_df.write.mode("merge").save_as_table("STOCK_PRICES")
        
        logger.info(f"Successfully ingested {len(df)} rows for {symbol}")
        
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        log_api_call(session, "yfinance", f"{symbol}:{start_date}:{end_date}", 500)
        raise
    finally:
        session.close()

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python yfinance_ingestion.py SYMBOL [start_date] [end_date]")
        sys.exit(1)
    
    symbol = sys.argv[1]
    start_date = sys.argv[2] if len(sys.argv) > 2 else None
    end_date = sys.argv[3] if len(sys.argv) > 3 else None
    
    ingest_stock_data(symbol, start_date, end_date)