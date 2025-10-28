import streamlit as st

st.set_page_config(page_title="Rare Index", layout="wide")

st.title("Hello, Rare Index!")

tab1, tab2 = st.tabs(["Overview", "Connectivity Tests"])

with tab1:
    st.write("Welcome to the Rare Index MVP.")
    st.write("Data ingestion and UI coming soon.")

with tab2:
    st.write("API connectivity will be tested here.")
    st.write("All APIs: yfinance, news, cards, toys, watches")