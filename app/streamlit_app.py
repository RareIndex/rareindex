import streamlit as st
from app.stock_price_viewer import show_stock_viewer

st.set_page_config(page_title="Rare Index", layout="wide")
st.title("Rare Index")

tab1, tab2 = st.tabs(["Stock Viewer", "Connectivity Tests"])

with tab1:
    show_stock_viewer()

with tab2:
    st.write("API connectivity tests coming soon")