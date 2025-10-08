st.set_page_config(
    page_title="The Rare Index",
    page_icon="favicon.png",
    layout="wide"
)

# --- Header & tagline ---
st.markdown("<h1 style='text-align: center; color: #2E8B57;'>The Rare Index</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size:18px;'>Tracking ROI of alternative assets</p>", unsafe_allow_html=True)

# --- About blurb (below tagline) ---
st.markdown("<p style='text-align: center; font-size:16px; color:#555;'>The Rare Index is a demo platform showcasing how alternative assets like trading cards and watches can be tracked against traditional markets.</p>", unsafe_allow_html=True)

st.markdown("---")
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="The Rare Index",
    page_icon="favicon.png",
    layout="wide"
)
st.markdown("<p style='text-align: center; font-size:16px; color:#555;'>The Rare Index is a demo platform showcasing how alternative assets like trading cards and watches can be tracked against traditional markets.</p>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: #2E8B57;'>The Rare Index</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size:18px;'>Tracking ROI of alternative assets</p>", unsafe_allow_html=True)
st.markdown("---")

# ---- Simple index generator (demo): spread a chosen YTD over N months ----
def make_index_series(final_return, n_points):
    """
    Returns a numpy array of index levels starting at 100 that compounds
    smoothly to 100*(1+final_return) over n_points.
    """
    target = 1.0 + final_return
    # equal monthly growth rate
    r = target ** (1 / max(n_points - 1, 1)) - 1
    levels = [100.0]
    for _ in range(n_points - 1):
        levels.append(levels[-1] * (1 + r))
    return np.array(levels)

# Fixed demo YTDs for 2025 (tweak later if you want)
MARKETS = {
    "S&P 500 (~+12% YTD 2025)": 0.12,
    "Nasdaq 100 (~+18% YTD 2025)": 0.18,
    "Dow Jones (~+9.5% YTD 2025)": 0.095,
}

# --- Helper to load and show one category ---
def show_category(title, csv_path):
    st.subheader(title)
    try:
        df = pd.read_csv(csv_path)
        # make sure the date column is read as dates and sorted
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        # basic ROI
        start = float(df["price_usd"].iloc[0])
        end = float(df["price_usd"].iloc[-1])
        roi = ((end - start) / start) * 100.0

        # UI: toggle + market choice
        with st.expander("Options", expanded=False):
            compare = st.checkbox("Compare to a stock index", value=False, key=f"cmp_{title}")
            market_choice = st.selectbox("Choose index", list(MARKETS.keys()), index=0, key=f"mkt_{title}")

        if compare:
            # Normalize item to 100 at first point
            item_index = (df["price_usd"] / df["price_usd"].iloc[0]) * 100.0
            # Build market line with same number of points
            market_ytd = MARKETS[market_choice]
            market_index = make_index_series(market_ytd, len(df))

            plot_df = pd.DataFrame({
                title: item_index.values,
                market_choice: market_index
            }, index=df["date"])

            st.line_chart(plot_df)

            col1, col2, col3 = st.columns(3)
            col1.metric("Starting Price (Jan 2025)", f"${start:,.0f}")
            col2.metric("Latest Price", f"${end:,.0f}")
            col3.metric("ROI Since Jan", f"{roi:.1f}%")
        else:
            # Price-only chart
            st.line_chart(df.set_index("date")["price_usd"])

            col1, col2, col3 = st.columns(3)
            col1.metric("Starting Price (Jan 2025)", f"${start:,.0f}")
            col2.metric("Latest Price", f"${end:,.0f}")
            col3.metric("ROI Since Jan", f"{roi:.1f}%")

        st.caption("Recent data points")
        st.dataframe(df.tail(5), width="stretch")

    except FileNotFoundError:
        st.warning(f"Could not find {csv_path}. Make sure the file exists.")
    except Exception as e:
        st.error(f"Error loading {csv_path}: {e}")

st.markdown("<h3 style='text-align: center; font-size:20px;'>Explore Rare Index Categories</h3>", unsafe_allow_html=True)
# --- Two tabs: Cards and Watches ---
tab_cards, tab_watches = st.tabs(["Cards", "Watches"])
with tab_cards:
    show_category("Lugia V Alt Art (Cards)", "cards.csv")
with tab_watches:
    show_category("Rolex Submariner 116610LN (Watches)", "watches.csv")

st.markdown("---")
st.caption("RI Beta — Demo Data Only. Market lines use fixed 2025 YTD endpoints for simplicity.")
st.markdown("---")
st.markdown("<p style='text-align: center; font-size:14px; color:#2E8B57;'>© 2025 The Rare Index · Demo Data Only</p>", unsafe_allow_html=True)









