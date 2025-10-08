import streamlit as st
import pandas as pd
import numpy as np
import feedparser
from datetime import datetime

def fetch_news(feed_url: str, limit: int = 5):
    """
    Returns a list of dicts: [{title, link, published}] from an RSS/Atom feed.
    Keeps it very small and safe.
    """
    try:
        d = feedparser.parse(feed_url)
        items = []
        for entry in d.entries[:limit]:
            title = entry.get("title", "Untitled")
            link = entry.get("link", "#")
            published = entry.get("published", "") or entry.get("updated", "")
            items.append({"title": title, "link": link, "published": published})
        return items
    except Exception:
        return []

st.set_page_config(
    page_title="The Rare Index",
    page_icon="favicon.png",
    layout="wide"
)

# --- Header & tagline ---
st.markdown("<h1 style='text-align: center; color: #2E8B57;'>The Rare Index</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; font-size:20px; color:#2E8B57;'><b>Explore Rare Index Categories</b></h3>", unsafe_allow_html=True)

# --- About blurb (below tagline) ---
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size:14px; color:#B22222;'>Demo Data Only — Not Financial Advice</p>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size:16px; color:#555;'>The Rare Index is a demo platform showcasing how alternative assets like trading cards and watches can be tracked against traditional markets.</p>", unsafe_allow_html=True)

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
# Simple RSS feeds for each category (feel free to adjust queries later)
FEEDS = {
    "Cards": [
        "https://news.google.com/rss/search?q=pokemon+trading+cards",
        "https://news.google.com/rss/search?q=trading+cards+market",
        "https://news.google.com/rss/search?q=TCG+sales"
    ],
    "Watches": [
        "https://news.google.com/rss/search?q=Rolex+watches",
        "https://news.google.com/rss/search?q=watch+auctions",
        "https://news.google.com/rss/search?q=luxury+watch+market"
    ],
    "Toys": [
        "https://news.google.com/rss/search?q=LEGO+retired+sets",
        "https://news.google.com/rss/search?q=LEGO+investment",
        "https://news.google.com/rss/search?q=toy+collectibles+market"
    ],
}
@st.cache_data(ttl=600)
def cached_fetch_news(feed_url: str, limit: int = 5):
    return fetch_news(feed_url, limit)
def render_news(category_name: str):
   def render_news(category_name: str):
    st.markdown("<h5 style='margin-top:0.5rem;'>Trending News</h5>", unsafe_allow_html=True)
    feeds = FEEDS.get(category_name, [])
    combined = []
    with st.spinner("Loading news..."):
        for url in feeds:
            combined.extend(cached_fetch_news(url, limit=3))
    # dedupe by title (very simple)
    seen = set()
    cleaned = []
    for item in combined:
        if item["title"] not in seen:
            seen.add(item["title"])
            cleaned.append(item)
        if len(cleaned) >= 5:
            break

    if not cleaned:
        st.caption("No news found right now.")
        return

    for item in cleaned:
        pub = item["published"]
        st.markdown(f"- [{item['title']}]({item['link']})  \n  <span style='color:#777;font-size:12px;'>{pub}</span>", unsafe_allow_html=True)
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

            # Title above the comparison chart
            st.markdown(f"<h4 style='text-align:center;'>{title} — Index vs {market_choice}</h4>", unsafe_allow_html=True)
            st.line_chart(plot_df)

            # metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Starting Price (Jan 2025)", f"${start:,.0f}")
            col2.metric("Latest Price", f"${end:,.0f}")
            col3.metric("ROI Since Jan", f"{roi:.1f}%")
            # outperformance in percentage points
            item_last = float(item_index.iloc[-1]) if hasattr(item_index, "iloc") else float(item_index[-1])
            market_last = float(market_index[-1])
            outperf_pp = (item_last - market_last)  # both are on a 100-based index
            col4.metric("Outperformance vs Index", f"{outperf_pp:+.1f} pp")

        else:
            # Price-only chart with title
            st.markdown(f"<h4 style='text-align:center;'>{title} — Price Trend</h4>", unsafe_allow_html=True)
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
# --- Three tabs: Cards, Watches, Toys ---
tab_cards, tab_watches, tab_toys = st.tabs(["Cards", "Watches", "Toys"])

with tab_cards:
    st.markdown("<p style='text-align:center; color:#555;'>Tracking monthly median sale prices for a representative Pokémon card.</p>", unsafe_allow_html=True)
    show_category("Lugia V Alt Art (Cards)", "cards.csv")
    render_news("Cards")

with tab_watches:
    st.markdown("<p style='text-align:center; color:#555;'>Tracking monthly median resale for a representative luxury watch reference.</p>", unsafe_allow_html=True)
    show_category("Rolex Submariner 116610LN (Watches)", "watches.csv")
    render_news("Watches")

with tab_toys:
    st.markdown("<p style='text-align:center; color:#555;'>Tracking monthly median resale for a flagship retired LEGO set.</p>", unsafe_allow_html=True)
    show_category("LEGO 75290 Mos Eisley Cantina (Toys)", "toys.csv")
    render_news("Toys")
    
st.markdown("---")
st.caption("RI Beta — Demo Data Only. Market lines use fixed 2025 YTD endpoints for simplicity.")
# --- Latest Updates Section ---
st.markdown("## 📢 Latest Updates")

st.markdown("""
- **Oct 2025:** Demo site launched with Cards, Watches, and Toys.
- **Sep 2025:** Lugia V Alt Art sales stable around $420–450 range.
- **Sep 2025:** Rolex Submariner 116610LN resale prices holding steady despite market dip.
- **Aug 2025:** LEGO 75290 Mos Eisley Cantina shows consistent ROI trend.
""")
st.markdown("---")
st.markdown("<p style='text-align: center; font-size:14px; color:#2E8B57;'>© 2025 The Rare Index · Demo Data Only</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size:14px;'><a href='mailto:david@therareindex.com'>Contact: david@therareindex.com</a></p>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size:14px;'><a href='https://forms.gle/KxufuFLcEVZD6qtD8' target='_blank'>Subscribe for updates</a></p>", unsafe_allow_html=True)






















