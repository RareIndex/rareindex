import streamlit as st
import pandas as pd
import numpy as np
import feedparser
from utils.newsletter import render_newsletter_tools

@st.cache_data(ttl=600)
def read_csv_cached(path):
    """Reads and cleans a CSV once, caches it for 10 minutes."""
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df

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

# --- News Feeds & Helpers ---
FEEDS = {
    "Cards": [
        "https://news.google.com/rss/search?q=pokemon+trading+cards",
        "https://news.google.com/rss/search?q=trading+cards+market",
        "https://news.google.com/rss/search?q=TCG+sales",
    ],
    "Watches": [
        "https://news.google.com/rss/search?q=Rolex+watches",
        "https://news.google.com/rss/search?q=watch+auctions",
        "https://news.google.com/rss/search?q=luxury+watch+market",
    ],
    "Toys": [
        "https://news.google.com/rss/search?q=LEGO+retired+sets",
        "https://news.google.com/rss/search?q=LEGO+investment",
        "https://news.google.com/rss/search?q=toy+collectibles+market",
    ],
}

@st.cache_data(ttl=600)
def cached_fetch_news(feed_url: str, limit: int = 5):
    """Cached wrapper so we don't fetch feeds on every rerun."""
    return fetch_news(feed_url, limit)

def render_news(category_name: str):
    st.markdown("<h5 style='margin-top:0.5rem;'>Trending News</h5>", unsafe_allow_html=True)
    feeds = FEEDS.get(category_name, [])
    combined = []
    with st.spinner("Loading news..."):
        for url in feeds:
            combined.extend(cached_fetch_news(url, limit=3))

        # Deduplicate by title (very simple)
    seen = set()
    cleaned = []
    for item in combined:
        t = item.get("title", "")
        if t and t not in seen:
            seen.add(t)
            cleaned.append(item)
        if len(cleaned) >= 5:
            break

    if not cleaned:
        st.caption("No news found right now.")
        return

    # Loop through cleaned items (properly indented)
    for item in cleaned:
        pub = item.get("published", "")
        st.markdown(
            f"- [{item.get('title','Untitled')}]({item.get('link','#')})  \n"
            f"  <span style='color:#777;font-size:12px;'>{pub}</span>",
            unsafe_allow_html=True,
        )

# --- Helper to load and show one category ---
def show_category(title, csv_path):
    st.subheader(title)
    try:
        # Cached CSV read
        df = read_csv_cached(csv_path)

        # ---- Options
        with st.expander("Options", expanded=False):
            compare = st.checkbox("Compare to a stock index", value=False, key=f"cmp_{title}")
            market_choice = st.selectbox("Choose index", list(MARKETS.keys()), index=0, key=f"mkt_{title}")
            jan_baseline = st.checkbox("Use Jan 2025 as baseline (if available)", value=True, key=f"jan_{title}")

        # ---- Choose the slice we use for plotting + metrics
        if jan_baseline:
            df_use = df[df["date"] >= pd.Timestamp("2025-01-01")].copy()
            if df_use.empty:
                df_use = df.copy()
        else:
            df_use = df.copy()

        # Safety check — if df_use is still empty (e.g., bad CSV)
        if df_use.empty:
            st.warning("No data available for the selected baseline window.")
            return

        # basic ROI on the chosen slice
        start = float(df_use["price_usd"].iloc[0])
        end = float(df_use["price_usd"].iloc[-1])
        roi = ((end - start) / start) * 100.0
        start_label = f"Starting Price ({df_use['date'].iloc[0].strftime('%b %Y')})"

        if compare:
            item_index = (df_use["price_usd"] / df_use["price_usd"].iloc[0]) * 100.0
            market_ytd = MARKETS[market_choice]
            market_index = make_index_series(market_ytd, len(df_use))

            plot_df = pd.DataFrame({title: item_index.values, market_choice: market_index},
                                   index=df_use["date"])
            st.markdown(f"<h4 style='text-align:center;'>{title} — Index vs {market_choice}</h4>", unsafe_allow_html=True)
            st.line_chart(plot_df)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric(start_label, f"${start:,.0f}")
            col2.metric("Latest Price", f"${end:,.0f}")
            col3.metric(f"ROI since {df_use['date'].iloc[0].strftime('%b %Y')}", f"{roi:.1f}%")
            item_last = float(item_index.iloc[-1])
            market_last = float(market_index[-1])
            col4.metric("Outperformance vs Index", f"{(item_last - market_last):+.1f} pp")
        else:
            st.markdown(f"<h4 style='text-align:center;'>{title} — Price Trend</h4>", unsafe_allow_html=True)
            st.line_chart(df_use.set_index("date")["price_usd"])

            col1, col2, col3 = st.columns(3)
            col1.metric(start_label, f"${start:,.0f}")
            col2.metric("Latest Price", f"${end:,.0f}")
            col3.metric(f"ROI since {df_use['date'].iloc[0].strftime('%b %Y')}", f"{roi:.1f}%")

        st.caption("Recent data points")
        st.dataframe(df_use.tail(5), width="stretch")

    except FileNotFoundError:
        st.warning(f"Could not find {csv_path}. Make sure the file exists.")
    except Exception as e:
        st.error(f"Error loading {csv_path}: {e}")

# --- Helper: compute ROI from a CSV (first vs latest) ---
def calc_roi_from_csv(name: str, category: str, csv_path: str):
    """
    Returns a dict with name, category, start, latest, roi_pct.
    If anything goes wrong (missing file/columns), roi_pct is None.
    """
    try:
        df = read_csv_cached(csv_path)  # use cached loader

        start = float(df["price_usd"].iloc[0])
        latest = float(df["price_usd"].iloc[-1])
        roi_pct = ((latest - start) / start) * 100.0
        return {"name": name, "category": category, "start": start, "latest": latest, "roi_pct": roi_pct}
    except Exception:
        return {"name": name, "category": category, "start": None, "latest": None, "roi_pct": None}

# --- Tabs: Cards, Watches, Toys, Live eBay, ROI, Top 10 ---
tab_cards, tab_watches, tab_toys, tab_live, tab_roi, tab_top10 = st.tabs(
    ["Cards", "Watches", "Toys", "Live eBay (beta)", "ROI Calculator", "Top 10 (Demo)"]
)

# --- Top 10 ROI (Demo) ---
with tab_top10:
    st.markdown("### 🏆 Top 10 ROI (Demo)")

    demo_items = [
        {"name": "Card #010 (Demo)", "category": "Cards", "csv": "cards/cards_010.csv"},
        {"name": "Card #011 (Demo)", "category": "Cards", "csv": "cards/cards_011.csv"},
        {"name": "Card #012 (Demo)", "category": "Cards", "csv": "cards/cards_012.csv"},
        {"name": "Rolex Submariner 116610LN", "category": "Watches", "csv": "watches.csv"},
        {"name": "LEGO 75290 Mos Eisley Cantina", "category": "Toys", "csv": "toys.csv"}
    ]

    # Compute ROI for each item (first vs latest row in each CSV)
    results = [calc_roi_from_csv(x["name"], x["category"], x["csv"]) for x in demo_items]

    # --- DEBUG (temporary) ---
    with st.expander("🔧 Debug Top 10 items (temporary)"):
        import os
        df_raw = pd.DataFrame(results)  # before dropping NaNs
        st.write("Raw results:", df_raw)
        for x in demo_items:
            p = x["csv"]
            exists = os.path.exists(p)
            st.write(f"{x['name']} → {p} → exists: {exists}")
            if exists:
                try:
                    st.write(pd.read_csv(p, nrows=2))
                except Exception as e:
                    st.write("read error:", str(e))

    # Build DataFrame and clean
    df_top = pd.DataFrame(results)
    df_top = df_top.dropna(subset=["roi_pct"]).copy()

    if not df_top.empty:
        # Sort by ROI desc and keep top 10
        df_top = df_top.sort_values("roi_pct", ascending=False).head(10).reset_index(drop=True)

        # Pretty columns
        df_top["Start ($)"]  = df_top["start"].apply(lambda v: f"${v:,.2f}" if pd.notnull(v) else "—")
        df_top["Latest ($)"] = df_top["latest"].apply(lambda v: f"${v:,.2f}" if pd.notnull(v) else "—")
        df_top["ROI (%)"]    = df_top["roi_pct"].apply(lambda v: f"{v:,.2f}%" if pd.notnull(v) else "—")

        # Show table
        st.dataframe(
            df_top[["name", "category", "Start ($)", "Latest ($)", "ROI (%)"]],
            use_container_width=True
        )

        # Quick metrics
        c1, c2 = st.columns(2)
        c1.metric("Entries ranked", f"{len(df_top)}")
        c2.metric("Top ROI", df_top.iloc[0]["ROI (%)"] if len(df_top) else "—")

        # Download Top 10 table as CSV
        csv_bytes = df_top[["name", "category", "Start ($)", "Latest ($)", "ROI (%)"]].to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Top 10 as CSV",
            csv_bytes,
            file_name="top10_demo.csv",
            mime="text/csv"
        )
        # --- Newsletter snippet (Markdown) ---
        st.markdown("#### ✉️ Newsletter snippet (Markdown)")
        today = pd.Timestamp.today().strftime("%b %d, %Y")

        # Build simple bullet list from current Top 10 table
        lines = [f"**Top ROI — {today}**", ""]
        for i, row in df_top.iterrows():
            lines.append(f"{i+1}. **{row['name']}** ({row['category']}): {row['ROI (%)']}")

        snippet = "\n".join(lines)

        snippet = render_newsletter_tools(snippet, key="top10-newsletter")

    else:
        st.info("No ROI data available yet. Make sure your CSVs exist and have 'date' and 'price_usd' columns.")

with tab_cards:
    st.markdown("<p style='text-align:center; color:#555;'>Tracking monthly median sale prices for a representative Pokémon card.</p>", unsafe_allow_html=True)
    show_category("Card #011 (Cards)", "cards/cards_011.csv")
    render_news("Cards")

with tab_watches:
    st.markdown("<p style='text-align:center; color:#555;'>Tracking monthly median resale for a representative luxury watch reference.</p>", unsafe_allow_html=True)
    show_category("Rolex Submariner 116610LN (Watches)", "watches.csv")
    render_news("Watches")

with tab_toys:
    st.markdown("<p style='text-align:center; color:#555;'>Tracking monthly median resale for a flagship retired LEGO set.</p>", unsafe_allow_html=True)
    show_category("LEGO 75290 Mos Eisley Cantina (Toys)", "toys.csv")
    render_news("Toys")

with tab_live:
    st.markdown("### Live eBay (beta)")

    # Demo toggle
    demo_mode = st.toggle("Demo mode (no API)", value=True, help="Shows a tiny sample so the heartbeat is visible.")

    # CSV upload (optional)
    uploaded = st.file_uploader(
        "Upload CSV (columns: title, price, currency, listingType, condition, category, viewItemURL)",
        type=["csv"]
    )

    # Buttons
    colA, colB = st.columns(2)
    show_demo = colA.button("Show demo results")
    show_csv  = colB.button("Show uploaded CSV")

    df = None

    if uploaded is not None and show_csv:
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            df = None

    elif demo_mode and show_demo:
        demo_rows = [
            {
                "title": "Pokemon Charizard Base Set Holo PSA 9",
                "price": 1750, "currency": "USD", "listingType": "FixedPrice",
                "condition": "Mint", "category": "Collectible Card Games",
                "viewItemURL": "https://www.ebay.com/itm/111111111111",
            },
            {
                "title": "Rolex Submariner 114060",
                "price": 8950, "currency": "USD", "listingType": "Auction",
                "condition": "Pre-owned", "category": "Watches",
                "viewItemURL": "https://www.ebay.com/itm/222222222222",
            },
            {
                "title": "Omega Speedmaster 3570.50",
                "price": 4650, "currency": "USD", "listingType": "FixedPrice",
                "condition": "Pre-owned", "category": "Watches",
                "viewItemURL": "https://www.ebay.com/itm/333333333333",
            },
        ]
        df = pd.DataFrame(demo_rows)

    if df is not None and not df.empty:
        st.subheader("Results")

        # Build a display copy with a formatted price column
        df_display = df.copy()
        df_display["price ($)"] = pd.to_numeric(df_display["price"], errors="coerce").apply(
            lambda v: f"${v:,.2f}" if pd.notnull(v) else "—"
        )

        st.dataframe(
            df_display[["title","price ($)","currency","listingType","condition","category","viewItemURL"]],
            use_container_width=True
        )

        # Divider
        st.markdown("---")

        # Quick Links
        st.markdown("### 🔗 Quick Links")
        for _, row in df.iterrows():
            title = str(row.get("title", "")).strip()
            url = str(row.get("viewItemURL", "")).strip()
            if title and url and url.startswith("http"):
                st.markdown(f"- [{title}]({url})")

        # Category Badges
        st.markdown("### 🏷 Categories")
        badge_colors = {"Watches": "#16a34a", "Toys": "#f59e0b", "Collectible Card Games": "#3b82f6"}
        st.markdown(" ".join(
            [f"<span style='display:inline-block;padding:2px 8px;margin-right:6px;border-radius:999px;background:{badge_colors.get(str(cat),'#64748b')};color:white;font-size:12px'>{str(cat)}</span>"
             for cat in sorted(set(df["category"].dropna().astype(str))) if cat]
        ), unsafe_allow_html=True)

        # Metrics
        prices = pd.to_numeric(df["price"], errors="coerce").dropna()
        c1, c2, c3 = st.columns(3)
        c1.metric("📦 Items", f"{len(df):,}")
        c2.metric("💲 Median price", f"${prices.median():,.2f}" if not prices.empty else "—")
        c3.metric("💰 Average price", f"${prices.mean():,.2f}" if not prices.empty else "—")

        # Download
        st.download_button(
            "Download table as CSV",
            df.to_csv(index=False).encode("utf-8"),
            file_name="rareindex_results.csv",
            mime="text/csv"
        )

with tab_roi:
    st.markdown("### 📊 ROI Calculator")

    buy_price = st.number_input("Enter Buy Price ($)", min_value=0.0, step=1.0, format="%.2f")
    current_price = st.number_input("Enter Current Price ($)", min_value=0.0, step=1.0, format="%.2f")
    years_held = st.number_input("Years Held", min_value=0.0, step=0.5, format="%.1f")

    if buy_price > 0 and current_price > 0:
        roi = ((current_price - buy_price) / buy_price) * 100
        st.metric("Return on Investment (ROI %)", f"{roi:.2f}%")

        if years_held > 0:
            cagr = ((current_price / buy_price) ** (1 / years_held) - 1) * 100
            st.metric("Compound Annual Growth Rate (CAGR %)", f"{cagr:.2f}%")

# Info note (always visible)
st.info("Waiting for eBay Growth Check approval. Live API calls will replace demo/CSV here when enabled.")

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












