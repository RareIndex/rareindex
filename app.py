import glob, re
import streamlit as st
import pandas as pd
import numpy as np
import feedparser
from utils.validation import validate_timeseries_csv
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

# --- Helper: Slice dataframe by date range (3M / 6M / 1Y / 2Y / YTD / All) ---
def slice_by_range(df: pd.DataFrame, range_key: str) -> pd.DataFrame:
    """
    Returns a sliced df by range:
    '3M','6M','1Y','2Y','YTD','All'
    Assumes df has a 'date' column (datetime) and is sorted.
    """
    if df.empty:
        return df.copy()

    end = df["date"].max()

    if range_key == "All":
        return df.copy()
    if range_key == "YTD":
        start = pd.Timestamp(year=end.year, month=1, day=1)
        return df[df["date"] >= start].copy()

    months_map = {"3M": 3, "6M": 6, "1Y": 12, "2Y": 24}
    if range_key in months_map:
        months = months_map[range_key]
        start = end - pd.DateOffset(months=months-1)
        return df[df["date"] >= start].copy()

    # Fallback
    return df.copy()

# --- Helper: build leaderboard for toys across a chosen window ---
def build_toy_leaderboard(df_all: pd.DataFrame, period: str) -> pd.DataFrame:
    """
    period: one of '3M','6M','1Y','2Y','YTD','All'
    Returns a DataFrame with per-item ROI (and CAGR when possible) for the chosen window.
    """
    rows = []
    # Ensure types
    df_all = df_all.copy()
    df_all["date"] = pd.to_datetime(df_all["date"])
    df_all = df_all.sort_values(["item_name", "date"])

    for name, g in df_all.groupby("item_name", sort=True):
        g = g.copy().sort_values("date")

        # keep some metadata from the latest row (nice for display)
        meta = g.iloc[-1]
        subtype = str(meta.get("category_subtype", "")) or "—"
        cond = str(meta.get("condition", "")) or "—"
        grade = str(meta.get("grade", "")) or "—"
        rel_year = meta.get("release_year", None)

        g_use = slice_by_range(g, period)
        if len(g_use) < 2:
            continue

        start_price = float(g_use["price_usd"].iloc[0])
        latest_price = float(g_use["price_usd"].iloc[-1])
        if start_price <= 0:
            continue

        roi_pct = ((latest_price - start_price) / start_price) * 100.0

        # CAGR if we have a real span
        span_years = (g_use["date"].iloc[-1] - g_use["date"].iloc[0]).days / 365.25
        if span_years > 0:
            cagr_pct = ((latest_price / start_price) ** (1.0 / span_years) - 1.0) * 100.0
        else:
            cagr_pct = None

        rows.append({
            "Item": name,
            "Subtype": subtype,
            "Condition": cond,
            "Grade": grade,
            "Release Year": int(rel_year) if pd.notnull(rel_year) else None,
            "Start ($)": start_price,
            "Latest ($)": latest_price,
            "ROI (%)": roi_pct,
            "CAGR (%)": cagr_pct,
        })

    if not rows:
        return pd.DataFrame(columns=["Item","Subtype","Condition","Grade","Release Year","Start ($)","Latest ($)","ROI (%)","CAGR (%)"])

    out = pd.DataFrame(rows)
    out = out.sort_values("ROI (%)", ascending=False).reset_index(drop=True)
    return out

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
    return fetch_news(feed_url, limit)

def render_news(category_name: str):
    st.markdown("<h5 style='margin-top:0.5rem;'>Trending News</h5>", unsafe_allow_html=True)
    feeds = FEEDS.get(category_name, [])
    combined = []
    with st.spinner("Loading news..."):
        for url in feeds:
            combined.extend(cached_fetch_news(url, limit=3))

    # Deduplicate by title (very simple)
    seen, cleaned = set(), []
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

    # Single loop to render
    for item in cleaned:
        pub = item.get("published", "")
        st.markdown(
            f"- [{item.get('title','Untitled')}]({item.get('link','#')})  \n"
            f"  <span style='color:#777;font-size:12px;'>{pub}</span>",
            unsafe_allow_html=True,
        )

# --- Helper to load and one category ---
def show_category(title, source):
    """
    source: either a CSV path (str) or a pandas DataFrame with 'date' and 'price_usd'
    """
    st.subheader(title)
    try:
        # Load from path or use provided DataFrame
        if isinstance(source, str):
            df = read_csv_cached(source)
        else:
            df = source.copy()

        # ---- Options
        with st.expander("Options", expanded=False):
            compare = st.checkbox("Compare to a stock index", value=False, key=f"cmp_{title}")
            market_choice = st.selectbox("Choose index", list(MARKETS.keys()), index=0, key=f"mkt_{title}")
            range_choice = st.radio(
                "Range",
                ["3M", "6M", "1Y", "2Y", "YTD", "All"],
                index=5,  # default to All
                key=f"rng_{title}",
                horizontal=True
            )

        # ---- Choose the slice we use for plotting + metrics
        df_use = slice_by_range(df, range_choice)
        if df_use.empty:
            st.warning("No data available for the selected range.")
            return

        # Add range suffix for chart titles
        title_suffix = f" — {range_choice}"

        # --- Basic ROI on the chosen slice
        start = float(df_use["price_usd"].iloc[0])
        end = float(df_use["price_usd"].iloc[-1])
        roi = ((end - start) / start) * 100.0
        start_label = f"Starting Price ({df_use['date'].iloc[0].strftime('%b %Y')})"

        if compare:
            item_index = (df_use["price_usd"] / df_use["price_usd"].iloc[0]) * 100.0
            market_ytd = MARKETS[market_choice]
            market_index = make_index_series(market_ytd, len(df_use))

            plot_df = pd.DataFrame(
                {title: item_index.values, market_choice: market_index},
                index=df_use["date"]
            )

            st.markdown(
                f"<h4 style='text-align:center;'>{title} — Index vs {market_choice}{title_suffix}</h4>",
                unsafe_allow_html=True
            )
            st.line_chart(plot_df)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric(start_label, f"${start:,.0f}")
            col2.metric("Latest Price", f"${end:,.0f}")
            col3.metric(f"ROI since {df_use['date'].iloc[0].strftime('%b %Y')}", f"{roi:.1f}%")
            item_last = float(item_index.iloc[-1])
            market_last = float(market_index[-1])
            col4.metric("Outperformance vs Index", f"{(item_last - market_last):+.1f} pp")
        else:
            st.markdown(
                f"<h4 style='text-align:center;'>{title} — Price Trend{title_suffix}</h4>",
                unsafe_allow_html=True
            )
            st.line_chart(df_use.set_index("date")["price_usd"])

            col1, col2, col3 = st.columns(3)
            col1.metric(start_label, f"${start:,.0f}")
            col2.metric("Latest Price", f"${end:,.0f}")
            col3.metric(f"ROI since {df_use['date'].iloc[0].strftime('%b %Y')}", f"{roi:.1f}%")

        # --- Recent data points (with item name + nicer formatting)
        recent = df_use.tail(5).copy()
        recent["Item"] = title
        recent["Date"] = recent["date"].dt.strftime("%Y-%m-%d")
        recent["Price ($)"] = recent["price_usd"].map(lambda v: f"${v:,.2f}")

        st.caption("Recent data points")
        st.dataframe(
            recent[["Item", "Date", "Price ($)"]],
            width="stretch"
        )

    except FileNotFoundError:
        st.warning(f"Could not find {source}. Make sure the file exists.")
    except Exception as e:
        st.error(f"Error loading {source}: {e}")

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

# --- Tabs: Cards, Watches, Toys, Live eBay, ROI, Top 10, Validator ---
tab_cards, tab_watches, tab_toys, tab_live, tab_roi, tab_top10, tab_validator = st.tabs(
    ["Cards", "Watches", "Toys", "Live eBay (beta)", "ROI Calculator", "Top 10 (Demo)", "Validator"]
)

# --- Validator Tab ---
with tab_validator:
    st.markdown("### ✅ CSV Validator")
    st.caption("Check your file before adding it to the repo. Required columns: **date, price_usd**. Dates should be monthly.")

    uploaded = st.file_uploader("Upload a CSV to validate", type=["csv"], key="validator_uploader")
    if uploaded is not None:
        result = validate_timeseries_csv(uploaded)
        level = result["level"]
        msgs = result["messages"]

        if level == "error":
            st.error("Validation failed.")
        elif level == "warn":
            st.warning("Validation warnings.")
        else:
            st.success("Validation OK.")

        for m in msgs:
            st.write("•", m)

        if result.get("df") is not None:
            st.markdown("**Preview (first & last 5 rows)**")
            df_clean = result["df"]
            st.dataframe(df_clean.head(5), width="stretch")
            st.dataframe(df_clean.tail(5), width="stretch")

# --- Top 10 ROI (Demo) ---
with tab_top10:
    st.markdown("### 🏆 Top 10 ROI (Demo)")

    # Auto-discover all card CSVs
    card_paths = sorted(glob.glob("data/cards/cards_*.csv"))
    card_items = []
    for p in card_paths:
        m = re.search(r"cards_(\d+)\.csv$", p)
        label = f"Card #{m.group(1)} (Demo)" if m else "Card (Demo)"
        card_items.append({"name": label, "category": "Cards", "csv": p})

    # Add non-card demo rows
    demo_items = card_items + [
        {"name": "Rolex Submariner 116610LN", "category": "Watches", "csv": "watches.csv"},
        {"name": "LEGO 75290 Mos Eisley Cantina", "category": "Toys",    "csv": "toys.csv"},
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
            width="stretch"
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
    st.markdown(
        "<p style='text-align:center; color:#555;'>Tracking monthly median resale for the top 50 trading cards by ROI (demo dataset).</p>",
        unsafe_allow_html=True
    )

    try:
        # Load the full 50-card dataset
        df_all_cards = read_csv_cached("data/cards/cards_top50.csv")
    except FileNotFoundError:
        st.error("Missing file: data/cards/cards_top50.csv — make sure it’s committed and pushed.")
    else:
        # Require columns we expect
        required_cols = {"item_name", "date", "price_usd"}
        if not required_cols.issubset(set(df_all_cards.columns)):
            st.error(f"'data/cards/cards_top50.csv' is missing required columns: {required_cols}")
        else:
            # Let the user pick a single card to visualize
            card_names = sorted(df_all_cards["item_name"].dropna().unique().tolist())
            choice_c = st.selectbox("Choose a card", card_names, index=0, key="card_picker")

            # --- Metadata badges (robust, no dropna filtering) ---
            meta_cols_c = [
                "release_year",
                "retirement_year",
                "condition",
                "grade",
                "category_subtype",
                "original_retail",
                "source_platform",
            ]

            # Normalize relevant string columns
            for c in ["item_name","condition","grade","category_subtype","source_platform"]:
                if c in df_all_cards.columns:
                    df_all_cards[c] = df_all_cards[c].astype(str).str.strip()

            missing_meta = [c for c in meta_cols_c if c not in df_all_cards.columns]
            if missing_meta:
                st.warning(f"Missing metadata columns: {missing_meta}")
            else:
                meta_row = (
                    df_all_cards.loc[df_all_cards["item_name"] == choice_c, meta_cols_c]
                    .head(1)  # no dropna(), show whatever we have
                )

                if not meta_row.empty:
                    m = meta_row.iloc[0]
                    rel = int(m["release_year"]) if pd.notnull(m["release_year"]) else "—"
                    ret = int(m["retirement_year"]) if pd.notnull(m["retirement_year"]) else "—"
                    cond = m["condition"] if pd.notnull(m["condition"]) else "—"
                    grade = m["grade"] if pd.notnull(m["grade"]) else "—"
                    subtype = m["category_subtype"] if pd.notnull(m["category_subtype"]) else "—"
                    retail_str = f"${float(m['original_retail']):,.2f}" if pd.notnull(m["original_retail"]) else "—"
                    source = m["source_platform"] if pd.notnull(m["source_platform"]) else "—"

                    st.markdown(
                        " ".join([
                            f"<span style='display:inline-block;padding:4px 10px;margin:0 6px 8px 0;border-radius:999px;background:#eef2ff;color:#1e40af;font-size:12px;'>Release: {rel}</span>",
                            f"<span style='display:inline-block;padding:4px 10px;margin:0 6px 8px 0;border-radius:999px;background:#eef2ff;color:#1e40af;font-size:12px;'>Retired: {ret}</span>",
                            f"<span style='display:inline-block;padding:4px 10px;margin:0 6px 8px 0;border-radius:999px;background:#ecfeff;color:#155e75;font-size:12px;'>Condition: {cond}</span>",
                            f"<span style='display:inline-block;padding:4px 10px;margin:0 6px 8px 0;border-radius:999px;background:#ecfeff;color:#155e75;font-size:12px;'>Grade: {grade}</span>",
                            f"<span style='display:inline-block;padding:4px 10px;margin:0 6px 8px 0;border-radius:999px;background:#f0fdf4;color:#166534;font-size:12px;'>Type: {subtype}</span>",
                            f"<span style='display:inline-block;padding:4px 10px;margin:0 6px 8px 0;border-radius:999px;background:#fff7ed;color:#9a3412;font-size:12px;'>Orig. Retail: {retail_str}</span>",
                            f"<span style='display:inline-block;padding:4px 10px;margin:0 6px 8px 0;border-radius:999px;background:#fdf4ff;color:#6b21a8;font-size:12px;'>Source: {source}</span>",
                        ]),
                        unsafe_allow_html=True
                    )
                else:
                    st.caption("No metadata found for this item.")

            # Filter to the chosen card and hand a tidy df to show_category()
            df_one_c = df_all_cards.loc[
                df_all_cards["item_name"] == choice_c, ["date", "price_usd"]
            ].copy()
            show_category(f"{choice_c} (Cards)", df_one_c)

            # --- ROI Leaderboard (Top 50 cards) ---
            st.markdown("### 🧮 ROI Leaderboard")

            lb_period_c = st.radio(
                "Window",
                ["3M", "6M", "1Y", "2Y", "YTD", "All"],
                index=2,  # default 1Y
                horizontal=True,
                key="cards_leaderboard_window",
            )

            # Re-use the same leaderboard builder (works on item_name/date/price_usd + metadata)
            df_lb_c = build_toy_leaderboard(df_all_cards, lb_period_c)

            # Optional quick search
            q_c = st.text_input("Search (item name contains…)", "", key="cards_lb_search")
            if q_c.strip():
                df_lb_c = df_lb_c[df_lb_c["Item"].str.contains(q_c.strip(), case=False, na=False)].copy()

            if df_lb_c.empty:
                st.info("No leaderboard rows for the selected window yet.")
            else:
                df_show_c = df_lb_c.copy()
                for col in ["Start ($)", "Latest ($)"]:
                    df_show_c[col] = df_show_c[col].apply(lambda v: f"${v:,.2f}" if pd.notnull(v) else "—")
                for col in ["ROI (%)", "CAGR (%)"]:
                    df_show_c[col] = df_show_c[col].apply(lambda v: f"{v:,.2f}%" if pd.notnull(v) else "—")

                st.dataframe(
                    df_show_c[["Item","Subtype","Condition","Grade","Release Year","Start ($)","Latest ($)","ROI (%)","CAGR (%)"]],
                    width="stretch",
                )

                csv_bytes_c = df_lb_c.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download leaderboard (CSV)",
                    data=csv_bytes_c,
                    file_name=f"cards_leaderboard_{lb_period_c}.csv",
                    mime="text/csv",
                )

    # News stays at the end of the tab
    render_news("Cards")

with tab_watches:
    st.markdown(
        "<p style='text-align:center; color:#555;'>Tracking monthly median resale for the top 50 watches by ROI (demo dataset).</p>",
        unsafe_allow_html=True
    )

    try:
        # Load the full 50-watch dataset
        df_all_watches = read_csv_cached("data/watches/watches_top50.csv")
    except FileNotFoundError:
        st.error("Missing file: data/watches/watches_top50.csv — make sure it’s committed and pushed.")
    else:
        # Require columns we expect
        required_cols = {"item_name", "date", "price_usd"}
        if not required_cols.issubset(set(df_all_watches.columns)):
            st.error(f"'data/watches/watches_top50.csv' is missing required columns: {required_cols}")
        else:
            # Let the user pick a single watch to visualize
            watch_names = sorted(df_all_watches["item_name"].dropna().unique().tolist())
            choice_w = st.selectbox("Choose a watch", watch_names, index=0, key="watch_picker")

            # --- Show metadata for the selected watch ---
            meta_cols = [
                "release_year",
                "retirement_year",
                "condition",
                "grade",
                "category_subtype",
                "original_retail",
                "source_platform",
            ]

            # Normalize strings to avoid mismatches
            for c in ["item_name", "condition", "grade", "category_subtype", "source_platform"]:
                if c in df_all_watches.columns:
                    df_all_watches[c] = df_all_watches[c].astype(str).str.strip()

            missing = [c for c in meta_cols if c not in df_all_watches.columns]
            if missing:
                st.warning(f"Missing metadata columns: {missing}")
            else:
                meta_row = (
                    df_all_watches.loc[df_all_watches["item_name"] == choice_w, meta_cols]
                    .head(1)  # no dropna() so we still show partial badges
                )

                if not meta_row.empty:
                    m = meta_row.iloc[0]
                    rel = int(m["release_year"]) if pd.notnull(m["release_year"]) else "—"
                    ret = int(m["retirement_year"]) if pd.notnull(m["retirement_year"]) else "—"
                    cond = m["condition"] if pd.notnull(m["condition"]) else "—"
                    grade = m["grade"] if pd.notnull(m["grade"]) else "—"
                    subtype = m["category_subtype"] if pd.notnull(m["category_subtype"]) else "—"
                    retail_str = f"${float(m['original_retail']):,.2f}" if pd.notnull(m["original_retail"]) else "—"
                    source = m["source_platform"] if pd.notnull(m["source_platform"]) else "—"

                    st.markdown(
                        " ".join([
                            f"<span style='display:inline-block;padding:4px 10px;margin:0 6px 8px 0;border-radius:999px;background:#eef2ff;color:#1e40af;font-size:12px;'>Release: {rel}</span>",
                            f"<span style='display:inline-block;padding:4px 10px;margin:0 6px 8px 0;border-radius:999px;background:#eef2ff;color:#1e40af;font-size:12px;'>Retired: {ret}</span>",
                            f"<span style='display:inline-block;padding:4px 10px;margin:0 6px 8px 0;border-radius:999px;background:#ecfeff;color:#155e75;font-size:12px;'>Condition: {cond}</span>",
                            f"<span style='display:inline-block;padding:4px 10px;margin:0 6px 8px 0;border-radius:999px;background:#ecfeff;color:#155e75;font-size:12px;'>Grade: {grade}</span>",
                            f"<span style='display:inline-block;padding:4px 10px;margin:0 6px 8px 0;border-radius:999px;background:#f0fdf4;color:#166534;font-size:12px;'>Type: {subtype}</span>",
                            f"<span style='display:inline-block;padding:4px 10px;margin:0 6px 8px 0;border-radius:999px;background:#fff7ed;color:#9a3412;font-size:12px;'>Orig. Retail: {retail_str}</span>",
                            f"<span style='display:inline-block;padding:4px 10px;margin:0 6px 8px 0;border-radius:999px;background:#fdf4ff;color:#6b21a8;font-size:12px;'>Source: {source}</span>",
                        ]),
                        unsafe_allow_html=True
                    )
                else:
                    st.caption("No metadata found for this item.")

            # Filter to the chosen watch and pass a tidy df to show_category()
            df_one_w = df_all_watches.loc[
                df_all_watches["item_name"] == choice_w, ["date", "price_usd"]
            ].copy()
            show_category(f"{choice_w} (Watches)", df_one_w)

            # --- ROI Leaderboard (Top 50 watches) ---
            st.markdown("### 🧮 ROI Leaderboard")

            lb_period_w = st.radio(
                "Window",
                ["3M", "6M", "1Y", "2Y", "YTD", "All"],
                index=2,  # default 1Y
                horizontal=True,
                key="watches_leaderboard_window",
            )

            # Re-use the generic leaderboard builder (works for any item_name/date/price_usd dataset)
            df_lb_w = build_toy_leaderboard(df_all_watches, lb_period_w)

            # Optional quick search
            q_w = st.text_input("Search (item name contains…)", "", key="watches_lb_search")
            if q_w.strip():
                df_lb_w = df_lb_w[df_lb_w["Item"].str.contains(q_w.strip(), case=False, na=False)].copy()

            if df_lb_w.empty:
                st.info("No leaderboard rows for the selected window yet.")
            else:
                df_show_w = df_lb_w.copy()
                for col in ["Start ($)", "Latest ($)"]:
                    df_show_w[col] = df_show_w[col].apply(lambda v: f"${v:,.2f}" if pd.notnull(v) else "—")
                for col in ["ROI (%)", "CAGR (%)"]:
                    df_show_w[col] = df_show_w[col].apply(lambda v: f"{v:,.2f}%" if pd.notnull(v) else "—")

                st.dataframe(
                    df_show_w[["Item","Subtype","Condition","Grade","Release Year","Start ($)","Latest ($)","ROI (%)","CAGR (%)"]],
                    width="stretch",
                )

                # Download raw (numeric) leaderboard as CSV
                csv_bytes_w = df_lb_w.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download leaderboard (CSV)",
                    data=csv_bytes_w,
                    file_name=f"watches_leaderboard_{lb_period_w}.csv",
                    mime="text/csv",
                )

    # News stays at the end of the tab
    render_news("Watches")

with tab_toys:
    st.markdown(
        "<p style='text-align:center; color:#555;'>Tracking monthly median resale for the top 50 collectible toys by ROI (demo dataset).</p>",
        unsafe_allow_html=True
    )

    try:
        # Load the full 50-toy dataset
        df_all_toys = read_csv_cached("data/toys/toys_top50.csv")
    except FileNotFoundError:
        st.error("Missing file: data/toys/toys_top50.csv — make sure it’s committed and pushed.")
    else:
        # Require columns we expect
        required_cols = {"item_name", "date", "price_usd"}
        if not required_cols.issubset(set(df_all_toys.columns)):
            st.error(f"'data/toys/toys_top50.csv' is missing required columns: {required_cols}")
        else:
            # ---- Single-item chart section ----
            toy_names = sorted(df_all_toys["item_name"].dropna().unique().tolist())
            choice = st.selectbox("Choose a toy", toy_names, index=0, key="toy_picker")
            # --- 📈 Category Summary (Toys) ---
            st.markdown("### 📈 Category Summary")

            sum_period = st.radio(
                "Summary window",
                ["3M", "6M", "1Y", "2Y", "YTD", "All"],
                index=2,  # default 1Y
                horizontal=True,
                key="toys_summary_window",
            )

            df_lb_sum = build_toy_leaderboard(df_all_toys, sum_period)

            if df_lb_sum.empty:
                st.info("No summary available for this window.")
            else:
                # KPIs
                n_items = int(len(df_lb_sum))
                avg_roi = float(df_lb_sum["ROI (%)"].mean())
                med_roi = float(df_lb_sum["ROI (%)"].median())
                top_row = df_lb_sum.iloc[0]
                bottom_row = df_lb_sum.iloc[-1]

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Items", f"{n_items}")
                c2.metric("Avg ROI", f"{avg_roi:,.2f}%")
                c3.metric("Median ROI", f"{med_roi:,.2f}%")
                c4.metric("Top Item ROI", f"{top_row['ROI (%)']:,.2f}%")

                # Top / Bottom tables (small)
                st.markdown("#### Top & Bottom performers")
                top3 = df_lb_sum.head(3).copy()
                bot3 = df_lb_sum.tail(3).copy()

                # Pretty format for display
                for df_show in (top3, bot3):
                    df_show["Start ($)"]  = df_show["Start ($)"].apply(lambda v: f"${v:,.2f}" if pd.notnull(v) else "—")
                    df_show["Latest ($)"] = df_show["Latest ($)"].apply(lambda v: f"${v:,.2f}" if pd.notnull(v) else "—")
                    df_show["ROI (%)"]    = df_show["ROI (%)"].apply(lambda v: f"{v:,.2f}%" if pd.notnull(v) else "—")
                    df_show["CAGR (%)"]   = df_show["CAGR (%)"].apply(lambda v: f"{v:,.2f}%" if pd.notnull(v) else "—")

                colA, colB = st.columns(2)
                with colA:
                    st.caption("Top 3 (by ROI)")
                    st.dataframe(
                        top3[["Item","Subtype","ROI (%)","CAGR (%)","Start ($)","Latest ($)"]],
                        width="stretch",
                    )
                with colB:
                    st.caption("Bottom 3 (by ROI)")
                    st.dataframe(
                        bot3[["Item","Subtype","ROI (%)","CAGR (%)","Start ($)","Latest ($)"]],
                        width="stretch",
                    )

            # Show metadata for the selected toy
            meta_cols = [
                "release_year",
                "retirement_year",
                "condition",
                "grade",
                "category_subtype",
                "original_retail",
                "source_platform",
            ]
            meta_row = (
                df_all_toys.loc[df_all_toys["item_name"] == choice, meta_cols]
                .dropna()
                .head(1)
            )

            if not meta_row.empty:
                m = meta_row.iloc[0]
                retail_str = f"${float(m['original_retail']):,.2f}" if pd.notnull(m["original_retail"]) else "—"
                st.markdown(
                    " ".join([
                        f"<span style='display:inline-block;padding:4px 10px;margin:0 6px 8px 0;border-radius:999px;background:#eef2ff;color:#1e40af;font-size:12px;'>Release: {int(m['release_year']) if pd.notnull(m['release_year']) else '—'}</span>",
                        f"<span style='display:inline-block;padding:4px 10px;margin:0 6px 8px 0;border-radius:999px;background:#eef2ff;color:#1e40af;font-size:12px;'>Retired: {int(m['retirement_year']) if pd.notnull(m['retirement_year']) else '—'}</span>",
                        f"<span style='display:inline-block;padding:4px 10px;margin:0 6px 8px 0;border-radius:999px;background:#ecfeff;color:#155e75;font-size:12px;'>Condition: {m['condition'] if pd.notnull(m['condition']) else '—'}</span>",
                        f"<span style='display:inline-block;padding:4px 10px;margin:0 6px 8px 0;border-radius:999px;background:#ecfeff;color:#155e75;font-size:12px;'>Grade: {m['grade'] if pd.notnull(m['grade']) else '—'}</span>",
                        f"<span style='display:inline-block;padding:4px 10px;margin:0 6px 8px 0;border-radius:999px;background:#f0fdf4;color:#166534;font-size:12px;'>Type: {m['category_subtype'] if pd.notnull(m['category_subtype']) else '—'}</span>",
                        f"<span style='display:inline-block;padding:4px 10px;margin:0 6px 8px 0;border-radius:999px;background:#fff7ed;color:#9a3412;font-size:12px;'>Orig. Retail: {retail_str}</span>",
                        f"<span style='display:inline-block;padding:4px 10px;margin:0 6px 8px 0;border-radius:999px;background:#fdf4ff;color:#6b21a8;font-size:12px;'>Source: {m['source_platform'] if pd.notnull(m['source_platform']) else '—'}</span>",
                    ]),
                    unsafe_allow_html=True
                )
            else:
                st.caption("No metadata found for this item.")

            # Filter to the chosen item and pass a tidy df to show_category()
            df_one = df_all_toys.loc[
                df_all_toys["item_name"] == choice, ["date", "price_usd"]
            ].copy()
            show_category(f"{choice} (Toys)", df_one)

            # --- ROI Leaderboard (Top 50 toys) ---
    st.markdown("### 🧮 ROI Leaderboard")

    # Window selector
    lb_period = st.radio(
        "Window",
        ["3M", "6M", "1Y", "2Y", "YTD", "All"],
        index=2,  # default 1Y
        horizontal=True,
        key="toys_leaderboard_window",
    )

    # Build raw leaderboard (numeric)
    df_lb = build_toy_leaderboard(df_all_toys, lb_period)

    # ---- Filters (Subtype / Condition / Grade / Release Year) ----
    # Build option lists from the full toys dataset (more complete than df_lb)
    subtypes = sorted(df_all_toys["category_subtype"].dropna().astype(str).unique().tolist())
    conditions = sorted(df_all_toys["condition"].dropna().astype(str).unique().tolist())
    grades = sorted(df_all_toys["grade"].dropna().astype(str).unique().tolist())
    years_all = df_all_toys["release_year"].dropna().astype(int)
    yr_min, yr_max = (int(years_all.min()), int(years_all.max())) if not years_all.empty else (1990, 2025)

    with st.expander("Filters", expanded=False):
        sel_subtypes = st.multiselect("Subtype", options=subtypes, default=[], key="lb_f_subtype")
        sel_conditions = st.multiselect("Condition", options=conditions, default=[], key="lb_f_condition")
        sel_grades = st.multiselect("Grade", options=grades, default=[], key="lb_f_grade")
        yr_range = st.slider("Release year range", min_value=yr_min, max_value=yr_max, value=(yr_min, yr_max), step=1, key="lb_f_years")
        top_k = st.slider("Show top N", min_value=10, max_value=50, value=25, step=5, key="lb_topk")

    # Apply filters to leaderboard (map to df_lb column names)
    if sel_subtypes:
        df_lb = df_lb[df_lb["Subtype"].isin(sel_subtypes)].copy()
    if sel_conditions:
        df_lb = df_lb[df_lb["Condition"].isin(sel_conditions)].copy()
    if sel_grades:
        df_lb = df_lb[df_lb["Grade"].isin(sel_grades)].copy()
    df_lb = df_lb[df_lb["Release Year"].between(yr_range[0], yr_range[1], inclusive="both") | df_lb["Release Year"].isna()].copy()

    # ---- Sorting controls ----
    sort_col = st.selectbox(
        "Sort by",
        ["ROI (%)", "CAGR (%)", "Latest ($)", "Start ($)"],
        index=0,
        key="lb_sort_col",
    )
    sort_dir = st.radio("Order", ["Desc", "Asc"], index=0, horizontal=True, key="lb_sort_dir")
    df_lb = df_lb.sort_values(sort_col, ascending=(sort_dir == "Asc"), na_position="last")

    # Limit to top N
    df_lb = df_lb.head(top_k).reset_index(drop=True)

    # Optional quick search by name
    q = st.text_input("Search (item name contains…)", "", key="toys_lb_search")
    if q.strip():
        df_lb = df_lb[df_lb["Item"].str.contains(q.strip(), case=False, na=False)].copy()

    if df_lb.empty:
        st.info("No leaderboard rows for the selected window/filters yet.")
    else:
        # Pretty display columns
        df_show = df_lb.copy()
        for col in ["Start ($)", "Latest ($)"]:
            df_show[col] = df_show[col].apply(lambda v: f"${v:,.2f}" if pd.notnull(v) else "—")
        for col in ["ROI (%)", "CAGR (%)"]:
            df_show[col] = df_show[col].apply(lambda v: f"{v:,.2f}%" if pd.notnull(v) else "—")

        st.dataframe(
            df_show[["Item","Subtype","Condition","Grade","Release Year","Start ($)","Latest ($)","ROI (%)","CAGR (%)"]],
            width="stretch",
        )

        # Download raw (numeric) leaderboard as CSV
        csv_bytes = df_lb.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download leaderboard (CSV)",
            data=csv_bytes,
            file_name=f"toys_leaderboard_{lb_period}.csv",
            mime="text/csv",
        )

    # News stays at the end of the tab
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
            width="stretch"
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



























