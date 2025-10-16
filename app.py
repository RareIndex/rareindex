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
def build_item_leaderboard(df_all: pd.DataFrame, period: str) -> pd.DataFrame:
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
        "<p style='text-align:center; color:#555;'>Tracking monthly median resale for top cards by ROI (demo datasets).</p>",
        unsafe_allow_html=True
    )

    # Try category dataset first
    have_category = True
    try:
        df_all_cards = read_csv_cached("data/cards/cards_top50.csv")
    except FileNotFoundError:
        have_category = False

    if have_category:
        df_all_cards = df_all_cards.copy()
        df_all_cards["date"] = pd.to_datetime(df_all_cards["date"])

        # ---------- Category Summary ----------
        st.markdown("### 🧭 Category Summary")
        lb_period_c = st.radio(
            "Summary window",
            ["3M", "6M", "1Y", "2Y", "YTD", "All"],
            index=2,  # default 1Y
            horizontal=True,
            key="cards_summary_window",
        )

        df_sum_c = slice_by_range(df_all_cards, lb_period_c)
        if df_sum_c.empty:
            st.info("No data in selected window.")
        else:
            df_lb_c = build_toy_leaderboard(df_sum_c, "All")
            items_ct = df_lb_c.shape[0]
            avg_roi = df_lb_c["ROI (%)"].mean() if items_ct else None
            med_roi = df_lb_c["ROI (%)"].median() if items_ct else None
            top_roi = df_lb_c["ROI (%)"].max() if items_ct else None

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Items", f"{items_ct:,}")
            c2.metric("Avg ROI", f"{avg_roi:.2f}%" if avg_roi is not None else "—")
            c3.metric("Median ROI", f"{med_roi:.2f}%" if med_roi is not None else "—")
            c4.metric("Top item ROI", f"{top_roi:.2f}%" if top_roi is not None else "—")

            # ---------- Category vs Benchmark ----------
            st.markdown("### 📈 Category vs Benchmark")
            benchmark_c = st.selectbox(
                "Choose benchmark",
                list(MARKETS.keys()),
                index=0,
                key="cards_bench",
            )

            df_norm_c = df_sum_c.sort_values(["item_name", "date"]).copy()
            df_norm_c["first_in_win"] = df_norm_c.groupby("item_name")["price_usd"].transform("first")
            df_norm_c = df_norm_c[df_norm_c["first_in_win"] > 0].copy()
            df_norm_c["item_idx"] = (df_norm_c["price_usd"] / df_norm_c["first_in_win"]) * 100.0
            cat_series_c = df_norm_c.groupby("date")["item_idx"].mean().sort_index()

            bench_ytd_c = MARKETS[benchmark_c]
            bench_series_c = pd.Series(
                make_index_series(bench_ytd_c, len(cat_series_c)),
                index=cat_series_c.index,
                name=benchmark_c
            )

            chart_df_c = pd.DataFrame({
                "Cards (Category Index)": cat_series_c,
                benchmark_c: bench_series_c.values
            })
            st.line_chart(chart_df_c)

            # ---------- Top & Bottom performers ----------
            st.markdown("### Top & Bottom performers")
            top3c = df_lb_c.head(3).copy()
            bot3c = df_lb_c.tail(3).copy()

            for _df in (top3c, bot3c):
                _df["Start ($)"]  = _df["Start ($)"].apply(lambda v: f"${v:,.2f}" if pd.notnull(v) else "—")
                _df["Latest ($)"] = _df["Latest ($)"].apply(lambda v: f"${v:,.2f}" if pd.notnull(v) else "—")
                _df["ROI (%)"]    = _df["ROI (%)"].apply(lambda v: f"{v:,.2f}%" if pd.notnull(v) else "—")
                _df["CAGR (%)"]   = _df["CAGR (%)"].apply(lambda v: f"{v:,.2f}%" if pd.notnull(v) else "—")

            lcol, rcol = st.columns(2)
            with lcol:
                st.caption("Top 3 by ROI")
                st.dataframe(
                    top3c[["Item","Subtype","Condition","Grade","Release Year","Start ($)","Latest ($)","ROI (%)","CAGR (%)"]],
                    width="stretch"
                )
            with rcol:
                st.caption("Bottom 3 by ROI")
                st.dataframe(
                    bot3c[["Item","Subtype","Condition","Grade","Release Year","Start ($)","Latest ($)","ROI (%)","CAGR (%)"]],
                    width="stretch"
                )

        st.markdown("---")
        st.markdown("### Individual Item")

        card_names = sorted(df_all_cards["item_name"].dropna().unique().tolist())
        choice_c = st.selectbox("Choose a card", card_names, index=0, key="card_picker")

        meta_cols = [
            "release_year","retirement_year","condition","grade",
            "category_subtype","original_retail","source_platform",
        ]
        meta_row = df_all_cards.loc[df_all_cards["item_name"] == choice_c, meta_cols].head(1)
        if not meta_row.empty:
            m = meta_row.iloc[0]
            retail_str = f"${float(m['original_retail']):,.2f}" if pd.notnull(m["original_retail"]) else "—"
            rel = int(m["release_year"]) if pd.notnull(m["release_year"]) else "—"
            ret = int(m["retirement_year"]) if pd.notnull(m["retirement_year"]) else "—"
            cond = m["condition"] if pd.notnull(m["condition"]) else "—"
            grade = m["grade"] if pd.notnull(m["grade"]) else "—"
            subtype = m["category_subtype"] if pd.notnull(m["category_subtype"]) else "—"
            src = m["source_platform"] if pd.notnull(m["source_platform"]) else "—"

            st.markdown(
                " ".join([
                    f"<span style='display:inline-block;padding:4px 10px;margin:0 6px 8px 0;border-radius:999px;background:#eef2ff;color:#1e40af;font-size:12px;'>Release: {rel}</span>",
                    f"<span style='display:inline-block;padding:4px 10px;margin:0 6px 8px 0;border-radius:999px;background:#eef2ff;color:#1e40af;font-size:12px;'>Retired: {ret}</span>",
                    f"<span style='display:inline-block;padding:4px 10px;margin:0 6px 8px 0;border-radius:999px;background:#ecfeff;color:#155e75;font-size:12px;'>Condition: {cond}</span>",
                    f"<span style='display:inline-block;padding:4px 10px;margin:0 6px 8px 0;border-radius:999px;background:#ecfeff;color:#155e75;font-size:12px;'>Grade: {grade}</span>",
                    f"<span style='display:inline-block;padding:4px 10px;margin:0 6px 8px 0;border-radius:999px;background:#f0fdf4;color:#166534;font-size:12px;'>Type: {subtype}</span>",
                    f"<span style='display:inline-block;padding:4px 10px;margin:0 6px 8px 0;border-radius:999px;background:#fff7ed;color:#9a3412;font-size:12px;'>Orig. Retail: {retail_str}</span>",
                    f"<span style='display:inline-block;padding:4px 10px;margin:0 6px 8px 0;border-radius:999px;background:#fdf4ff;color:#6b21a8;font-size:12px;'>Source: {src}</span>",
                ]),
                unsafe_allow_html=True
            )
        else:
            st.caption("No metadata found for this item.")

        df_one_c = df_all_cards.loc[df_all_cards["item_name"] == choice_c, ["date", "price_usd"]].copy()
        show_category(f"{choice_c} (Cards)", df_one_c)

        render_news("Cards")

    else:
        # ---- Fallback: single demo card path you already had ----
        st.caption("No category dataset found yet — showing single demo card.")
        show_category("Pokémon Card #011 (Cards)", "data/cards/cards_011.csv")
        render_news("Cards")

with tab_watches:
    st.markdown(
        "<p style='text-align:center; color:#555;'>Tracking monthly median resale for the top 50 watches by ROI (demo dataset).</p>",
        unsafe_allow_html=True
    )

    # ---- Load dataset
    try:
        df_all_watches = read_csv_cached("data/watches/watches_top50.csv")
    except FileNotFoundError:
        st.error("Missing file: data/watches/watches_top50.csv — make sure it’s committed and pushed.")
        render_news("Watches")
    else:
        # Ensure types
        df_all_watches = df_all_watches.copy()
        df_all_watches["date"] = pd.to_datetime(df_all_watches["date"])

        # ---------- Category Summary ----------
        st.markdown("### 🧭 Category Summary")

        # Window selector (same as Toys)
        lb_period_w = st.radio(
            "Summary window",
            ["3M", "6M", "1Y", "2Y", "YTD", "All"],
            index=2,  # default 1Y
            horizontal=True,
            key="watches_summary_window",
        )

        df_sum = slice_by_range(df_all_watches, lb_period_w)
        if df_sum.empty:
            st.info("No data in selected window.")
        else:
            # Per-item ROI
            df_lb_w = build_toy_leaderboard(df_sum, "All")  # already sliced; treat as All inside
            items_ct = df_lb_w.shape[0]
            avg_roi = df_lb_w["ROI (%)"].mean() if items_ct else None
            med_roi = df_lb_w["ROI (%)"].median() if items_ct else None
            top_roi = df_lb_w["ROI (%)"].max() if items_ct else None

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Items", f"{items_ct:,}")
            c2.metric("Avg ROI", f"{avg_roi:.2f}%" if avg_roi is not None else "—")
            c3.metric("Median ROI", f"{med_roi:.2f}%" if med_roi is not None else "—")
            c4.metric("Top item ROI", f"{top_roi:.2f}%" if top_roi is not None else "—")

            # ---------- Category vs Benchmark ----------
            st.markdown("### 📈 Category vs Benchmark")

            benchmark = st.selectbox(
                "Choose benchmark",
                list(MARKETS.keys()),
                index=0,
                key="watch_bench",
            )

            # Build category index: for each item, normalize to first value in window, then average
            df_norm = df_sum.sort_values(["item_name", "date"]).copy()
            df_norm["first_in_win"] = df_norm.groupby("item_name")["price_usd"].transform("first")
            df_norm = df_norm[df_norm["first_in_win"] > 0].copy()
            df_norm["item_idx"] = (df_norm["price_usd"] / df_norm["first_in_win"]) * 100.0
            cat_series = df_norm.groupby("date")["item_idx"].mean().sort_index()

            # Benchmark synthetic series sized to same number of points as dates we have
            bench_ytd = MARKETS[benchmark]
            bench_series = pd.Series(
                make_index_series(bench_ytd, len(cat_series)),
                index=cat_series.index,
                name=benchmark
            )

            chart_df = pd.DataFrame({
                "Watches (Category Index)": cat_series,
                benchmark: bench_series.values
            })

            st.line_chart(chart_df)

            # ---------- Top & Bottom performers ----------
            st.markdown("### Top & Bottom performers")
            top3 = df_lb_w.head(3).copy()
            bot3 = df_lb_w.tail(3).copy()

            # Pretty columns
            for _df in (top3, bot3):
                _df["Start ($)"]  = _df["Start ($)"].apply(lambda v: f"${v:,.2f}" if pd.notnull(v) else "—")
                _df["Latest ($)"] = _df["Latest ($)"].apply(lambda v: f"${v:,.2f}" if pd.notnull(v) else "—")
                _df["ROI (%)"]    = _df["ROI (%)"].apply(lambda v: f"{v:,.2f}%" if pd.notnull(v) else "—")
                _df["CAGR (%)"]   = _df["CAGR (%)"].apply(lambda v: f"{v:,.2f}%" if pd.notnull(v) else "—")

            lcol, rcol = st.columns(2)
            with lcol:
                st.caption("Top 3 by ROI")
                st.dataframe(
                    top3[["Item","Subtype","Condition","Grade","Release Year","Start ($)","Latest ($)","ROI (%)","CAGR (%)"]],
                    width="stretch"
                )
            with rcol:
                st.caption("Bottom 3 by ROI")
                st.dataframe(
                    bot3[["Item","Subtype","Condition","Grade","Release Year","Start ($)","Latest ($)","ROI (%)","CAGR (%)"]],
                    width="stretch"
                )

        # ---------- Per-item drilldown ----------
        st.markdown("---")
        st.markdown("### Individual Item")

        watch_names = sorted(df_all_watches["item_name"].dropna().unique().tolist())
        choice_w = st.selectbox("Choose a watch", watch_names, index=0, key="watch_picker")

        # Metadata badges
        meta_cols = [
            "release_year","retirement_year","condition","grade",
            "category_subtype","original_retail","source_platform",
        ]
        meta_row = df_all_watches.loc[df_all_watches["item_name"] == choice_w, meta_cols].head(1)
        if not meta_row.empty:
            m = meta_row.iloc[0]
            retail_str = f"${float(m['original_retail']):,.2f}" if pd.notnull(m["original_retail"]) else "—"
            rel = int(m["release_year"]) if pd.notnull(m["release_year"]) else "—"
            ret = int(m["retirement_year"]) if pd.notnull(m["retirement_year"]) else "—"
            cond = m["condition"] if pd.notnull(m["condition"]) else "—"
            grade = m["grade"] if pd.notnull(m["grade"]) else "—"
            subtype = m["category_subtype"] if pd.notnull(m["category_subtype"]) else "—"
            src = m["source_platform"] if pd.notnull(m["source_platform"]) else "—"

            st.markdown(
                " ".join([
                    f"<span style='display:inline-block;padding:4px 10px;margin:0 6px 8px 0;border-radius:999px;background:#eef2ff;color:#1e40af;font-size:12px;'>Release: {rel}</span>",
                    f"<span style='display:inline-block;padding:4px 10px;margin:0 6px 8px 0;border-radius:999px;background:#eef2ff;color:#1e40af;font-size:12px;'>Retired: {ret}</span>",
                    f"<span style='display:inline-block;padding:4px 10px;margin:0 6px 8px 0;border-radius:999px;background:#ecfeff;color:#155e75;font-size:12px;'>Condition: {cond}</span>",
                    f"<span style='display:inline-block;padding:4px 10px;margin:0 6px 8px 0;border-radius:999px;background:#ecfeff;color:#155e75;font-size:12px;'>Grade: {grade}</span>",
                    f"<span style='display:inline-block;padding:4px 10px;margin:0 6px 8px 0;border-radius:999px;background:#f0fdf4;color:#166534;font-size:12px;'>Type: {subtype}</span>",
                    f"<span style='display:inline-block;padding:4px 10px;margin:0 6px 8px 0;border-radius:999px;background:#fff7ed;color:#9a3412;font-size:12px;'>Orig. Retail: {retail_str}</span>",
                    f"<span style='display:inline-block;padding:4px 10px;margin:0 6px 8px 0;border-radius:999px;background:#fdf4ff;color:#6b21a8;font-size:12px;'>Source: {src}</span>",
                ]),
                unsafe_allow_html=True
            )
        else:
            st.caption("No metadata found for this item.")

        # Filter → show line chart via existing helper
        df_one_w = df_all_watches.loc[df_all_watches["item_name"] == choice_w, ["date", "price_usd"]].copy()
        show_category(f"{choice_w} (Watches)", df_one_w)

        # News
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
            # -----------------------------
            # CATEGORY SUMMARY (global view)
            # -----------------------------
            st.subheader("Category Summary")

            # Window selector for ALL category metrics
            lb_period = st.radio(
                "Summary window",
                ["3M", "6M", "1Y", "2Y", "YTD", "All"],
                index=2,  # default 1Y
                horizontal=True,
                key="toys_summary_window",
            )

            # Build leaderboard (per-item ROI) for the chosen window
            df_lb = build_item_leaderboard(df_all_toys, lb_period)
            df_lb_w = build_item_leaderboard(df_sum, "All")
            df_lb_c = build_item_leaderboard(df_sum_c, "All")

            # KPIs
            num_items = df_lb["Item"].nunique() if not df_lb.empty else 0
            avg_roi = df_lb["ROI (%)"].mean() if not df_lb.empty else np.nan
            med_roi = df_lb["ROI (%)"].median() if not df_lb.empty else np.nan
            top_roi = df_lb["ROI (%)"].max() if not df_lb.empty else np.nan

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Items", f"{num_items}")
            c2.metric("Avg ROI", f"{avg_roi:,.2f}%" if pd.notnull(avg_roi) else "—")
            c3.metric("Median ROI", f"{med_roi:,.2f}%" if pd.notnull(med_roi) else "—")
            c4.metric("Top item ROI", f"{top_roi:,.2f}%" if pd.notnull(top_roi) else "—")

            # Top & Bottom performers (tables)
            st.markdown("### Top & Bottom performers")
            if df_lb.empty:
                st.info("No leaderboard rows for the selected window yet.")
            else:
                top3 = df_lb.head(3).copy()
                bottom3 = df_lb.tail(3).copy()

                for col in ["Start ($)", "Latest ($)"]:
                    top3[col] = top3[col].apply(lambda v: f"${v:,.2f}" if pd.notnull(v) else "—")
                    bottom3[col] = bottom3[col].apply(lambda v: f"${v:,.2f}" if pd.notnull(v) else "—")
                for col in ["ROI (%)", "CAGR (%)"]:
                    top3[col] = top3[col].apply(lambda v: f"{v:,.2f}%" if pd.notnull(v) else "—")
                    bottom3[col] = bottom3[col].apply(lambda v: f"{v:,.2f}%" if pd.notnull(v) else "—")

                cL, cR = st.columns(2)
                with cL:
                    st.caption("Top 3 by ROI")
                    st.dataframe(
                        top3[["Item","Subtype","ROI (%)","CAGR (%)","Start ($)","Latest ($)"]],
                        width="stretch",
                    )
                with cR:
                    st.caption("Bottom 3 by ROI")
                    st.dataframe(
                        bottom3[["Item","Subtype","ROI (%)","CAGR (%)","Start ($)","Latest ($)"]],
                        width="stretch",
                    )

            # -----------------------------
            # NEW: Category vs Benchmark
            # -----------------------------
            st.markdown("### Category vs Benchmark")

            bench_key = st.selectbox(
                "Benchmark",
                list(MARKETS.keys()),
                index=0,
                key="toys_benchmark_choice",
            )

            # Aggregate ALL toys -> average monthly price
            ts = (
                df_all_toys.groupby("date", as_index=False)["price_usd"]
                .mean()
                .rename(columns={"price_usd": "avg_price"})
                .sort_values("date")
            )
            ts_use = slice_by_range(ts, lb_period)

            if len(ts_use) >= 2:
                cat_index = (ts_use["avg_price"] / ts_use["avg_price"].iloc[0]) * 100.0
                bench_index = make_index_series(MARKETS[bench_key], len(ts_use))

                plot_df = pd.DataFrame(
                    {"Toys (avg)": cat_index.values, bench_key: bench_index},
                    index=ts_use["date"]
                )
                st.line_chart(plot_df, width="stretch")
            else:
                st.info("Not enough data points to plot the category vs benchmark chart for this window.")

            # ---------------------------------------------
            # INDIVIDUAL TOY (moved BELOW the summary block)
            # ---------------------------------------------
            st.markdown("---")
            # Let the user pick a single toy to visualize
            toy_names = sorted(df_all_toys["item_name"].dropna().unique().tolist())
            choice = st.selectbox("Choose a toy", toy_names, index=0, key="toy_picker")

            # --- Metadata badges for the chosen toy
            meta_cols = [
                "release_year","retirement_year","condition","grade",
                "category_subtype","original_retail","source_platform",
            ]
            meta_row = df_all_toys.loc[df_all_toys["item_name"] == choice, meta_cols].head(1)
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

            # Individual toy time series → show_category()
            df_one = df_all_toys.loc[
                df_all_toys["item_name"] == choice, ["date", "price_usd"]
            ].copy()
            show_category(f"{choice} (Toys)", df_one)

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



























