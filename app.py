import glob, re, os
import streamlit as st
import pandas as pd
import numpy as np
import feedparser
import re

def slugify(s: str) -> str:
    s = re.sub(r"\s+", "_", str(s).strip())
    s = re.sub(r"[^A-Za-z0-9_.-]", "", s)
    return s.lower()

# -------------------------
# Basics & Config
# -------------------------
st.set_page_config(page_title="The Rare Index", page_icon="favicon.png", layout="wide")

st.markdown("<h1 style='text-align:center;color:#2E8B57;'>The Rare Index</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align:center;font-size:20px;color:#2E8B57;'><b>Explore Rare Index Categories</b></h3>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;font-size:14px;color:#B22222;'>Demo Data Only — Not Financial Advice</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;font-size:16px;color:#555;'>Track alternative assets (toys, watches, cards) and compare to traditional markets.</p>", unsafe_allow_html=True)
st.markdown("---")

# -------------------------
# Helpers
# -------------------------
@st.cache_data(ttl=600)
def read_csv_cached(path):
    df = pd.read_csv(path)
    if "date" not in df.columns:
        raise ValueError(f"{path} missing 'date' column")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date"]).reset_index(drop=True)
    return df

def make_index_series(final_return, n_points):
    target = 1.0 + float(final_return)
    if n_points <= 1:
        return np.array([100.0]*max(1, n_points))
    r = target ** (1/(n_points-1)) - 1
    levels = [100.0]
    for _ in range(n_points-1):
        levels.append(levels[-1]*(1+r))
    return np.array(levels)

def slice_by_range(df: pd.DataFrame, range_key: str) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    end = pd.to_datetime(df["date"]).max()
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
    return df.copy()

def build_leaderboard(df_all: pd.DataFrame, period: str) -> pd.DataFrame:
    """
    Generic leaderboard across items for a window.
    Requires: item_name, date, price_usd
    Optional for badges: category_subtype, condition, grade, release_year
    """
    if df_all.empty:
        return pd.DataFrame(columns=["Item","Subtype","Condition","Grade","Release Year","Start ($)","Latest ($)","ROI (%)","CAGR (%)"])

    rows = []
    d = df_all.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values(["item_name", "date"])

    for name, g in d.groupby("item_name", sort=True):
        g = g.copy().sort_values("date")
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
        span_years = (g_use["date"].iloc[-1] - g_use["date"].iloc[0]).days / 365.25
        cagr_pct = ((latest_price / start_price) ** (1.0 / span_years) - 1.0) * 100.0 if span_years > 0 else None

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

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["Item","Subtype","Condition","Grade","Release Year","Start ($)","Latest ($)","ROI (%)","CAGR (%)"])
    return out.sort_values("ROI (%)", ascending=False).reset_index(drop=True)

# Markets (demo)
MARKETS = {
    "S&P 500 (~+12% YTD 2025)": 0.12,
    "Nasdaq 100 (~+18% YTD 2025)": 0.18,
    "Dow Jones (~+9.5% YTD 2025)": 0.095,
}

# News
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

def render_news(category_name: str):
    st.markdown("<h5 style='margin-top:0.5rem;'>Trending News</h5>", unsafe_allow_html=True)
    feeds = FEEDS.get(category_name, [])
    combined = []
    with st.spinner("Loading news..."):
        for url in feeds:
            combined.extend(cached_fetch_news(url, limit=3))
    # Deduplicate by title
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
    for item in cleaned:
        pub = item.get("published", "")
        st.markdown(f"- [{item.get('title','Untitled')}]({item.get('link','#')})  \n"
                    f"  <span style='color:#777;font-size:12px;'>{pub}</span>", unsafe_allow_html=True)

# Per-item chart helper (accepts df with date, price_usd)
def show_item_chart(title: str, df_in: pd.DataFrame):
    st.subheader(title)
    if df_in.empty:
        st.warning("No data to display.")
        return
    with st.expander("Options", expanded=False):
        compare = st.checkbox("Compare to a stock index", value=False, key=f"cmp_{title}")
        market_choice = st.selectbox("Choose index", list(MARKETS.keys()), index=0, key=f"mkt_{title}")
        range_choice = st.radio(
            "Range", ["3M", "6M", "1Y", "2Y", "YTD", "All"],
            index=5, horizontal=True, key=f"rng_{title}"
        )

    df_use = slice_by_range(df_in, range_choice)
    if df_use.empty:
        st.warning("No data available for the selected range.")
        return

    start = float(df_use["price_usd"].iloc[0])
    end = float(df_use["price_usd"].iloc[-1])
    roi = ((end - start) / start) * 100.0
    start_label = f"Starting Price ({df_use['date'].iloc[0].strftime('%b %Y')})"
    title_suffix = f" — {range_choice}"

    if compare:
        item_index = (df_use["price_usd"] / df_use["price_usd"].iloc[0]) * 100.0
        market_ytd = MARKETS[market_choice]
        market_index = make_index_series(market_ytd, len(df_use))
        plot_df = pd.DataFrame({title: item_index.values, market_choice: market_index}, index=df_use["date"])
        st.markdown(f"<h4 style='text-align:center;'>{title} — Index vs {market_choice}{title_suffix}</h4>", unsafe_allow_html=True)
        st.line_chart(plot_df, width="stretch")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(start_label, f"${start:,.0f}")
        c2.metric("Latest Price", f"${end:,.0f}")
        c3.metric(f"ROI since {df_use['date'].iloc[0].strftime('%b %Y')}", f"{roi:.1f}%")
        item_last = float(item_index.iloc[-1])
        market_last = float(market_index[-1])
        c4.metric("Outperformance vs Index", f"{(item_last - market_last):+.1f} pp")
    else:
        st.markdown(f"<h4 style='text-align:center;'>{title} — Price Trend{title_suffix}</h4>", unsafe_allow_html=True)
        st.line_chart(df_use.set_index("date")["price_usd"], width="stretch")
        c1, c2, c3 = st.columns(3)
        c1.metric(start_label, f"${start:,.0f}")
        c2.metric("Latest Price", f"${end:,.0f}")
        c3.metric(f"ROI since {df_use['date'].iloc[0].strftime('%b %Y')}", f"{roi:.1f}%")

    # Recent rows
    recent = df_use.tail(5).copy()
    recent["Date"] = recent["date"].dt.strftime("%Y-%m-%d")
    recent["Price ($)"] = recent["price_usd"].map(lambda v: f"${v:,.2f}")
    st.caption("Recent data points")
    st.dataframe(recent[["Date", "Price ($)"]], width="stretch")

def render_category_tab(cat_label: str, csv_path: str, news_key: str, index_title: str):
    st.markdown(f"<p style='text-align:center;color:#555;'>Tracking monthly median resale for the top items by ROI (demo dataset).</p>", unsafe_allow_html=True)

    try:
        df_all = read_csv_cached(csv_path)
    except FileNotFoundError:
        st.error(f"Missing file: {csv_path} — make sure it’s committed and pushed.")
        render_news(news_key)
        return

    # -------------- Category Summary --------------
    st.markdown("### 🧭 Category Summary")
    win_key = st.radio(
        "Summary window", ["3M", "6M", "1Y", "2Y", "YTD", "All"],
        index=2, horizontal=True, key=f"{news_key}_summary_window",
    )

    df_lb = build_leaderboard(df_all, win_key)
    if df_lb.empty:
        st.info("No data in selected window.")
    else:
        items_ct = df_lb.shape[0]
        avg_roi = df_lb["ROI (%)"].mean() if items_ct else None
        med_roi = df_lb["ROI (%)"].median() if items_ct else None
        top_roi = df_lb["ROI (%)"].max() if items_ct else None

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Items", f"{items_ct:,}")
        c2.metric("Avg ROI", f"{avg_roi:.2f}%" if avg_roi is not None else "—")
        c3.metric("Median ROI", f"{med_roi:.2f}%" if med_roi is not None else "—")
        c4.metric("Top item ROI", f"{top_roi:.2f}%" if top_roi is not None else "—")

        # Top & Bottom tables
        st.markdown("### Top & Bottom performers")
        top3 = df_lb.head(3).copy()
        bot3  = df_lb.tail(3).copy()
        for _df in (top3, bot3):
            for col in ["Start ($)", "Latest ($)"]:
                _df[col] = _df[col].apply(lambda v: f"${v:,.2f}" if pd.notnull(v) else "—")
            for col in ["ROI (%)", "CAGR (%)"]:
                _df[col] = _df[col].apply(lambda v: f"{v:,.2f}%" if pd.notnull(v) else "—")
        lcol, rcol = st.columns(2)
        with lcol:
            st.caption("Top 3 by ROI")
            st.dataframe(top3[["Item","Subtype","Condition","Grade","Release Year","Start ($)","Latest ($)","ROI (%)","CAGR (%)"]], width="stretch")
        with rcol:
            st.caption("Bottom 3 by ROI")
            st.dataframe(bot3[["Item","Subtype","Condition","Grade","Release Year","Start ($)","Latest ($)","ROI (%)","CAGR (%)"]], width="stretch")

        # Download leaderboard
        st.download_button(
            "Download Leaderboard (CSV)",
            data=df_lb.to_csv(index=False).encode("utf-8"),
            file_name=f"{news_key}_leaderboard_{win_key}.csv",
            mime="text/csv",
        )

    # -------------- Category vs Benchmark --------------
    st.markdown("### 📈 Category vs Benchmark")
    bench_choice = st.selectbox("Choose benchmark", list(MARKETS.keys()), index=0, key=f"{news_key}_bench")

    df_win = slice_by_range(df_all, win_key)
    if df_win.empty:
        st.info("No data in selected window to plot benchmark.")
    else:
        # Per-item normalize to first in window, then average across items
        d = df_win.sort_values(["item_name", "date"]).copy()
        d["first_in_win"] = d.groupby("item_name")["price_usd"].transform("first")
        d = d[d["first_in_win"] > 0].copy()
        d["item_idx"] = (d["price_usd"] / d["first_in_win"]) * 100.0
        cat_series = d.groupby("date")["item_idx"].mean().sort_index()

        bench = pd.Series(make_index_series(MARKETS[bench_choice], len(cat_series)), index=cat_series.index, name=bench_choice)
        chart_df = pd.DataFrame({index_title: cat_series.values, bench_choice: bench.values}, index=cat_series.index)
        chart_df = chart_df.reset_index(names="date")
        st.line_chart(chart_df.set_index("date"), width="stretch")

        # Download chart data
        st.download_button(
            "Download Category Index (CSV)",
            data=chart_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{news_key}_category_vs_benchmark_{win_key}.csv",
            mime="text/csv",
        )

    # -------------- Individual Item --------------
    st.markdown("---")
    st.markdown("### Individual Item")
    item_names = sorted(df_all["item_name"].dropna().unique().tolist())
    choice = st.selectbox(f"Choose a {news_key[:-1] if news_key.endswith('s') else news_key}", item_names, index=0, key=f"{news_key}_picker")

    meta_cols = ["release_year","retirement_year","condition","grade","category_subtype","original_retail","source_platform"]
    meta_row = df_all.loc[df_all["item_name"] == choice, meta_cols].head(1)
    if not meta_row.empty:
        m = meta_row.iloc[0]
        retail_str = f"${float(m['original_retail']):,.2f}" if pd.notnull(m.get("original_retail")) else "—"
        def val(k): 
            v = m.get(k)
            if pd.isnull(v): return "—"
            if k in ("release_year","retirement_year"):
                try: return str(int(v))
                except: return str(v)
            return str(v)
        st.markdown(" ".join([
            f"<span style='display:inline-block;padding:4px 10px;margin:0 6px 8px 0;border-radius:999px;background:#eef2ff;color:#1e40af;font-size:12px;'>Release: {val('release_year')}</span>",
            f"<span style='display:inline-block;padding:4px 10px;margin:0 6px 8px 0;border-radius:999px;background:#eef2ff;color:#1e40af;font-size:12px;'>Retired: {val('retirement_year')}</span>",
            f"<span style='display:inline-block;padding:4px 10px;margin:0 6px 8px 0;border-radius:999px;background:#ecfeff;color:#155e75;font-size:12px;'>Condition: {val('condition')}</span>",
            f"<span style='display:inline-block;padding:4px 10px;margin:0 6px 8px 0;border-radius:999px;background:#ecfeff;color:#155e75;font-size:12px;'>Grade: {val('grade')}</span>",
            f"<span style='display:inline-block;padding:4px 10px;margin:0 6px 8px 0;border-radius:999px;background:#f0fdf4;color:#166534;font-size:12px;'>Type: {val('category_subtype')}</span>",
            f"<span style='display:inline-block;padding:4px 10px;margin:0 6px 8px 0;border-radius:999px;background:#fff7ed;color:#9a3412;font-size:12px;'>Orig. Retail: {retail_str}</span>",
            f"<span style='display:inline-block;padding:4px 10px;margin:0 6px 8px 0;border-radius:999px;background:#fdf4ff;color:#6b21a8;font-size:12px;'>Source: {val('source_platform')}</span>",
        ]), unsafe_allow_html=True)
     else:
        st.caption("No metadata found for this item.")

    # Filter the selected item and plot
    df_one = df_all.loc[df_all["item_name"] == choice, ["date", "price_usd"]].copy()
    show_item_chart(f"{choice} ({news_key})", df_one)

    # Per-item CSV download (place directly after the chart)
    # If you don't have `category_name` in scope, hardcode it: e.g., "toys" / "watches" / "cards"
    st.download_button(
        label="Download selected item (CSV)",
        data=df_one.to_csv(index=False).encode("utf-8"),
        file_name=f"{slugify(category_name)}_{slugify(choice)}.csv",  # or slugify("toys")
        mime="text/csv",
    )

    # -------------- News --------------

    render_news(news_key)

# -------------------------
# Tabs
# -------------------------
tab_cards, tab_watches, tab_toys, tab_live, tab_roi, tab_top10, tab_validator = st.tabs(
    ["Cards", "Watches", "Toys", "Live eBay (beta)", "ROI Calculator", "Top 10 (Demo)", "Validator"]
)

# --- Cards ---
with tab_cards:
    # Prefer category dataset if present; fallback to demo card CSV
    cards_csv = "data/cards/cards_top50.csv"
    if os.path.exists(cards_csv):
        render_category_tab("Cards", cards_csv, "Cards", "Cards (Category Index)")
    else:
        st.caption("No category dataset found yet — showing single demo card.")
        # Fallback: if you have a single demo CSV in repo
        demo = "data/cards/cards_011.csv"
        if os.path.exists(demo):
            df_demo = read_csv_cached(demo)[["date","price_usd"]].copy()
            show_item_chart("Pokémon Card #011 (Cards)", df_demo)
        render_news("Cards")

# --- Watches ---
with tab_watches:
    render_category_tab("Watches", "data/watches/watches_top50.csv", "Watches", "Watches (Category Index)")

# --- Toys ---
with tab_toys:
    render_category_tab("Toys", "data/toys/toys_top50.csv", "Toys", "Toys (Category Index)")

# --- Live eBay (beta) ---
with tab_live:
    st.markdown("### Live eBay (beta)")
    demo_mode = st.toggle("Demo mode (no API)", value=True, help="Shows a tiny sample so the heartbeat is visible.")
    uploaded = st.file_uploader(
        "Upload CSV (columns: title, price, currency, listingType, condition, category, viewItemURL)",
        type=["csv"]
    )
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
        df = pd.DataFrame([
            {"title":"Pokemon Charizard Base Set Holo PSA 9","price":1750,"currency":"USD","listingType":"FixedPrice","condition":"Mint","category":"Collectible Card Games","viewItemURL":"https://www.ebay.com/itm/111111111111"},
            {"title":"Rolex Submariner 114060","price":8950,"currency":"USD","listingType":"Auction","condition":"Pre-owned","category":"Watches","viewItemURL":"https://www.ebay.com/itm/222222222222"},
            {"title":"Omega Speedmaster 3570.50","price":4650,"currency":"USD","listingType":"FixedPrice","condition":"Pre-owned","category":"Watches","viewItemURL":"https://www.ebay.com/itm/333333333333"},
        ])

    if df is not None and not df.empty:
        st.subheader("Results")
        df_display = df.copy()
        df_display["price ($)"] = pd.to_numeric(df_display["price"], errors="coerce").apply(lambda v: f"${v:,.2f}" if pd.notnull(v) else "—")
        st.dataframe(df_display[["title","price ($)","currency","listingType","condition","category","viewItemURL"]], width="stretch")

        st.markdown("---")
        st.markdown("### 🔗 Quick Links")
        for _, row in df.iterrows():
            title = str(row.get("title", "")).strip()
            url = str(row.get("viewItemURL", "")).strip()
            if title and url and url.startswith("http"):
                st.markdown(f"- [{title}]({url})")

        st.markdown("### 🏷 Categories")
        badge_colors = {"Watches":"#16a34a","Toys":"#f59e0b","Collectible Card Games":"#3b82f6"}
        tags = sorted(set(df["category"].dropna().astype(str)))
        st.markdown(" ".join(
            [f"<span style='display:inline-block;padding:2px 8px;margin-right:6px;border-radius:999px;background:{badge_colors.get(cat,'#64748b')};color:white;font-size:12px'>{cat}</span>"
             for cat in tags if cat]), unsafe_allow_html=True)

        prices = pd.to_numeric(df["price"], errors="coerce").dropna()
        c1, c2, c3 = st.columns(3)
        c1.metric("📦 Items", f"{len(df):,}")
        c2.metric("💲 Median price", f"${prices.median():,.2f}" if not prices.empty else "—")
        c3.metric("💰 Average price", f"${prices.mean():,.2f}" if not prices.empty else "—")

        st.download_button(
            "Download table as CSV",
            df.to_csv(index=False).encode("utf-8"),
            file_name="rareindex_results.csv",
            mime="text/csv"
        )

# --- ROI Calculator ---
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

# --- Top 10 (Demo) ---
def calc_roi_from_csv(name: str, category: str, csv_path: str):
    try:
        df = read_csv_cached(csv_path)
        start = float(df["price_usd"].iloc[0])
        latest = float(df["price_usd"].iloc[-1])
        roi_pct = ((latest - start) / start) * 100.0
        return {"name": name, "category": category, "start": start, "latest": latest, "roi_pct": roi_pct}
    except Exception:
        return {"name": name, "category": category, "start": None, "latest": None, "roi_pct": None}

with tab_top10:
    st.markdown("### 🏆 Top 10 ROI (Demo)")
    card_paths = sorted(glob.glob("data/cards/cards_*.csv"))
    card_items = []
    for p in card_paths:
        m = re.search(r"cards_(\d+)\.csv$", p)
        label = f"Card #{m.group(1)} (Demo)" if m else "Card (Demo)"
        card_items.append({"name": label, "category": "Cards", "csv": p})

    demo_items = card_items + [
        {"name": "Rolex Submariner 116610LN", "category": "Watches", "csv": "watches.csv"},
        {"name": "LEGO 75290 Mos Eisley Cantina", "category": "Toys", "csv": "toys.csv"},
    ]
    results = [calc_roi_from_csv(x["name"], x["category"], x["csv"]) for x in demo_items]
    df_top = pd.DataFrame(results)
    df_top = df_top.dropna(subset=["roi_pct"]).copy()

    if not df_top.empty:
        df_top = df_top.sort_values("roi_pct", ascending=False).head(10).reset_index(drop=True)
        df_top["Start ($)"]  = df_top["start"].apply(lambda v: f"${v:,.2f}" if pd.notnull(v) else "—")
        df_top["Latest ($)"] = df_top["latest"].apply(lambda v: f"${v:,.2f}" if pd.notnull(v) else "—")
        df_top["ROI (%)"]    = df_top["roi_pct"].apply(lambda v: f"{v:,.2f}%" if pd.notnull(v) else "—")
        st.dataframe(df_top[["name","category","Start ($)","Latest ($)","ROI (%)"]], width="stretch")
        c1, c2 = st.columns(2)
        c1.metric("Entries ranked", f"{len(df_top)}")
        c2.metric("Top ROI", df_top.iloc[0]["ROI (%)"] if len(df_top) else "—")
        st.download_button(
            "Download Top 10 as CSV",
            df_top[["name","category","Start ($)","Latest ($)","ROI (%)"]].to_csv(index=False).encode("utf-8"),
            file_name="top10_demo.csv",
            mime="text/csv"
        )
    else:
        st.info("No ROI data available yet. Make sure your CSVs exist and have 'date' and 'price_usd' columns.")

# Footer
st.info("Waiting for eBay Growth Check approval. Live API calls will replace demo/CSV here when enabled.")
st.markdown("---")
st.caption("RI Beta — Demo Data Only. Market lines use fixed 2025 YTD endpoints for simplicity.")
st.markdown("## 📢 Latest Updates")
st.markdown("""
- **Oct 2025:** Demo site launched with Cards, Watches, and Toys.
- **Sep 2025:** Lugia V Alt Art sales stable around $420–450 range.
- **Sep 2025:** Rolex Submariner 116610LN resale prices holding steady despite market dip.
- **Aug 2025:** LEGO 75290 Mos Eisley Cantina shows consistent ROI trend.
""")
st.markdown("---")
st.markdown("<p style='text-align:center;font-size:14px;color:#2E8B57;'>© 2025 The Rare Index · Demo Data Only</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;font-size:14px;'><a href='mailto:david@therareindex.com'>Contact: david@therareindex.com</a></p>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;font-size:14px;'><a href='https://forms.gle/KxufuFLcEVZD6qtD8' target='_blank'>Subscribe for updates</a></p>", unsafe_allow_html=True)
