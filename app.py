# app.py — Rare Index (clean, unified)

import re
import glob
import zipfile
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
import feedparser

# =============================
# Basic config & global styles
# =============================
st.set_page_config(page_title="The Rare Index", page_icon="favicon.png", layout="wide")

# ==== Minimal, safe header injection (no markdown sanitation) ====
def inject_header():
    head = """
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=IBM+Plex+Sans:wght@600;700&family=IBM+Plex+Mono:wght@400;600&display=swap" rel="stylesheet">
    """
    css = """
    <style>
    :root{
      --ri-accent:#17663f;       /* brand green */
      --ri-accent-2:#0f4f31;
      --ri-soft:#f7faf9;
    }

    /* App container spacing + max width */
    .block-container{
      padding-top: 1.25rem;
      padding-bottom: 2.0rem;
      max-width: 1200px;
    }

    /* Headings */
    h1, h2, h3, h4{
      font-family: 'IBM Plex Sans', Inter, sans-serif;
      letter-spacing: .2px;
    }
    h1{
      font-weight: 800;          /* bolder */
      font-size: 2.4rem;         /* larger */
      color: #111111;            /* black instead of green */
      text-align: center;
      margin-top: .8rem;
      margin-bottom: .6rem;
    }

    /* Metrics font tuning */
    [data-testid="stMetricValue"]{
      font-variant-numeric: tabular-nums;
      font-family: 'IBM Plex Mono', monospace;
    }
    [data-testid="stMetricLabel"]{ color:#374151; }

    /* Brand bar */
    .ri-brandbar{
      background: linear-gradient(90deg,var(--ri-accent),var(--ri-accent-2));
      color:#fff;
      padding:10px 16px;
      border-radius:12px;
      display:flex; align-items:center; justify-content:space-between;
      /* more breathing room from the very top + a bit more below */
      margin: calc(env(safe-area-inset-top) + 30px) 0 28px 0;
    }
    .ri-brandbar .left{ font-weight:700; letter-spacing:.6px; }
    .ri-brandbar .right{ font-weight:500; opacity:.9; }
    .ri-badge{
      display:inline-block; padding:2px 10px; margin-left:10px;
      background:rgba(255,255,255,.18); color:#fff; border:1px solid rgba(255,255,255,.35);
      border-radius:999px; font-size:12px;
    }

    /* Hero */
    .ri-hero{ text-align:center; margin:.35rem 0 .6rem 0; }
    .ri-hero .ri-sub{ color:#4b5563; font-size:18px; margin:.25rem 0 .4rem 0; }
    .ri-callout{
      display:inline-block; background:var(--ri-soft);
      padding:14px 16px; border-radius:12px; color:#111827;
      border:1px solid rgba(0,0,0,.05); font-size:16px;
      margin-bottom: 1.8rem;   /* extra gap above the tabs */
    }

    /* Radio spacing */
    .stRadio > div{ gap:10px; }
    </style>
    """
    # Use st.html so tags are not stripped
    st.html(head + css)

def render_brandbar():
    st.html("""
    <div class="ri-brandbar">
      <div class="left">RARE INDEX <span class="ri-badge">BETA</span></div>
      <div class="right">Demo data · Not financial advice</div>
    </div>
    """)

def render_hero():
    st.html("""
    <div class="ri-hero">
      <h1>The Rare Index</h1>
      <div class="ri-sub">Explore alternative assets versus market benchmarks</div>
      <div class="ri-callout">
        Demo platform for tracking cards, watches, and toys against S&amp;P 500, Nasdaq, and Dow. Data is illustrative only.
      </div>
    </div>
    """)

# --- call these once, immediately after set_page_config ---
inject_header()
render_brandbar()
render_hero()

# =============================
# Utils
# =============================
@st.cache_data(ttl=600)
def read_csv_cached(path: str) -> pd.DataFrame:
    """Read and clean a CSV once; cache for 10 minutes."""
    df = pd.read_csv(path)
    if "date" not in df.columns:
        raise ValueError(f"{path} missing required 'date' column")
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def slugify(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_")


def make_index_series(final_return: float, n_points: int) -> np.ndarray:
    """
    Return index levels starting at 100 and compounding smoothly to 100*(1+final_return).
    """
    n = max(int(n_points), 1)
    if n == 1:
        return np.array([100.0])
    target = 1.0 + float(final_return)
    r = target ** (1.0 / (n - 1)) - 1.0
    levels = [100.0]
    for _ in range(n - 1):
        levels.append(levels[-1] * (1 + r))
    return np.array(levels, dtype=float)


def slice_by_range(df: pd.DataFrame, range_key: str) -> pd.DataFrame:
    """
    Slice df by range_key in {'3M','6M','1Y','2Y','YTD','All'}.
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
        start = end - pd.DateOffset(months=months - 1)
        return df[df["date"] >= start].copy()

    return df.copy()


def build_leaderboard(df_all: pd.DataFrame, period: str) -> pd.DataFrame:
    """
    Generic per-item ROI/CAGR leaderboard for a chosen window.
    df_all requires: ['item_name','date','price_usd'] (+optional metadata)
    """
    base_cols = [
        "Item",
        "Subtype",
        "Condition",
        "Grade",
        "Release Year",
        "Start ($)",
        "Latest ($)",
        "ROI (%)",
        "CAGR (%)",
    ]
    if df_all.empty:
        return pd.DataFrame(columns=base_cols)

    rows = []
    df_all = df_all.copy()
    df_all["date"] = pd.to_datetime(df_all["date"])
    df_all = df_all.sort_values(["item_name", "date"])

    for name, g in df_all.groupby("item_name", sort=True):
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
        cagr_pct = (
            ((latest_price / start_price) ** (1.0 / span_years) - 1.0) * 100.0
            if span_years > 0
            else None
        )

        rows.append(
            {
                "Item": name,
                "Subtype": subtype,
                "Condition": cond,
                "Grade": grade,
                "Release Year": int(rel_year) if pd.notnull(rel_year) else None,
                "Start ($)": start_price,
                "Latest ($)": latest_price,
                "ROI (%)": roi_pct,
                "CAGR (%)": cagr_pct,
            }
        )

    if not rows:
        return pd.DataFrame(columns=base_cols)

    return pd.DataFrame(rows).sort_values("ROI (%)", ascending=False).reset_index(drop=True)


# Fixed demo YTDs for 2025 (illustrative)
MARKETS = {
    "S&P 500 (~+12% YTD 2025)": 0.12,
    "Nasdaq 100 (~+18% YTD 2025)": 0.18,
    "Dow Jones (~+9.5% YTD 2025)": 0.095,
}

# =============================
# News
# =============================
def fetch_news(feed_url: str, limit: int = 5):
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


@st.cache_data(ttl=600)
def cached_fetch_news(feed_url: str, limit: int = 5):
    return fetch_news(feed_url, limit)


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


def render_news(category_name: str):
    st.markdown("<h5 style='margin-top:.5rem;'>Trending News</h5>", unsafe_allow_html=True)
    feeds = FEEDS.get(category_name, [])
    combined = []
    with st.spinner("Loading news..."):
        for url in feeds:
            combined.extend(cached_fetch_news(url, limit=3))

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
        st.markdown(
            f"- [{item.get('title','Untitled')}]({item.get('link','#')})  \n"
            f"  <span class='ri-muted' style='font-size:12px;'>{pub}</span>",
            unsafe_allow_html=True,
        )


# =============================
# Charts & Export helpers
# =============================
def show_item_chart(title: str, df_source: pd.DataFrame):
    """Render a timeseries line chart and quick stats for a single item."""
    df = df_source.copy().sort_values("date")
    if df.empty:
        st.warning("No data available.")
        return

    st.markdown(f"<h4 style='text-align:center;'>{title} — Price Trend</h4>", unsafe_allow_html=True)
    st.line_chart(df.set_index("date")["price_usd"])

    start = float(df["price_usd"].iloc[0])
    end = float(df["price_usd"].iloc[-1])
    roi = ((end - start) / start) * 100.0
    start_label = f"Starting Price ({df['date'].iloc[0].strftime('%b %Y')})"

    c1, c2, c3 = st.columns(3)
    c1.metric(start_label, f"${start:,.0f}")
    c2.metric("Latest Price", f"${end:,.0f}")
    c3.metric(f"ROI since {df['date'].iloc[0].strftime('%b %Y')}", f"{roi:.1f}%")

    recent = df.tail(5).copy()
    recent["Date"] = recent["date"].dt.strftime("%Y-%m-%d")
    recent["Price ($)"] = recent["price_usd"].map(lambda v: f"${v:,.2f}")
    st.caption("Recent data points")
    st.dataframe(recent[["Date", "Price ($)"]])


def build_export_zip(
    category_slug: str,
    win_key: str,
    df_all: pd.DataFrame,
    df_lb_raw: pd.DataFrame,
    top3_raw: pd.DataFrame,
    bot3_raw: pd.DataFrame,
    choice: str,
    df_one_raw: pd.DataFrame,
) -> bytes:
    """
    Create a ZIP (as bytes) containing:
      - full dataset
      - leaderboard for the current window
      - Top+Bottom combined
      - selected item timeseries
    All are raw numeric DataFrames (no formatting applied).
    """
    if (top3_raw is None) or (bot3_raw is None):
        combo_raw = pd.DataFrame()
    else:
        combo_raw = pd.concat(
            [top3_raw.assign(_rank="Top"), bot3_raw.assign(_rank="Bottom")],
            ignore_index=True,
        )

    bio = BytesIO()
    with zipfile.ZipFile(bio, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"{category_slug}_full_dataset.csv", df_all.to_csv(index=False))
        z.writestr(f"{category_slug}_leaderboard_{win_key}.csv", df_lb_raw.to_csv(index=False))
        z.writestr(f"{category_slug}_top_bottom_{win_key}.csv", combo_raw.to_csv(index=False))
        z.writestr(f"{category_slug}_{slugify(choice)}.csv", df_one_raw.to_csv(index=False))
    return bio.getvalue()


# =============================
# Category tab renderer
# =============================
def render_category_tab(category_name: str, csv_path: str, news_key: str):
    """
    category_name: "Toys" | "Watches" | "Cards"
    csv_path: path to dataset: expects item_name,date,price_usd (+optional metadata)
    news_key: same label used for news feeds
    """
    st.markdown(
        f"<p class='ri-note'>Tracking monthly median resale for top {category_name.lower()} by ROI (demo dataset).</p>",
        unsafe_allow_html=True,
    )

    # Load data
    try:
        df_all = read_csv_cached(csv_path)
    except FileNotFoundError:
        st.error(f"Missing file: {csv_path} — make sure it’s committed and pushed.")
        render_news(news_key)
        return
    except Exception as e:
        st.error(f"Error loading {csv_path}: {e}")
        render_news(news_key)
        return

    if not {"item_name", "date", "price_usd"}.issubset(df_all.columns):
        st.error(f"'{csv_path}' is missing required columns: item_name, date, price_usd")
        render_news(news_key)
        return

    df_all = df_all.copy()
    df_all["date"] = pd.to_datetime(df_all["date"])
    df_all = df_all.sort_values(["item_name", "date"])

    # ---------- Category Summary ----------
    st.markdown("### 🧭 Category Summary")
    win_key = st.radio(
        "Summary window",
        ["3M", "6M", "1Y", "2Y", "YTD", "All"],
        index=2,
        horizontal=True,
        key=f"{slugify(category_name)}_summary_window",
    )

    # Prepare defaults so Export ZIP is always safe
    df_lb = pd.DataFrame()
    top3_raw = None
    bot3_raw = None

    df_win = slice_by_range(df_all, win_key)
    if df_win.empty:
        st.info("No data in selected window.")
    else:
        # Per-item ROI leaderboard for this window (RAW)
        df_lb = build_leaderboard(df_all, win_key)

        items_ct = df_lb.shape[0]
        avg_roi = df_lb["ROI (%)"].mean() if items_ct else None
        med_roi = df_lb["ROI (%)"].median() if items_ct else None
        top_roi = df_lb["ROI (%)"].max() if items_ct else None

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Items", f"{items_ct:,}")
        c2.metric("Avg ROI", f"{avg_roi:.2f}%" if avg_roi is not None else "—")
        c3.metric("Median ROI", f"{med_roi:.2f}%" if med_roi is not None else "—")
        c4.metric("Top item ROI", f"{top_roi:.2f}%" if top_roi is not None else "—")

        # Category vs Benchmark
        st.markdown("### 📈 Category vs Benchmark")
        bench_key = st.selectbox(
            "Choose benchmark",
            list(MARKETS.keys()),
            index=0,
            key=f"{slugify(category_name)}_bench",
        )

        df_norm = df_win.sort_values(["item_name", "date"]).copy()
        df_norm["first_in_win"] = df_norm.groupby("item_name")["price_usd"].transform("first")
        df_norm = df_norm[df_norm["first_in_win"] > 0].copy()
        if df_norm.empty:
            st.info("Not enough data after normalization to show the benchmark comparison.")
        else:
            df_norm["item_idx"] = (df_norm["price_usd"] / df_norm["first_in_win"]) * 100.0
            cat_series = df_norm.groupby("date")["item_idx"].mean().sort_index()

            bench_ytd = MARKETS[bench_key]
            bench_series = pd.Series(
                make_index_series(bench_ytd, len(cat_series)), index=cat_series.index, name=bench_key
            )

            chart_df = pd.DataFrame(
                {f"{category_name} (Category Index)": cat_series, bench_key: bench_series.values}
            )
            st.line_chart(chart_df)

        # Top & Bottom performers
        st.markdown("### Top & Bottom performers")
        if df_lb.empty:
            st.info("No leaderboard rows for the selected window yet.")
        else:
            # RAW copies (for downloads/ZIP)
            top3_raw = df_lb.head(3).copy()
            bot3_raw = df_lb.tail(3).copy()

            # Pretty copies for on-screen tables
            top3 = top3_raw.copy()
            bot3 = bot3_raw.copy()
            fmt_money = lambda v: f"${v:,.2f}" if pd.notnull(v) else "—"
            fmt_pct = lambda v: f"{v:,.2f}%" if pd.notnull(v) else "—"
            for _df in (top3, bot3):
                _df["Start ($)"] = _df["Start ($)"].apply(fmt_money)
                _df["Latest ($)"] = _df["Latest ($)"].apply(fmt_money)
                _df["ROI (%)"] = _df["ROI (%)"].apply(fmt_pct)
                _df["CAGR (%)"] = _df["CAGR (%)"].apply(fmt_pct)

            lcol, rcol = st.columns(2)
            with lcol:
                st.caption("Top 3 by ROI")
                st.dataframe(
                    top3[
                        [
                            "Item",
                            "Subtype",
                            "Condition",
                            "Grade",
                            "Release Year",
                            "Start ($)",
                            "Latest ($)",
                            "ROI (%)",
                            "CAGR (%)",
                        ]
                    ]
                )
            with rcol:
                st.caption("Bottom 3 by ROI")
                st.dataframe(
                    bot3[
                        [
                            "Item",
                            "Subtype",
                            "Condition",
                            "Grade",
                            "Release Year",
                            "Start ($)",
                            "Latest ($)",
                            "ROI (%)",
                            "CAGR (%)",
                        ]
                    ]
                )

            # Download Top & Bottom combined (RAW)
            combo = pd.concat(
                [top3_raw.copy().assign(Rank="Top"), bot3_raw.copy().assign(Rank="Bottom")],
                ignore_index=True,
            )
            st.download_button(
                "Download Top & Bottom (CSV)",
                data=combo.to_csv(index=False).encode("utf-8"),
                file_name=f"{slugify(news_key)}_top_bottom_{slugify(win_key)}.csv",
                mime="text/csv",
            )

    # ---------- Per-item drilldown ----------
    st.markdown("---")
    st.markdown("### Individual Item")

    names = sorted(df_all["item_name"].dropna().unique().tolist())
    choice = st.selectbox("Choose an item", names, index=0, key=f"{slugify(category_name)}_picker")

    # Metadata badges
    meta_cols = [
        "release_year",
        "retirement_year",
        "condition",
        "grade",
        "category_subtype",
        "original_retail",
        "source_platform",
    ]
    if set(meta_cols).issubset(df_all.columns):
        meta_row = df_all.loc[df_all["item_name"] == choice, meta_cols].head(1)
        if not meta_row.empty:
            m = meta_row.iloc[0]

            def _val(col):
                v = m.get(col, None)
                if pd.isna(v):
                    return "—"
                if col in ("release_year", "retirement_year"):
                    try:
                        return str(int(v))
                    except Exception:
                        return str(v)
                if col == "original_retail":
                    try:
                        return f"${float(v):,.2f}"
                    except Exception:
                        return "—"
                return str(v)

            st.markdown(
                " ".join(
                    [
                        f"<span style='display:inline-block;padding:4px 10px;margin:0 6px 8px 0;border-radius:999px;background:#eef2ff;color:#1e40af;font-size:12px;'>Release: {_val('release_year')}</span>",
                        f"<span style='display:inline-block;padding:4px 10px;margin:0 6px 8px 0;border-radius:999px;background:#eef2ff;color:#1e40af;font-size:12px;'>Retired: {_val('retirement_year')}</span>",
                        f"<span style='display:inline-block;padding:4px 10px;margin:0 6px 8px 0;border-radius:999px;background:#ecfeff;color:#155e75;font-size:12px;'>Condition: {_val('condition')}</span>",
                        f"<span style='display:inline-block;padding:4px 10px;margin:0 6px 8px 0;border-radius:999px;background:#ecfeff;color:#155e75;font-size:12px;'>Grade: {_val('grade')}</span>",
                        f"<span style='display:inline-block;padding:4px 10px;margin:0 6px 8px 0;border-radius:999px;background:#f0fdf4;color:#166534;font-size:12px;'>Type: {_val('category_subtype')}</span>",
                        f"<span style='display:inline-block;padding:4px 10px;margin:0 6px 8px 0;border-radius:999px;background:#fff7ed;color:#9a3412;font-size:12px;'>Orig. Retail: {_val('original_retail')}</span>",
                        f"<span style='display:inline-block;padding:4px 10px;margin:0 6px 8px 0;border-radius:999px;background:#fdf4ff;color:#6b21a8;font-size:12px;'>Source: {_val('source_platform')}</span>",
                    ]
                ),
                unsafe_allow_html=True,
            )
        else:
            st.caption("No metadata found for this item.")

    # Filter + chart
    df_one = df_all.loc[df_all["item_name"] == choice, ["date", "price_usd"]].copy()
    show_item_chart(f"{choice} ({news_key})", df_one)

    # Per-item CSV (RAW)
    st.download_button(
        "Download selected item (CSV)",
        data=df_one.to_csv(index=False).encode("utf-8"),
        file_name=f"{slugify(news_key)}_{slugify(choice)}.csv",
        mime="text/csv",
    )

    # Export All (ZIP) — after per-item so choice/df_one exist
    zip_bytes = build_export_zip(
        category_slug=slugify(news_key),
        win_key=win_key,
        df_all=df_all,
        df_lb_raw=df_lb,
        top3_raw=top3_raw if top3_raw is not None else pd.DataFrame(),
        bot3_raw=bot3_raw if bot3_raw is not None else pd.DataFrame(),
        choice=choice,
        df_one_raw=df_one,
    )
    st.download_button(
        "Export All (ZIP)",
        data=zip_bytes,
        file_name=f"{slugify(news_key)}_export_{slugify(win_key)}.zip",
        mime="application/zip",
    )

    # News
    render_news(news_key)


# =============================
# Tabs
# =============================
tab_cards, tab_watches, tab_toys, tab_live, tab_roi, tab_top10, tab_validator = st.tabs(
    ["Cards", "Watches", "Toys", "Live eBay (beta)", "ROI Calculator", "Top 10 (Demo)", "Validator"]
)

# ---- Cards
with tab_cards:
    cards_path = "data/cards/cards_top50.csv"
    try:
        render_category_tab("Cards", cards_path, "Cards")
    except Exception:
        st.caption("No category dataset found yet — showing single demo card if present.")
        try:
            df_single = read_csv_cached("data/cards/cards_011.csv")
            show_item_chart("Pokémon Card #011 (Cards)", df_single[["date", "price_usd"]])
            st.download_button(
                "Download selected item (CSV)",
                data=df_single[["date", "price_usd"]].to_csv(index=False).encode("utf-8"),
                file_name="cards_pokemon_011.csv",
                mime="text/csv",
            )
        except Exception:
            st.warning("No demo card CSV found.")

# ---- Watches
with tab_watches:
    render_category_tab("Watches", "data/watches/watches_top50.csv", "Watches")

# ---- Toys
with tab_toys:
    render_category_tab("Toys", "data/toys/toys_top50.csv", "Toys")

# ---- Live eBay (beta)
with tab_live:
    st.markdown("### Live eBay (beta)")
    demo_mode = st.toggle("Demo mode (no API)", value=True, help="Shows a tiny sample so the heartbeat is visible.")

    uploaded = st.file_uploader(
        "Upload CSV (columns: title, price, currency, listingType, condition, category, viewItemURL)", type=["csv"]
    )

    colA, colB = st.columns(2)
    show_demo = colA.button("Show demo results")
    show_csv = colB.button("Show uploaded CSV")

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
                "price": 1750,
                "currency": "USD",
                "listingType": "FixedPrice",
                "condition": "Mint",
                "category": "Collectible Card Games",
                "viewItemURL": "https://www.ebay.com/itm/111111111111",
            },
            {
                "title": "Rolex Submariner 114060",
                "price": 8950,
                "currency": "USD",
                "listingType": "Auction",
                "condition": "Pre-owned",
                "category": "Watches",
                "viewItemURL": "https://www.ebay.com/itm/222222222222",
            },
            {
                "title": "Omega Speedmaster 3570.50",
                "price": 4650,
                "currency": "USD",
                "listingType": "FixedPrice",
                "condition": "Pre-owned",
                "category": "Watches",
                "viewItemURL": "https://www.ebay.com/itm/333333333333",
            },
        ]
        df = pd.DataFrame(demo_rows)

    if df is not None and not df.empty:
        st.subheader("Results")

        df_display = df.copy()
        df_display["price ($)"] = pd.to_numeric(df_display["price"], errors="coerce").apply(
            lambda v: f"${v:,.2f}" if pd.notnull(v) else "—"
        )
        st.dataframe(df_display[["title", "price ($)", "currency", "listingType", "condition", "category", "viewItemURL"]])

        st.markdown("---")
        st.markdown("### 🔗 Quick Links")
        for _, row in df.iterrows():
            title = str(row.get("title", "")).strip()
            url = str(row.get("viewItemURL", "")).strip()
            if title and url and url.startswith("http"):
                st.markdown(f"- [{title}]({url})")

        st.markdown("### 🏷 Categories")
        badge_colors = {"Watches": "#16a34a", "Toys": "#f59e0b", "Collectible Card Games": "#3b82f6"}
        st.markdown(
            " ".join(
                [
                    f"<span style='display:inline-block;padding:2px 8px;margin-right:6px;border-radius:999px;background:{badge_colors.get(str(cat),'#64748b')};color:white;font-size:12px'>{str(cat)}</span>"
                    for cat in sorted(set(df["category"].dropna().astype(str)))
                    if cat
                ]
            ),
            unsafe_allow_html=True,
        )

        prices = pd.to_numeric(df["price"], errors="coerce").dropna()
        c1, c2, c3 = st.columns(3)
        c1.metric("📦 Items", f"{len(df):,}")
        c2.metric("💲 Median price", f"${prices.median():,.2f}" if not prices.empty else "—")
        c3.metric("💰 Average price", f"${prices.mean():,.2f}" if not prices.empty else "—")

        st.download_button(
            "Download table as CSV",
            df.to_csv(index=False).encode("utf-8"),
            file_name="rareindex_results.csv",
            mime="text/csv",
        )

# ---- ROI Calculator
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

# ---- Top 10 (Demo)
with tab_top10:
    st.markdown("### 🏆 Top 10 ROI (Demo)")
    st.caption("This demo scans any 'data/cards/cards_*.csv' plus two sample rows.")
    card_paths = sorted(glob.glob("data/cards/cards_*.csv"))
    items = []
    for p in card_paths:
        m = re.search(r"cards_(\d+)\.csv$", p)
        label = f"Card #{m.group(1)} (Demo)" if m else "Card (Demo)"
        items.append({"name": label, "category": "Cards", "csv": p})
    items += [
        {"name": "Rolex Submariner 116610LN", "category": "Watches", "csv": "watches.csv"},
        {"name": "LEGO 75290 Mos Eisley Cantina", "category": "Toys", "csv": "toys.csv"},
    ]

    def _calc_roi(csv_path):
        try:
            df = read_csv_cached(csv_path)
            start = float(df["price_usd"].iloc[0])
            latest = float(df["price_usd"].iloc[-1])
            return start, latest, ((latest - start) / start) * 100.0
        except Exception:
            return None, None, None

    results = []
    for x in items:
        s, l, r = _calc_roi(x["csv"])
        if r is not None:
            results.append({"name": x["name"], "category": x["category"], "start": s, "latest": l, "roi_pct": r})
    df_top = pd.DataFrame(results)
    if df_top.empty:
        st.info("No ROI data available for demo files.")
    else:
        df_top = df_top.sort_values("roi_pct", ascending=False).head(10).reset_index(drop=True)
        df_top["Start ($)"] = df_top["start"].apply(lambda v: f"${v:,.2f}")
        df_top["Latest ($)"] = df_top["latest"].apply(lambda v: f"${v:,.2f}")
        df_top["ROI (%)"] = df_top["roi_pct"].apply(lambda v: f"{v:,.2f}%")
        st.dataframe(df_top[["name", "category", "Start ($)", "Latest ($)", "ROI (%)"]])

# ---- Validator
with tab_validator:
    st.markdown("### ✅ CSV Validator")
    st.caption("Required columns: **date, price_usd**. Dates should be monthly.")
    uploaded = st.file_uploader("Upload a CSV to validate", type=["csv"], key="validator_uploader")
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            msgs = []
            level = "ok"
            if "date" not in df.columns or "price_usd" not in df.columns:
                level = "error"
                msgs.append("Missing required columns: 'date' and/or 'price_usd'.")
            else:
                try:
                    df["date"] = pd.to_datetime(df["date"])
                    df = df.sort_values("date")
                    if df["date"].diff().dt.days.dropna().median() < 15:
                        msgs.append("Dates do not look monthly; verify frequency.")
                    if (pd.to_numeric(df["price_usd"], errors="coerce") <= 0).any():
                        msgs.append("Some price_usd values are non-positive.")
                except Exception as e:
                    level = "error"
                    msgs.append(f"Date parsing error: {e}")
            if level == "error":
                st.error("Validation failed.")
            elif msgs:
                st.warning("Validation warnings.")
            else:
                st.success("Validation OK.")
            for m in msgs:
                st.write("•", m)
            if "date" in df.columns and "price_usd" in df.columns:
                st.markdown("**Preview (first & last 5 rows)**")
                st.dataframe(df.head(5))
                st.dataframe(df.tail(5))
        except Exception as e:
            st.error(f"Could not validate: {e}")

# =============================
# Footer / Info
# =============================
from datetime import datetime
BUILD = "cloud-verify 2025-10-17 04:30Z"
st.caption(f"Build: {BUILD} · Python {pd.__version__ if 'pd' in globals() else ''}")

st.info("Waiting for eBay Growth Check approval. Live API calls will replace demo/CSV here when enabled.")
st.markdown("---")
st.caption("RI Beta — Demo Data Only. Market lines use fixed 2025 YTD endpoints for simplicity.")
st.markdown("## 📢 Latest Updates")
st.markdown(
    """
- **Oct 2025:** Demo site launched with Cards, Watches, and Toys.
- **Sep 2025:** Lugia V Alt Art sales stable around $420–450 range.
- **Sep 2025:** Rolex Submariner 116610LN resale prices holding steady despite market dip.
- **Aug 2025:** LEGO 75290 Mos Eisley Cantina shows consistent ROI trend.
"""
)
st.markdown("---")
st.markdown(
    "<p style='text-align:center;font-size:14px;color:#2E8B57;'>© 2025 The Rare Index · Demo Data Only</p>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center;font-size:14px;'><a href='mailto:david@therareindex.com'>Contact: david@therareindex.com</a></p>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center;font-size:14px;'><a href='https://forms.gle/KxufuFLcEVZD6qtD8' target='_blank'>Subscribe for updates</a></p>",
    unsafe_allow_html=True,
)
