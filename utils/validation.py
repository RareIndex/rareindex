# utils/validation.py
import io
import re
import pandas as pd

REQUIRED_COLS = {"date", "price_usd"}

def _month_gaps(periods):
    """Count gaps in monthly periods (e.g., 2024-01, 2024-03 has 1 gap)."""
    if len(periods) < 2:
        return 0
    diffs = periods[1:].astype(int) - periods[:-1].astype(int)
    # each value > 1 represents missing months
    return int((diffs[diffs > 1] - 1).sum())

def validate_timeseries_csv(file_or_path) -> dict:
    """
    Validates a CSV for Rare Index format.
    Returns a dict with:
      - level: 'ok' | 'warn' | 'error'
      - messages: [str...]
      - df: cleaned pandas DataFrame (sorted, with parsed dtypes) or None
    """
    out = {"level": "ok", "messages": [], "df": None}
    try:
        if isinstance(file_or_path, (str, bytes)):
            df = pd.read_csv(file_or_path)
        else:
            # file-like object from Streamlit uploader
            data = file_or_path.read()
            df = pd.read_csv(io.BytesIO(data))
    except Exception as e:
        out["level"] = "error"
        out["messages"].append(f"Could not read CSV: {e}")
        return out

    # Columns
    cols = set(map(str.lower, df.columns))
    if not REQUIRED_COLS.issubset(cols):
        out["level"] = "error"
        missing = ", ".join(sorted(REQUIRED_COLS - cols))
        out["messages"].append(f"Missing required columns: {missing}")
        return out

    # Normalize columns (allow case-varied headers)
    df = df.rename(columns={c: c.lower() for c in df.columns})

    # Parse types
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["price_usd"] = pd.to_numeric(df["price_usd"], errors="coerce")

    # Drop bad rows
    before = len(df)
    df = df.dropna(subset=["date", "price_usd"]).copy()
    dropped = before - len(df)
    if dropped > 0:
        out["messages"].append(f"Dropped {dropped} rows with invalid date/price.")

    # Sort & basic checks
    df = df.sort_values("date").reset_index(drop=True)

    if len(df) < 6:
        out["level"] = "warn" if out["level"] != "error" else out["level"]
        out["messages"].append("Very few rows (< 6). Consider adding more history.")

    # Duplicate dates
    dups = df["date"].duplicated().sum()
    if dups > 0:
        out["level"] = "warn" if out["level"] != "error" else out["level"]
        out["messages"].append(f"{dups} duplicate date(s) found. Consider removing.")

    # Non-positive prices
    nonpos = (df["price_usd"] <= 0).sum()
    if nonpos > 0:
        out["level"] = "warn" if out["level"] != "error" else out["level"]
        out["messages"].append(f"{nonpos} non-positive price(s) found.")

    # Monthly continuity & gaps
    periods = df["date"].dt.to_period("M").to_list()
    gaps = _month_gaps(pd.Series(periods))
    if gaps > 0:
        out["level"] = "warn" if out["level"] != "error" else out["level"]
        out["messages"].append(f"{gaps} missing month(s) in the series.")

    # Volatility sanity (flag huge jumps > 100% MoM)
    if len(df) >= 2:
        pct = df["price_usd"].pct_change()
        spikes = (pct.abs() > 1.0).sum()
        if spikes > 0:
            out["level"] = "warn" if out["level"] != "error" else out["level"]
            out["messages"].append(f"{spikes} month-over-month jumps > 100% detected. Verify data.")

    out["df"] = df
    if out["level"] == "ok" and not out["messages"]:
        out["messages"].append("Looks good ✔️")

    return out
