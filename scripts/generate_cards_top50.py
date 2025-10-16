import os, random
from datetime import datetime
import numpy as np
import pandas as pd

OUT_DIR = "data/cards"
OUT_CSV = os.path.join(OUT_DIR, "cards_top50.csv")

# 50 representative card SKUs across Pokémon/MTG/ Sports (demo names)
POKE = [
    "Pokémon Charizard Base Set Holo PSA 9",
    "Pokémon Lugia V Alt Art PSA 10",
    "Pokémon Pikachu Illustrator Promo (Replica Demo)",
    "Pokémon Umbreon Gold Star PSA 9",
    "Pokémon Neo Genesis Typhlosion 17 Holo PSA 9",
]
MTG = [
    "MTG Black Lotus Unlimited BGS 8",
    "MTG Mox Sapphire Unlimited PSA 8",
    "MTG Underground Sea Rev. PSA 9",
    "MTG Dual Land Volcanic Island PSA 9",
    "MTG Time Walk Unlimited PSA 8",
]
SPORTS = [
    "Jordan 1986 Fleer RC PSA 7",
    "LeBron 2003 Topps Chrome PSA 9",
    "Kobe 1996 Topps RC PSA 9",
    "Brady 2000 Bowman Chrome PSA 9",
    "Griffey 1989 Upper Deck PSA 9",
]

# Build 50 items by mixing and adding variants
ITEMS = []
def add_block(prefix, base_list):
    for n in base_list:
        ITEMS.append((n, 1999, 2007, "Graded", "PSA/BGS", "Card", 50.0, "eBay"))
        ITEMS.append((n + " - Alt", 2016, 2021, "Graded", "PSA 10", "Card", 120.0, "Goldin"))
for _ in range(3):
    add_block("Poke", POKE)
    add_block("MTG", MTG)
    add_block("Sports", SPORTS)
ITEMS = ITEMS[:50]

# dates monthly 2024-01 … 2025-12 (24 points)
dates = pd.date_range("2024-01-01", "2025-12-01", freq="MS")

rows = []
rng = np.random.default_rng(42)
for name, rel, ret, cond, grade, subtype, retail, source in ITEMS:
    # seed per item
    base = rng.uniform(80, 350)
    trend = rng.uniform(0.005, 0.02)  # 0.5%–2% monthly uptrend
    noise = rng.normal(0, 2, len(dates))
    prices = []
    level = base
    for i, d in enumerate(dates):
        level = level * (1 + trend) + noise[i]
        prices.append(max(10.0, float(level)))
    for d, p in zip(dates, prices):
        rows.append({
            "item_name": name,
            "release_year": rel,
            "retirement_year": ret,
            "condition": cond,
            "grade": grade,
            "category_subtype": subtype,
            "original_retail": retail,
            "source_platform": source,
            "date": d.strftime("%Y-%m-%d"),
            "price_usd": round(p, 2),
        })

df = pd.DataFrame(rows)
os.makedirs(OUT_DIR, exist_ok=True)
df.to_csv(OUT_CSV, index=False)
print(f"Wrote {OUT_CSV} with {len(df):,} rows and {df['item_name'].nunique()} items.")
