import os, random
from datetime import datetime
import pandas as pd
import numpy as np

OUT_DIR = "data/cards"
OUT_CSV = os.path.join(OUT_DIR, "cards_top50.csv")

# name, release_year, retirement_year, condition, grade, subtype, retail, source
ITEMS = [
    # Pokémon (mix of graded and raw)
    ("Pokémon Lugia V Alt Art (SWSH Lugia V #186)", 2022, 2023, "Graded", "PSA 10", "Pokemon TCG", 99.99, "eBay"),
    ("Pokémon Charizard Base Set Holo", 1999, 2000, "Graded", "PSA 9", "Pokemon TCG", 99.99, "PWCC"),
    ("Pokémon Umbreon Gold Star", 2005, 2006, "Graded", "BGS 9.5", "Pokemon TCG", 99.99, "Goldin"),
    ("Pokémon Pikachu Promo 20th Anniv.", 2016, 2017, "Raw", "NM", "Pokemon TCG", 9.99, "eBay"),
    ("Pokémon Evolving Skies Booster Box", 2021, 2022, "Sealed", "Factory", "Pokemon TCG", 143.00, "StockX"),
    ("Pokémon 151 Booster Box JP", 2023, 2024, "Sealed", "Factory", "Pokemon TCG", 85.00, "Mercari JP"),
    ("Pokémon Crown Zenith ETB", 2023, 2024, "Sealed", "Factory", "Pokemon TCG", 49.99, "TCGplayer"),
    ("Pokémon Charizard UPC (SWSH)", 2022, 2023, "Sealed", "Factory", "Pokemon TCG", 119.99, "Amazon"),
    ("Pokémon Neo Genesis Typhlosion 17 Holo", 2000, 2001, "Graded", "PSA 9", "Pokemon TCG", 99.99, "eBay"),
    ("Pokémon Japanese Promo Mario Pikachu", 2016, 2017, "Graded", "PSA 10", "Pokemon TCG", 25.00, "Yahoo JP"),

    # Magic: The Gathering
    ("MTG Black Lotus (Unlimited)", 1993, 1994, "Graded", "BGS 8.5", "Magic", 100.00, "Heritage"),
    ("MTG Mox Sapphire (Unlimited)", 1993, 1994, "Graded", "PSA 8", "Magic", 50.00, "Heritage"),
    ("MTG Underground Sea (Revised)", 1994, 1994, "Graded", "PSA 9", "Magic", 7.00, "eBay"),
    ("MTG Modern Horizons 2 Booster Box", 2021, 2022, "Sealed", "Factory", "Magic", 250.00, "TCGplayer"),
    ("MTG Commander Masters Collector Box", 2023, 2024, "Sealed", "Factory", "Magic", 250.00, "TCGplayer"),

    # Sports cards
    ("1996 Topps Kobe Bryant RC #138", 1996, 1997, "Graded", "PSA 9", "Sports", 1.29, "eBay"),
    ("2018 Prizm Luka Doncic Silver RC", 2018, 2019, "Graded", "PSA 10", "Sports", 3.99, "Goldin"),
    ("2003 Topps Chrome LeBron James RC", 2003, 2004, "Graded", "PSA 10", "Sports", 3.99, "PWCC"),
    ("2011 Topps Update Mike Trout RC", 2011, 2012, "Graded", "PSA 9", "Sports", 2.99, "eBay"),
    ("2000 Playoff Contenders Tom Brady Auto RC", 2000, 2001, "Graded", "BGS 9", "Sports", 150.00, "Goldin"),

    # Yu-Gi-Oh!
    ("Yu-Gi-Oh! Blue-Eyes White Dragon (LOB-001) 1st Ed.", 2002, 2003, "Graded", "PSA 8", "Yu-Gi-Oh!", 3.00, "eBay"),
    ("Yu-Gi-Oh! Dark Magician Girl (MFC-000) 1st Ed.", 2003, 2004, "Graded", "PSA 9", "Yu-Gi-Oh!", 3.00, "eBay"),
    ("Yu-Gi-Oh! Legend of Blue Eyes Booster Box", 2002, 2003, "Sealed", "Factory", "Yu-Gi-Oh!", 69.99, "Heritage"),

    # One Piece / DBZ / Others
    ("One Piece Romance Dawn Booster Box JP", 2022, 2023, "Sealed", "Factory", "One Piece", 60.00, "Mercari JP"),
    ("Dragon Ball Z Score 2000 Goku Ultra Rare", 2000, 2001, "Raw", "NM", "DBZ", 3.00, "eBay"),
    ("Lorcana The First Chapter Booster Box", 2023, 2023, "Sealed", "Factory", "Disney Lorcana", 143.64, "TCGplayer"),
]

# Fill up to ~50 items with believable variants
while len(ITEMS) < 50:
    n = len(ITEMS) + 1
    ITEMS.append((
        f"Pokémon Modern Alt-Art #{n:03d}",
        random.choice(range(2018, 2024)),
        random.choice(range(2019, 2025)),
        random.choice(["Raw","Graded","Sealed"]),
        random.choice(["NM","PSA 9","PSA 10","BGS 9.5","Factory"]),
        "Pokemon TCG",
        random.choice([49.99, 99.99, 119.99, 143.00, 200.00]),
        random.choice(["eBay","TCGplayer","StockX","PWCC"])
    ))

def month_range():
    # 24 months: 2024-01 ... 2025-12
    dates = pd.period_range("2024-01", "2025-12", freq="M").to_timestamp("M")
    return dates

def synth_series(n_points, start_price):
    # random CAGR between -5% and +60% over 2 years
    cagr = random.uniform(-0.05, 0.60)
    end_mult = (1.0 + cagr) ** 2.0
    # smooth compounding + a bit of noise
    base = np.geomspace(start_price, start_price * end_mult, num=n_points)
    noise = np.random.normal(0, 0.02, size=n_points)  # +/-2% wiggle
    series = base * (1 + noise)
    # keep reasonable bounds
    series = np.clip(series, a_min=max(3.0, start_price*0.5), a_max=start_price*10)
    return series

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    dates = month_range()
    for (name, rel, ret, cond, grade, subtype, retail, source) in ITEMS:
        # a starting market price guess
        start_guess = {
            "Graded": random.uniform(150, 12000),
            "Sealed": random.uniform(80, 800),
            "Raw": random.uniform(20, 400)
        }.get(cond, 200.0)

        prices = synth_series(len(dates), start_guess)
        for d, p in zip(dates, prices):
            rows.append({
                "item_name": name,
                "release_year": rel,
                "retirement_year": ret,
                "condition": cond,
                "grade": grade,
                "category_subtype": subtype,
                "original_retail": float(retail),
                "source_platform": source,
                "date": pd.to_datetime(d).date().isoformat(),
                "price_usd": round(float(p), 2)
            })
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV} with {len(df):,} rows and {df['item_name'].nunique()} items.")
if __name__ == "__main__":
    main()
