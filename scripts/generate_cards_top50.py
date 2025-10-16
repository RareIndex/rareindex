import os, random
from datetime import datetime
import pandas as pd
import numpy as np

OUT_DIR = "data/cards"
OUT_CSV = os.path.join(OUT_DIR, "cards_top50.csv")

# 50 named items across Pokémon / MTG / Yu-Gi-Oh! / Sports TCG
ITEMS = [
    # name, release_year, retirement_year, condition, grade, subtype, orig_retail, source
    ("Pokémon Charizard Base Set Holo PSA 9", 1999, 2000, "Graded", "PSA 9", "Pokemon", 3.99, "eBay"),
    ("Pokémon Lugia Neo Genesis 1st Ed PSA 9", 2000, 2001, "Graded", "PSA 9", "Pokemon", 3.99, "PWCC"),
    ("Pokémon Umbreon Gold Star PSA 9", 2005, 2007, "Graded", "PSA 9", "Pokemon", 3.99, "Goldin"),
    ("Pokémon Pikachu Illustrator Promo (Reprint) PSA 8", 2023, 2023, "Graded", "PSA 8", "Pokemon", 0.00, "eBay"),
    ("Pokémon Charizard UPC Metal PSA 10", 2021, 2022, "Graded", "PSA 10", "Pokemon", 0.00, "eBay"),
    ("Pokémon 151 Charizard SAR PSA 10", 2023, 2024, "Graded", "PSA 10", "Pokemon", 0.00, "TCGPlayer"),
    ("Pokémon Celebrations Charizard PSA 10", 2021, 2022, "Graded", "PSA 10", "Pokemon", 0.00, "eBay"),
    ("Pokémon Evolving Skies Umbreon VMAX Alt PSA 10", 2021, 2023, "Graded", "PSA 10", "Pokemon", 0.00, "Goldin"),
    ("Pokémon Crown Zenith Mewtwo VSTAR SAR PSA 10", 2023, 2024, "Graded", "PSA 10", "Pokemon", 0.00, "eBay"),
    ("Pokémon Skyridge Crystal Charizard PSA 8", 2003, 2004, "Graded", "PSA 8", "Pokemon", 3.99, "PWCC"),

    ("MTG Black Lotus (Unlimited) BGS 9", 1993, 1994, "Graded", "BGS 9", "MTG", 2.45, "PWCC"),
    ("MTG Mox Sapphire (Unlimited) BGS 8.5", 1993, 1994, "Graded", "BGS 8.5", "MTG", 2.45, "PWCC"),
    ("MTG Dual Land Underground Sea (Revised) PSA 9", 1994, 1995, "Graded", "PSA 9", "MTG", 3.95, "eBay"),
    ("MTG Time Walk (Unlimited) BGS 8.5", 1993, 1994, "Graded", "BGS 8.5", "MTG", 2.45, "Goldin"),
    ("MTG Gaea's Cradle (Urza’s Saga) PSA 9", 1998, 1999, "Graded", "PSA 9", "MTG", 3.99, "TCGPlayer"),
    ("MTG Jeweled Lotus (Extended Art) PSA 10", 2020, 2021, "Graded", "PSA 10", "MTG", 0.00, "eBay"),
    ("MTG Ragavan, Nimble Pilferer (Etched) PSA 10", 2021, 2022, "Graded", "PSA 10", "MTG", 0.00, "TCGPlayer"),
    ("MTG The One Ring (Elven Script) PSA 9", 2023, 2023, "Graded", "PSA 9", "MTG", 0.00, "eBay"),
    ("MTG Sheoldred, the Apocalypse Foil PSA 10", 2022, 2023, "Graded", "PSA 10", "MTG", 0.00, "eBay"),
    ("MTG Wasteland (Tempest) PSA 9", 1997, 1998, "Graded", "PSA 9", "MTG", 3.99, "PWCC"),

    ("Yu-Gi-Oh! Blue-Eyes White Dragon LOB 1st Ed PSA 8", 2002, 2003, "Graded", "PSA 8", "YuGiOh", 3.29, "Goldin"),
    ("Yu-Gi-Oh! Dark Magician LOB 1st Ed PSA 9", 2002, 2003, "Graded", "PSA 9", "YuGiOh", 3.29, "PWCC"),
    ("Yu-Gi-Oh! Red-Eyes B. Dragon LOB 1st Ed PSA 8", 2002, 2003, "Graded", "PSA 8", "YuGiOh", 3.29, "eBay"),
    ("Yu-Gi-Oh! Blue-Eyes Toon Dragon PSA 9", 2003, 2004, "Graded", "PSA 9", "YuGiOh", 3.29, "PWCC"),
    ("Yu-Gi-Oh! Chaos Emperor Dragon Envoy PSA 9", 2003, 2004, "Graded", "PSA 9", "YuGiOh", 3.29, "eBay"),
    ("Yu-Gi-Oh! Dark Magician Girl (Magician’s Force) PSA 9", 2003, 2004, "Graded", "PSA 9", "YuGiOh", 3.29, "Goldin"),
    ("Yu-Gi-Oh! Blue-Eyes White Dragon SDK PSA 10", 2002, 2003, "Graded", "PSA 10", "YuGiOh", 3.29, "eBay"),
    ("Yu-Gi-Oh! Black Luster Soldier Envoy PSA 10", 2004, 2005, "Graded", "PSA 10", "YuGiOh", 3.29, "PWCC"),
    ("Yu-Gi-Oh! Judgment Dragon (Ghost Rare) PSA 9", 2008, 2009, "Graded", "PSA 9", "YuGiOh", 3.29, "eBay"),
    ("Yu-Gi-Oh! Stardust Dragon (Ghost Rare) PSA 9", 2008, 2009, "Graded", "PSA 9", "YuGiOh", 3.29, "eBay"),

    ("1996 Kobe Bryant Topps #138 PSA 9", 1996, 1997, "Graded", "PSA 9", "Sports", 1.99, "Goldin"),
    ("1989 Ken Griffey Jr. Upper Deck #1 PSA 9", 1989, 1990, "Graded", "PSA 9", "Sports", 1.00, "PWCC"),
    ("2003 LeBron James Topps #221 PSA 9", 2003, 2004, "Graded", "PSA 9", "Sports", 1.99, "eBay"),
    ("2018 Luka Dončić Prizm Silver PSA 10", 2018, 2019, "Graded", "PSA 10", "Sports", 2.99, "eBay"),
    ("2000 Tom Brady Bowman #236 PSA 9", 2000, 2001, "Graded", "PSA 9", "Sports", 1.29, "PWCC"),
    ("2019 Shohei Ohtani Topps Chrome PSA 10", 2019, 2020, "Graded", "PSA 10", "Sports", 2.99, "eBay"),
    ("2011 Mike Trout Update US175 PSA 9", 2011, 2012, "Graded", "PSA 9", "Sports", 2.99, "Goldin"),
    ("2017 Patrick Mahomes Donruss Rated Rookie PSA 10", 2017, 2018, "Graded", "PSA 10", "Sports", 2.99, "eBay"),
    ("2018 Jayson Tatum Prizm Silver PSA 10", 2018, 2019, "Graded", "PSA 10", "Sports", 2.99, "eBay"),
    ("1993 Derek Jeter SP Foil PSA 8", 1993, 1994, "Graded", "PSA 8", "Sports", 1.99, "PWCC"),

    # Fill to 50 with additional modern/popular TCG pieces
    ("Pokémon Mew Gold Star PSA 9", 2006, 2007, "Graded", "PSA 9", "Pokemon", 3.99, "eBay"),
    ("Pokémon Rayquaza EX Deoxys PSA 8", 2005, 2006, "Graded", "PSA 8", "Pokemon", 3.99, "PWCC"),
    ("Pokémon Charizard VMAX Shiny PSA 10", 2020, 2021, "Graded", "PSA 10", "Pokemon", 0.00, "eBay"),
    ("MTG Liliana of the Veil (MM3) PSA 10", 2017, 2018, "Graded", "PSA 10", "MTG", 0.00, "TCGPlayer"),
    ("MTG Force of Will (Alliances) PSA 9", 1996, 1997, "Graded", "PSA 9", "MTG", 3.99, "PWCC"),
    ("Yu-Gi-Oh! Blue-Eyes Alternative Dragon PSA 10", 2016, 2017, "Graded", "PSA 10", "YuGiOh", 0.00, "eBay"),
    ("Yu-Gi-Oh! Dark Armed Dragon PSA 9", 2008, 2009, "Graded", "PSA 9", "YuGiOh", 3.29, "PWCC"),
    ("2013 Giannis Antetokounmpo Prizm PSA 10", 2013, 2014, "Graded", "PSA 10", "Sports", 2.99, "Goldin"),
    ("2014 Lionel Messi Prizm World Cup PSA 10", 2014, 2015, "Graded", "PSA 10", "Sports", 2.99, "eBay"),
    ("2018 Kylian Mbappé Prizm World Cup PSA 10", 2018, 2019, "Graded", "PSA 10", "Sports", 2.99, "eBay"),
]

def monthly_dates():
    return pd.date_range("2024-01-01", "2025-12-01", freq="MS")

def synth_series(n, start_price, drift=0.18, vol=0.06):
    """Geometric-ish random walk: positive, trending."""
    prices = [start_price]
    months = n - 1
    # set per-month drift so total ~ drift over the period
    mu = (1.0 + drift) ** (1.0 / max(months, 1)) - 1.0
    for _ in range(months):
        shock = np.random.normal(mu, vol / np.sqrt(12))
        prices.append(max(1.0, prices[-1] * (1.0 + shock)))
    return np.round(prices, 2)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []

    dates = monthly_dates()

    for (name, rel, ret, cond, grade, subtype, retail, source) in ITEMS:
        # pick a plausible graded starting price per sub-type
        base = {
            "Pokemon": (120, 900),
            "MTG": (150, 1200),
            "YuGiOh": (100, 800),
            "Sports": (80, 2000),
        }.get(subtype, (80, 600))
        start = random.uniform(*base)
        drift = random.uniform(0.05, 0.65)   # 5%–65% total 24-mo drift
        vol   = random.uniform(0.04, 0.10)

        series = synth_series(len(dates), start, drift=drift, vol=vol)

        for d, p in zip(dates, series):
            rows.append({
                "item_name": name,
                "release_year": rel,
                "retirement_year": ret,
                "condition": cond,
                "grade": grade,
                "category_subtype": subtype,
                "original_retail": float(retail),
                "source_platform": source,
                "date": d.strftime("%Y-%m-%d"),
                "price_usd": float(p),
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV} with {len(df):,} rows and {df['item_name'].nunique()} items.")

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    main()
