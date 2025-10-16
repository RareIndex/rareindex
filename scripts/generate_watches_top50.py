# scripts/generate_watches_top50.py
import os
import random
import numpy as np
import pandas as pd

random.seed(7)
np.random.seed(7)

WATCHES = [
    # (Name, Reference, Release, Discontinued, Retail, Subtype)
    ("Rolex Submariner Date", "126610LN", 2020, None, 10100, "Diver"),
    ("Rolex Submariner No-Date", "114060", 2012, 2020, 7500, "Diver"),
    ("Rolex GMT-Master II Pepsi", "126710BLRO", 2018, None, 10800, "GMT"),
    ("Rolex Daytona Steel", "116500LN", 2016, 2023, 13950, "Chronograph"),
    ("Omega Speedmaster Pro Moonwatch", "310.30.42.50.01.001", 2021, None, 7600, "Chronograph"),
    ("Omega Seamaster Diver 300M", "210.30.42.20.01.001", 2018, None, 5900, "Diver"),
    ("Omega Aqua Terra 41", "220.10.41.21.02.001", 2017, None, 6200, "All-rounder"),
    ("Tudor Black Bay 58", "M79030N-0001", 2018, None, 3950, "Diver"),
    ("Tudor Pelagos 39", "M25407N-0001", 2022, None, 4700, "Diver"),
    ("Patek Philippe Nautilus", "5711/1A", 2006, 2021, 30000, "Luxury Sports"),
    ("Patek Philippe Aquanaut", "5167A", 2007, None, 23000, "Luxury Sports"),
    ("Audemars Piguet Royal Oak", "15510ST", 2022, None, 27500, "Luxury Sports"),
    ("Audemars Piguet Royal Oak Chrono", "26331ST", 2017, 2023, 32000, "Chronograph"),
    ("Vacheron Constantin Overseas", "4500V", 2016, None, 24000, "Luxury Sports"),
    ("Cartier Santos Large", "WSSA0018", 2018, None, 7600, "Dress Sports"),
    ("Cartier Tank Must Large", "WSTA0053", 2021, None, 3050, "Dress"),
    ("IWC Portugieser Chronograph", "IW371605", 2020, None, 8900, "Chronograph"),
    ("IWC Pilot Mark XX", "IW328201", 2022, None, 5950, "Pilot"),
    ("Jaeger-LeCoultre Reverso Classic Large", "Q3858520", 2016, None, 8800, "Dress"),
    ("Grand Seiko Snowflake", "SBGA211", 2010, None, 6000, "Spring Drive"),
    ("Grand Seiko White Birch", "SLGH005", 2021, None, 9200, "Hi-Beat"),
    ("Seiko Prospex Turtle", "SRP777", 2016, None, 475, "Diver"),
    ("Seiko 5 Sports", "SRPD55", 2019, None, 275, "Sports"),
    ("Longines Spirit 37", "L3.410.4.63.6", 2022, None, 2600, "Pilot"),
    ("Longines HydroConquest 41", "L3.781.4.56.6", 2018, None, 1975, "Diver"),
    ("Hamilton Khaki Field Mechanical", "H69439931", 2018, None, 595, "Field"),
    ("Hamilton Jazzmaster", "H32455557", 2018, None, 775, "Dress"),
    ("TAG Heuer Carrera", "CBN2010.BA0642", 2020, None, 6150, "Chronograph"),
    ("TAG Heuer Aquaracer 200", "WBP2111.BA0627", 2022, None, 2850, "Diver"),
    ("Breitling Navitimer 41", "A17326161C1A1", 2022, None, 7000, "Pilot"),
    ("Breitling Superocean 42", "A17375E71C1A1", 2022, None, 5300, "Diver"),
    ("Panerai Luminor Base Logo", "PAM01086", 2020, None, 6100, "Diver"),
    ("Panerai Submersible 42", "PAM00683", 2019, None, 9200, "Diver"),
    ("Zenith Chronomaster Sport", "03.3100.3600/69.M3100", 2021, None, 10800, "Chronograph"),
    ("Zenith Defy Skyline", "03.9300.3620/51.I001", 2022, None, 9000, "Sports"),
    ("Hublot Classic Fusion 42", "542.NX.7071.LR", 2015, None, 7700, "Dress Sports"),
    ("Tissot PRX Powermatic 80", "T137.407.11.051.00", 2021, None, 725, "Sports"),
    ("Tissot Seastar 1000", "T120.407.11.051.00", 2018, None, 775, "Diver"),
    ("Casio G-Shock Full Metal", "GMW-B5000D-1", 2018, None, 550, "Digital"),
    ("Casio G-Shock GA2100", "GA-2100-1A1", 2019, None, 99, "Digital"),
    ("Citizen Promaster Diver", "NY0040", 1997, None, 350, "Diver"),
    ("Citizen Tsuyosa", "NJ0150-81Z", 2022, None, 450, "Sports"),
    ("Baltic Aquascaphe", "BAWSC", 2019, None, 700, "Microbrand"),
    ("Baltic MR01", "BMR01", 2021, None, 545, "Microbrand"),
    ("Nomos Tangente 38", "164", 2015, None, 2350, "Dress"),
    ("Nomos Club Campus 36", "708", 2017, None, 1500, "Sports"),
    ("Mido Ocean Star 200C", "M042.430.11.091.00", 2022, None, 1100, "Diver"),
    ("Oris Aquis Date 41.5", "01 733 7730 4157-07 8 24 05PEB", 2017, None, 2400, "Diver"),
    ("Oris Big Crown Pointer Date", "01 754 7741 4065-07 5 20 58", 2018, None, 2100, "Pilot"),
    ("Damasko DS30", "DS30", 2018, None, 1300, "Field"),
]

def monthly_series(start_price, months=24, drift=0.12, vol=0.10):
    # geometric-ish random walk ending around drift
    steps = months - 1
    if steps <= 0:
        return [start_price]
    # target end factor ~ (1+drift)
    mu = np.log(1 + drift) / steps
    sigma = vol / np.sqrt(12)
    prices = [float(start_price)]
    for _ in range(steps):
        prices.append(prices[-1] * float(np.exp(np.random.normal(mu, sigma))))
    # small stabilization
    return [round(p, 2) for p in prices]

def main():
    os.makedirs("data/watches", exist_ok=True)

    rows = []
    months = pd.period_range("2024-01", "2025-12", freq="M").to_timestamp()
    source_pool = ["Chrono24", "eBay", "WatchCharts", "RedBar Sales", "Dealer Ask"]

    # Pick 50 from the list (if list > 50)
    picks = WATCHES[:50]

    for (name, ref, rel, disc, retail, subtype) in picks:
        # starting resale ~ 90–130% of retail for most, ± different for hype models
        hype = 1.15 if ("Rolex" in name or "Patek" in name or "Audemars" in name or "AP" in name) else 1.0
        start_mult = np.clip(np.random.normal(1.0 * hype, 0.12), 0.75, 1.6)
        start_price = max(200.0, retail * start_mult)

        # drift: hype pieces 0–20%, others −5%–+15% over 2y
        base_drift = np.random.uniform(-0.05, 0.20) * (1.2 if hype > 1.0 else 1.0)
        vol = np.random.uniform(0.08, 0.18)

        path = monthly_series(start_price, months=len(months), drift=base_drift, vol=vol)

        condition = random.choice(["New", "Unworn", "Pre-owned", "Pre-owned"])
        grade = random.choice(["Full Set", "Box & Papers", "Watch Only"])
        source = random.choice(source_pool)

        for dt, px in zip(months, path):
            rows.append({
                "item_name": f"{name} ({ref})",
                "release_year": rel,
                "retirement_year": disc if disc is not None else "",
                "condition": condition,
                "grade": grade,
                "category_subtype": subtype,
                "original_retail": retail,
                "source_platform": source,
                "date": dt.date().isoformat(),
                "price_usd": round(px, 2),
            })

    df = pd.DataFrame(rows)
    df.to_csv("data/watches/watches_top50.csv", index=False)
    print(f"Wrote data/watches/watches_top50.csv with {len(df):,} rows and {df['item_name'].nunique()} items.")

if __name__ == "__main__":
    main()
