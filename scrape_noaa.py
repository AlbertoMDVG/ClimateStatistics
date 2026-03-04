import requests
import numpy as np
import pandas as pd
import json
import sys


BASE_URL = "https://storage.googleapis.com/noaa-ncei-ipg/datasets/cag/data/mapping/global"
COORDS_URL = "https://www.ncei.noaa.gov/monitoring-content/cag/metadata/global-grid-coords.json"


# ==========================
# Utilities
# ==========================

def month_range(start_y, start_m, end_y, end_m):
    months = []
    y, m = start_y, start_m
    while (y < end_y) or (y == end_y and m <= end_m):
        months.append(f"{y}{m:02d}")
        if m == 12:
            y += 1
            m = 1
        else:
            m += 1
    return months


def load_grid_coords():
    coords = requests.get(COORDS_URL).json()
    return {
        key: (
            np.array(coords[key]["latitudes"]),
            np.array(coords[key]["longitudes"]),
        )
        for key in coords
    }


def fetch_month(variable, mode, yyyymm):
    url = f"{BASE_URL}/{variable}/{mode}_{yyyymm}.json"
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return np.array(r.json())


# ==========================
# Builder
# ==========================

def build_single_dataset(config, variable, mode, coords):

    months = month_range(
        config["START_YEAR"],
        config["START_MONTH"],
        config["END_YEAR"],
        config["END_MONTH"]
    )

    lats, lons = coords[variable]
    n_cells = len(lats) * len(lons)

    data_rows = []
    valid_months = []

    for ym in months:
        print(f"[{variable} {mode}] Processing {ym}")

        arr = fetch_month(variable, mode, ym)

        if arr is None:
            print(f"  Missing — skipping")
            continue

        if len(arr) != n_cells:
            print(f"  Unexpected grid size — skipping")
            continue

        data_rows.append(arr)
        valid_months.append(ym)

    if not data_rows:
        print(f"No data found for {variable} {mode}")
        return None

    data_matrix = np.vstack(data_rows)

    df = pd.DataFrame(
        data_matrix,
        index=valid_months
    )

    df.index.name = "YYYYMM"

    return df


# ==========================
# Entry Point
# ==========================

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python scrape_cag_global.py config.json")
        sys.exit(1)

    config_path = sys.argv[1]

    with open(config_path, "r") as f:
        config = json.load(f)

    coords = load_grid_coords()

    for combo in config["DATASETS"]:

        variable = combo["variable"]
        mode = combo["mode"]

        df = build_single_dataset(config, variable, mode, coords)

        if df is None:
            continue

        output_name = f"{variable}_{mode}.parquet"
        df.to_parquet(output_name)

        print(f"Saved {output_name}")