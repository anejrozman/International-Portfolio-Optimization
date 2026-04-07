"""

Description: Script to preprocess interim data and compute excess currency returns and 
fully hedged asset returns for 1m, 3m, and 6m forward contracts.

Author: Anej Rozman
Last edited: 2026-03-18

"""

import polars as pl
from pathlib import Path

def main():
    current_dir = Path(__file__).resolve().parent
    interim_dir = current_dir.parent.parent / "data" / "interim"
    processed_dir = current_dir.parent.parent / "data" / "processed"

    # Load Interim Data
    df_returns = pl.read_csv(interim_dir / "stockreturns.csv")
    df_spot = pl.read_csv(interim_dir / "exchangerates.csv")
    
    fwd_files = {
        "1m": pl.read_csv(interim_dir / "forwardrates_1m.csv"),
        "3m": pl.read_csv(interim_dir / "forwardrates_3m.csv"),
        "6m": pl.read_csv(interim_dir / "forwardrates_6m.csv")
    }

    date_col = "date"
    curr_cols = [c for c in df_spot.columns if c != date_col]

    # ---------------------------------------------------------
    # Local Asset Returns (R_{t+1})
    # ---------------------------------------------------------
    df_returns.write_csv(processed_dir / "local_asset_returns.csv")

    # ---------------------------------------------------------
    # Spot Currency Returns e_{t+1} = (S_{t+1} / S_t) - 1
    # ---------------------------------------------------------
    df_e = df_spot.with_columns([
        ((pl.col(c) / pl.col(c).shift(1)) - 1).alias(c) 
        for c in curr_cols
    ]).select([date_col] + curr_cols).slice(1)
    
    df_e.write_csv(processed_dir / "currency_spot_returns.csv")
    
    # ---------------------------------------------------------
    # Unscaled Raw Forward Premiums f_t^{raw} = (F_t - S_t) / S_t
    # ---------------------------------------------------------
    for tenor, df_fwd in fwd_files.items():
        df_f = df_spot.join(df_fwd, on=date_col, suffix="_fwd")
        df_f = df_f.with_columns([
            ((pl.col(f"{c}_fwd") - pl.col(c)) / pl.col(c)).alias(c) 
            for c in curr_cols
        ]).select([date_col] + curr_cols)

        df_f.write_csv(processed_dir / f"raw_forward_premiums_{tenor}.csv")

if __name__ == "__main__":
    main()
