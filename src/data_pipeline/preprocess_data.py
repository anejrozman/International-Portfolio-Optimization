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
    df_mapping = pl.read_csv(interim_dir / "ric_mapping.csv")
    
    fwd_files = {
        "1m": pl.read_csv(interim_dir / "forwardrates_1m.csv"),
        "3m": pl.read_csv(interim_dir / "forwardrates_3m.csv"),
        "6m": pl.read_csv(interim_dir / "forwardrates_6m.csv")
    }

    # Scales by 21 * trading_period (1, 3, or 6 months)
    period_scaling = {
        "1m": 21 * 1,
        "3m": 21 * 3,
        "6m": 21 * 6
    }

    date_col = "date"
    curr_cols = [c for c in df_spot.columns if c != date_col]
    assets = [c for c in df_returns.columns if c != date_col]

    ric_to_curr = dict(zip(df_mapping["ric_code"], df_mapping["currency"]))

    # ---------------------------------------------------------
    # Spot Currency Returns e_{t+1}
    # Pad missing values during pct_change
    # ---------------------------------------------------------
    df_e = df_spot.with_columns([
        ((pl.col(c).forward_fill() / pl.col(c).forward_fill().shift(1)) - 1).alias(f"E_{c}") 
        for c in curr_cols
    ]).select([date_col] + [f"E_{c}" for c in curr_cols])
    
    # ---------------------------------------------------------
    # Process Each Forward Contract 
    # ---------------------------------------------------------
    for tenor, df_fwd in fwd_files.items():
        scaling_factor = period_scaling[tenor]

        # Monthly Forward Premium f_t = (F_t - S_t) / S_t
        df_f = df_spot.join(df_fwd, on=date_col, suffix="_fwd")
        df_f = df_f.with_columns([
            (((pl.col(f"{c}_fwd") - pl.col(c)) / pl.col(c)) / scaling_factor).shift(1).alias(f"F_{c}") 
            for c in curr_cols
        ]).select([date_col] + [f"F_{c}" for c in curr_cols])

        # Join spot returns and forward premiums
        df_combined = df_e.join(df_f, on=date_col)

        # -------------------------------------------------------------
        # Excess Currency Returns (e_{t+1} - f_t)
        # -------------------------------------------------------------
        excess_exprs = [
            (pl.col(f"E_{c}") - pl.col(f"F_{c}")).alias(c) for c in curr_cols
        ]
        df_excess = df_combined.select([pl.col(date_col)] + excess_exprs).slice(1)
        
        # USD excess return is 0, so we drop it
        if "USD" in df_excess.columns:
            df_excess = df_excess.drop("USD")
            
        df_excess.write_csv(processed_dir / f"excesscurrencyreturns_{tenor}.csv")

        # -------------------------------------------------------------
        # Fully Hedged Asset Returns
        # -------------------------------------------------------------
        df_joined = df_returns.join(df_combined, on=date_col)

        fh_exprs = [pl.col(date_col)]
        for ric in assets:
            cur = ric_to_curr.get(ric, "USD")
            
            if cur in curr_cols:
                # R_fh = R_{t+1} + f_t + (R_{t+1} * e_{t+1})
                expr = pl.col(ric) + pl.col(f"F_{cur}") + (pl.col(ric) * pl.col(f"E_{cur}"))
            else:
                expr = pl.col(ric) # Fallback if currency mapping is missing
                
            fh_exprs.append(expr.alias(ric))

        df_fh = df_joined.select(fh_exprs).slice(1)
        df_fh.write_csv(processed_dir / f"fullyhedgedassetreturns_{tenor}.csv")

if __name__ == "__main__":
    main()
