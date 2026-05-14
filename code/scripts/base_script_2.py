"""

Description: Runs an equally weighted portfolio through the new calendar-based Backtester
with quarterly rebalancing and 3m forward hedging. Uses the top-50 stock universe.
Computes R_fh and R_c from processed files before passing them to the DataHandler.

Author: Anej Rozman
Last edited: -

"""

import os
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.backtest.Backtester import Backtester
from src.backtest.DataHandler import DataHandler
from src.optimizers.EqualWeightOptimizer import EqualWeightOptimizer


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
script_dir      = os.path.dirname(__file__)
data_dir        = os.path.abspath(os.path.join(script_dir, '..', '..', 'data'))
raw_dir         = os.path.join(data_dir, 'raw')
processed_dir   = os.path.join(data_dir, 'processed')
results_dir     = os.path.abspath(os.path.join(script_dir, '..', '..', 'results'))
plots_dir       = os.path.join(results_dir, 'plots')
runs_dir        = os.path.join(results_dir, 'runs')

universe_file           = os.path.join(raw_dir,       'currency_alocation_50_202603081747.csv')
local_returns_file      = os.path.join(processed_dir, 'local_asset_returns.csv')
spot_returns_file       = os.path.join(processed_dir, 'currency_spot_returns.csv')
forward_premiums_file   = os.path.join(processed_dir, 'raw_forward_premiums_3m.csv')
rfh_out_file            = os.path.join(processed_dir, 'fullyhedgedassetreturns_3m.csv')
rc_out_file             = os.path.join(processed_dir, 'excesscurrencyreturns_3m.csv')
rf_file                 = os.path.join(raw_dir,       'us_risk_free_rates.csv')


# ---------------------------------------------------------------------------
# Run directory setup (auto-incrementing: equally_weighted_run_1, run_2, ...)
# ---------------------------------------------------------------------------
os.makedirs(runs_dir, exist_ok=True)
existing_nums = []
for d in os.listdir(runs_dir):
    if d.startswith('equally_weighted_run_'):
        try:
            existing_nums.append(int(d.replace('equally_weighted_run_', '')))
        except ValueError:
            pass
run_number = max(existing_nums, default=0) + 1
run_dir = os.path.join(runs_dir, f'equally_weighted_run_{run_number}')
os.makedirs(run_dir, exist_ok=True)


# ---------------------------------------------------------------------------
# Step 1: Load processed data
# ---------------------------------------------------------------------------
def load_df(path):
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    # Strip angle brackets from column names if present (Refinitiv Datastream format)
    df.columns = df.columns.str.replace('<', '', regex=False).str.replace('>', '', regex=False)
    return df

R_local     = load_df(local_returns_file)       # daily local equity returns
e           = load_df(spot_returns_file)         # e_{t+1}: spot currency returns at t+1
f_raw       = load_df(forward_premiums_file)     # f_t:    3m forward premium at t

# Load daily risk-free rates (Fama-French, in % → convert to decimal)
rf_raw = pd.read_csv(rf_file, index_col=0, parse_dates=True)
rf_raw.index = pd.to_datetime(rf_raw.index, format='%Y%m%d')
rf_series = rf_raw['RF'] / 100.0   # daily RF in decimal form

# Filter: replace single-day returns with |r| > 50% with NaN.
# The 200% threshold used in return_statistics.py is too lenient — OOS data contains
# repeated returns of exactly +100% or +200% for certain Singapore/SA small-caps
# (e.g. MDRT.SI: 120 days at +100%, SOUE.SI: 108 days at +200%), which are
# unadjusted corporate actions rather than real price moves.
RETURN_FILTER = 1.0
R_local = R_local.where(R_local.abs() < RETURN_FILTER)


# ---------------------------------------------------------------------------
# Step 2: Build asset-to-currency mapping from the top-50 universe file
# ---------------------------------------------------------------------------
universe_df = pd.read_csv(universe_file)
ric_col = 'ric_code' if 'ric_code' in universe_df.columns else 'riccode'
universe_df['clean_ric'] = (
    universe_df[ric_col].astype(str)
    .str.replace('<', '', regex=False)
    .str.replace('>', '', regex=False)
)
# Use the most recent entry per RIC to get a stable mapping
sorted_univ = universe_df.sort_values('date', ascending=False)
asset_to_currency_raw = dict(zip(sorted_univ['clean_ric'], sorted_univ['currency']))


# ---------------------------------------------------------------------------
# Step 3: Compute excess currency returns R_c = e_{t+1} - f_t
#
# f_raw contains the 3-MONTH forward premium observed at date t.
# We divide by 63 (trading days per quarter) to get the DAILY forward premium,
# so that the formula applies correctly at daily frequency.
# f_raw is then shifted forward by 1 row so that f_t aligns with the return date t+1.
# ---------------------------------------------------------------------------
TENOR_DAYS = 63                                  # 3m forward ≈ 63 trading days
f_daily = f_raw / TENOR_DAYS                     # scale 3m → daily forward premium
f_shifted = f_daily.shift(1)                     # f_t now indexed at t+1

# Align on common dates
common_dates = e.index.intersection(f_shifted.index).dropna()
common_ccy   = e.columns.intersection(f_shifted.columns)
R_c = (e.loc[common_dates, common_ccy] - f_shifted.loc[common_dates, common_ccy]).dropna(how='all')


# ---------------------------------------------------------------------------
# Step 4: Compute fully hedged asset returns
#
# R^fh_{i,t+1} = R_{i,t+1} + f_{c(i),t} + R_{i,t+1} * e_{c(i),t+1}
#
# f_shifted and e are already aligned to date t+1.
# ---------------------------------------------------------------------------
# Restrict R_local to assets with a known currency mapping
known_assets = [a for a in R_local.columns if a in asset_to_currency_raw]

all_dates  = R_local.index.intersection(f_shifted.index).intersection(e.index)
R_local_a  = R_local.loc[all_dates, known_assets]
f_s        = f_shifted.loc[all_dates]
e_a        = e.loc[all_dates]

# Build per-asset currency-indexed slices of f and e
f_asset = pd.DataFrame(index=all_dates, columns=known_assets, dtype=float)
e_asset = pd.DataFrame(index=all_dates, columns=known_assets, dtype=float)

for asset in known_assets:
    ccy = asset_to_currency_raw[asset]
    if ccy in f_s.columns:
        f_asset[asset] = f_s[ccy].values
    if ccy in e_a.columns:
        e_asset[asset] = e_a[ccy].values

R_fh = R_local_a + f_asset + R_local_a * e_asset
R_fh.dropna(how='all', inplace=True)


# ---------------------------------------------------------------------------
# Step 5: Save computed R_fh and R_c so DataHandler can read them
# ---------------------------------------------------------------------------
R_fh.index.name = 'date'
R_c.index.name  = 'date'
R_fh.to_csv(rfh_out_file)
R_c.to_csv(rc_out_file)


# ---------------------------------------------------------------------------
# Step 6: Load via DataHandler and build point-in-time universe mask
# ---------------------------------------------------------------------------
handler = DataHandler(
    universe_file=universe_file,
    hedged_returns_file=rfh_out_file,
    excess_currency_returns_file=rc_out_file,
)
R_fh_filtered, asset_to_currency, R_c_filtered, universe_mask = handler.load_and_process()

# Align to common dates
common = R_fh_filtered.index.intersection(R_c_filtered.index)
R_fh_filtered = R_fh_filtered.loc[common]
R_c_filtered  = R_c_filtered.loc[common]

# Align universe mask to the same date index (forward-fill so every trading
# day inherits the most-recent annual universe snapshot).
universe_mask = universe_mask.reindex(common, method='ffill')


# ---------------------------------------------------------------------------
# Step 7: Run backtest
# ---------------------------------------------------------------------------
data_bundle = {
    'R_fh':             R_fh_filtered,
    'R_c':              R_c_filtered,
    'asset_to_currency': asset_to_currency,
    'universe_mask':    universe_mask,
}

backtester = Backtester(
    optimizer=EqualWeightOptimizer,
    param_grid=None,
    window_years=10,
    val_years=1,
    rebalancing_frequency='quarterly',
    logging=True
)

config = {
    "strategy": "EQUALLY-WEIGHTED",
    "run_number": run_number,
    "run_date": datetime.now().isoformat(),
    "optimizer": "EqualWeightOptimizer",
    "param_grid": None,
    "window_years": 10,
    "val_years": 1,
    "rebalancing_frequency": "quarterly",
    "asset_return_key": "R_fh",
    "universe_file": os.path.basename(universe_file),
    "return_filter": RETURN_FILTER,
    "rf_source": os.path.basename(rf_file),
}
with open(os.path.join(run_dir, 'config.json'), 'w') as f:
    json.dump(config, f, indent=2)

oos_returns = backtester.run(data_bundle)


# ---------------------------------------------------------------------------
# Step 8: Save OOS returns
# ---------------------------------------------------------------------------
oos_df = oos_returns.rename('return').to_frame()
oos_df.index.name = 'date'
oos_df.to_csv(os.path.join(run_dir, 'oos_returns.csv'))


# ---------------------------------------------------------------------------
# Step 9: Performance metrics
# ---------------------------------------------------------------------------
metrics = backtester.compute_performance_metrics(
    oos_returns,
    label="1/N Equal Weight — Fully Hedged (3m, top-50)",
    rf_series=rf_series,
)

with open(os.path.join(run_dir, 'metrics.json'), 'w') as f:
    json.dump(metrics, f, indent=2)


# ---------------------------------------------------------------------------
# Step 10: Plot cumulative returns
# ---------------------------------------------------------------------------
cumulative_returns = (1 + oos_returns).cumprod()

plt.figure(figsize=(10, 6))
cumulative_returns.plot(label='Equal Weight Fully Hedged (3m, top-50)')
plt.title('Cumulative Returns — Equally Weighted Portfolio')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.grid(True)
plt.legend()
plt.tight_layout()

plot_path = os.path.join(run_dir, 'cumulative_returns_plot.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to: {plot_path}")
print(f"Run artifacts saved to: {run_dir}")
# plt.show()


if __name__ == '__main__':
    pass
