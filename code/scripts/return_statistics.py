"""

Description: Script for obtaining summary statistics as those reported in 
table 1 in Regularized Multi-Currency Expected Shortfall Portfolios by Ulrych and Lucescu (2026)

Author: Anej Rozman
Last edited: 2026-04-27

"""

import os
import sys
import pandas as pd
import numpy as np

# Add the src directory to the path 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def main():

    processed_data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed')
    raw_data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw')
    
    # File paths
    universe_file = os.path.join(raw_data_dir, 'currency_alocation_50_202603081747.csv') 
    local_asset_returns_file = os.path.join(processed_data_dir, 'local_asset_returns.csv')
    currency_returns_file = os.path.join(processed_data_dir, 'currency_spot_returns.csv')
    forward_premium_file = os.path.join(processed_data_dir, 'raw_forward_premiums_1m.csv')

    # Load data
    local_asset_returns = pd.read_csv(local_asset_returns_file, index_col=0, parse_dates=True)
    currency_returns = pd.read_csv(currency_returns_file, index_col=0, parse_dates=True)
    forward_premiums = pd.read_csv(forward_premium_file, index_col=0, parse_dates=True)
    universe = pd.read_csv(universe_file)

    # ---------------------------------------------
    # Investable universe and asset to currency mapping
    # ---------------------------------------------    
    
    # Use the most recent currency assignment per RIC across all rebalancing dates
    universe['date'] = pd.to_datetime(universe['date'])
    sorted_universe = universe.sort_values('date', ascending=False)
    asset_to_currency = dict(zip(sorted_universe["ric_code"], sorted_universe['currency']))

    # ---------------------------------------------
    # Daily Excess Currency Returns
    # ---------------------------------------------

    # Scale the 1-month raw forward premium to a daily premium
    # and shift by 1 day to align the premium from t with the spot return from t+1
    daily_fwd_premium = (forward_premiums / 21).shift(1)

    # Excess Currency Return  e_{t+1} - f_t
    excess_currency_returns = currency_returns - daily_fwd_premium
    excess_currency_returns = excess_currency_returns.dropna(how='all')
    excess_currency_returns = excess_currency_returns.drop(columns=['USD'])

    # ---------------------------------------------
    # Daily Fully Hedged Asset Returns
    # ---------------------------------------------

    # Statistics are computed over each stock's full data history (no rebalancing mask),
    # matching the paper's description of "1,705 individual stocks from June 1990 to June 2023".
    # Observations with |return| > 2.0 are dropped as spurious data errors
    # (unadjusted corporate actions such as reverse splits produce single-day
    # returns of 100x–250x that would otherwise dominate the statistics).
    RETURN_FILTER = 2.0
    universe_rics = set(asset_to_currency.keys())
    fully_hedged_returns = {}
    for ric in local_asset_returns.columns:
        if ric not in universe_rics:
            continue
        cur = asset_to_currency.get(ric, 'USD')
        if cur == 'USD' or cur not in currency_returns.columns:
            s = local_asset_returns[ric].dropna()
        else:
            # R_fh = (1 + R_loc)(1 + f_t) - 1 = R_loc + f_t + R_loc * f_t
            s = (
                local_asset_returns[ric] +
                daily_fwd_premium[cur] +
                (local_asset_returns[ric] * daily_fwd_premium[cur])
            ).dropna()
        fully_hedged_returns[ric] = s[s.abs() <= RETURN_FILTER]

    # Average the annualized metrics across the stocks in each economy
    economy_asset_stats = pd.DataFrame(columns=['Asset_Return', 'Asset_Volatility'])
    for cur in set(asset_to_currency.values()):
        cur_rics = [r for r, c in asset_to_currency.items() if c == cur and r in fully_hedged_returns]
        if not cur_rics:
            continue
        cagr_list, vol_list = [], []
        for r in cur_rics:
            s = fully_hedged_returns[r]
            n = len(s)
            if n < 10:
                continue
            cagr_list.append(((1 + s).prod() ** (252 / n) - 1) * 100)
            vol_list.append(s.std() * np.sqrt(252) * 100)
        if cagr_list:
            economy_asset_stats.loc[cur] = [np.mean(cagr_list), np.mean(vol_list)]

    economy_asset_stats.index.name = 'Currency'
    
    # Currency Statistics
    # Annualized return and volatility for the excess currency returns
    currency_count = excess_currency_returns.count()
    currency_ann_ret = (((1 + excess_currency_returns).prod() ** (252 / currency_count)) - 1) * 100
    currency_ann_vol = excess_currency_returns.std() * np.sqrt(252) * 100
    
    currency_stats = pd.DataFrame({
        'Currency_Excess_Return': currency_ann_ret,
        'Currency_Excess_Volatility': currency_ann_vol
    })
    
    # USD is the base currency
    currency_stats.loc['USD'] = {'Currency_Excess_Return': 0.0, 'Currency_Excess_Volatility': 0.0}
    
    # Summary Table
    summary_table = economy_asset_stats.join(currency_stats)
    order = ['AUD', 'CAD', 'DKK', 'EUR', 'HKD', 'JPY', 'NZD', 'NOK', 'SGD', 'ZAR', 'SEK', 'CHF', 'GBP', 'USD']
    summary_table = summary_table.reindex(order)
    summary_table = summary_table.round(2)
    
    print("\nTable 1: Annualized Average Returns and Volatilities (%)")
    print("-" * 75)
    print(summary_table)
    

if __name__ == "__main__":
    main()
