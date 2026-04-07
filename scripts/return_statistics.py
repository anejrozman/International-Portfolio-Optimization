"""

Description: Script for obtaining summary statistics as those reported in 
table 1 in Regularized Multi-Currency Expected Shortfall Portfolios by Ulrych and Lucescu (2026)

Author: Anej Rozman
Last edited: 3.4.2026

"""

import os
import sys
import pandas as pd
import numpy as np

# Add the src directory to the path 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def main():

    processed_data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
    raw_data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
    
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
    
    # Determine column names and strip angle brackets
    ric_col = 'ric_code' if 'ric_code' in universe.columns else 'riccode'
    universe['clean_ric'] = universe[ric_col].astype(str).str.replace('<', '').str.replace('>', '')
    universe['date'] = pd.to_datetime(universe['date'])
    
    # Build static dictionary asset to currency mapping 
    sorted_universe = universe.sort_values('date', ascending=False)
    asset_to_currency = dict(zip(sorted_universe['clean_ric'], sorted_universe['currency']))
   
    local_asset_returns.columns = local_asset_returns.columns.str.replace('<', '', regex=False).str.replace('>', '', regex=False)

    # Create the Mask to keep only the active assets per year
    # Create a dataframe of 1s where the asset is selected on a specific rebalance date
    universe['selected'] = 1
    mask = universe.pivot_table(index='date', columns='clean_ric', values='selected').fillna(0)
    
    # Forward fill the mask so the 1s carry over to every daily row until the next rebalance year
    # The reindex aligns the mask's dates with the daily trading dates in asset_returns
    daily_mask = mask.reindex(local_asset_returns.index, method='ffill')
    daily_mask = daily_mask.replace(0, np.nan)
    
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
    
    fully_hedged_returns = pd.DataFrame(index=local_asset_returns.index, columns=local_asset_returns.columns)
    
    for ric in local_asset_returns.columns:
        cur = asset_to_currency.get(ric, 'USD')
        
        if cur == 'USD' or cur not in currency_returns.columns:
            fully_hedged_returns[ric] = local_asset_returns[ric]
        else:
            # R_fh = R_{loc} + f_t + (R_{loc} * e_{t+1})
            fully_hedged_returns[ric] = (
                local_asset_returns[ric] + 
                daily_fwd_premium[cur] + 
                (local_asset_returns[ric] * currency_returns[cur])
            )

    final_active_returns = (fully_hedged_returns[daily_mask.columns.intersection(fully_hedged_returns.columns)] * daily_mask).dropna(how='all')

    # ---------------------------------------------
    # Annualized Statistics 
    # ---------------------------------------------
    
    # Stock Statistics
    
    # Compute the fully hedged daily returns for all the assets, and then 
    # average these returns daily over the different economies
    daily_economy_returns = pd.DataFrame(index=final_active_returns.index)
    currencies = set(asset_to_currency.values())
    for cur in currencies:
        cur_rics = [ric for ric in final_active_returns.columns if asset_to_currency.get(ric, 'USD') == cur]
        if cur_rics:
            daily_economy_returns[cur] = final_active_returns[cur_rics].mean(axis=1)

    # Compute the annualized returns from these daily economy returns
    economy_asset_stats = pd.DataFrame({
        'Asset_Return': daily_economy_returns.mean() * 252 * 100,
        'Asset_Volatility': daily_economy_returns.std() * np.sqrt(252) * 100
    })
    economy_asset_stats.index.name = 'Currency'
    
    # Currency Statistics
    # Annualized return and volatility for the excess currency returns using geometric mean 
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
