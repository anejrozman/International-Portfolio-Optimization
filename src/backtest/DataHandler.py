"""

Description: DataHandler class to load and process the investment universe, 
fully hedged returns, and excess currency returns for the UniversalBacktester framework.

Author: Anej Rozman
Last edited: 2026-03-12

"""

import pandas as pd
from typing import Tuple


class DataHandler:
    
    def __init__(self, universe_file: str, 
                 hedged_returns_file: str, 
                 excess_currency_returns_file: str):
        
        self.universe_file = universe_file
        self.hedged_returns_file = hedged_returns_file
        self.excess_currency_returns_file = excess_currency_returns_file
        self.asset_to_currency: dict[str, str] = {}
        
    def load_and_process(self) -> Tuple[pd.DataFrame, dict, pd.DataFrame]:

        # ----------------------------------------------------------------------------
        # Investment universe
        # ----------------------------------------------------------------------------
        universe_df = pd.read_csv(self.universe_file)
        
        # Determine column names and strip angle brackets
        ric_col = 'ric_code' if 'ric_code' in universe_df.columns else 'riccode'
        universe_df['clean_ric'] = universe_df[ric_col].astype(str).str.replace('<', '').str.replace('>', '')
        universe_df['date'] = pd.to_datetime(universe_df['date'])
        
        # Build static dictionary asset to currency mapping 
        sorted_universe = universe_df.sort_values('date', ascending=False)
        self.asset_to_currency = dict(zip(sorted_universe['clean_ric'], sorted_universe['currency']))
        
        # ----------------------------------------------------------------------------
        # Fully hedged daily returns
        # ----------------------------------------------------------------------------
        hedged_returns_df = pd.read_csv(self.hedged_returns_file)
        hedged_returns_df['date'] = pd.to_datetime(hedged_returns_df['date'])
        hedged_returns_df.set_index('date', inplace=True)
        hedged_returns_df.columns = hedged_returns_df.columns.str.replace('<', '', regex=False).str.replace('>', '', regex=False)
        
        # Keep only columns that exist in the asset_to_currency mapping (assets in investment universe)
        universe_assets = list(self.asset_to_currency.keys())
        available_assets = [col for col in hedged_returns_df.columns if col in universe_assets]
        asset_returns = hedged_returns_df[available_assets]
        
        # Create the Mask to keep only the active assets per year
        # Create a dataframe of 1s where the asset is selected on a specific rebalance date
        universe_df['selected'] = 1
        mask = universe_df.pivot_table(index='date', columns='clean_ric', values='selected')
        
        # Forward fill the mask so the 1s carry over to every daily row until the next rebalance year
        # The reindex aligns the mask's dates with the daily trading dates in asset_returns
        daily_mask = mask.reindex(asset_returns.index, method='ffill')
        
        # Where the mask is 1, the return stays. Where the mask is NaN, the return becomes NaN.
        final_active_returns = asset_returns * daily_mask
        
        # ----------------------------------------------------------------------------
        # Excess currency daily returns
        # ----------------------------------------------------------------------------
        excess_currency_returns_df = pd.read_csv(self.excess_currency_returns_file)
        excess_currency_returns_df['date'] = pd.to_datetime(excess_currency_returns_df['date'])
        excess_currency_returns_df.set_index('date', inplace=True)
        
        return final_active_returns, self.asset_to_currency, excess_currency_returns_df

