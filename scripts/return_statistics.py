"""

Description: Script for obtaining summary statistics as those reported in 
table 1 in Regularized Multi-Currency Expected Shortfall Portfolios by Ulrych and Lucescu (2026)

Author: Anej Rozman
Last edited: -

"""

import os
import sys
import pandas as pd
import numpy as np

# Add the src directory to the path so we can import from it
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.backtest.DataHandler import DataHandler

def main():

    processed_data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
    raw_data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
    
    # File paths
    universe_file = os.path.join(raw_data_dir, 'currency_alocation_50_202603081747.csv') 
    hedged_returns_file = os.path.join(processed_data_dir, 'fullyhedgedassetreturns_3m.csv')
    excess_currency_returns_file = os.path.join(processed_data_dir, 'excesscurrencyreturns_3m.csv')
    
    # Initialize and run the DataHandler
    handler = DataHandler(
        universe_file=universe_file,
        hedged_returns_file=hedged_returns_file,
        excess_currency_returns_file=excess_currency_returns_file
    )
    
    R_fh, asset_to_currency, R_c = handler.load_and_process()
    
    # Ensure they have the same dates
    common_dates = R_fh.index.intersection(R_c.index)
    R_fh = R_fh.loc[common_dates]
    R_c = R_c.loc[common_dates]

    # Calculate annualized average return and volatility for each stock using 252 trading days
    stock_returns = R_fh.mean() * 252 * 100
    stock_vols = R_fh.std() * np.sqrt(252) * 100
    
    # Store stock statistics and map them to their respective currencies
    stock_stats = pd.DataFrame({'Return': stock_returns, 'Volatility': stock_vols})
    stock_stats['Currency'] = stock_stats.index.map(asset_to_currency)
    
    # Average the stock statistics across all single stocks denominated in the local currency
    stock_stats_by_currency = stock_stats.groupby('Currency').mean()
    
    # Calculate annualized average return and volatility for excess currency returns (using 252 days)
    curr_returns = R_c.mean() * 252 * 100
    curr_vols = R_c.std() * np.sqrt(252) * 100
    
    currency_stats = pd.DataFrame({'Currency Return': curr_returns, 'Currency Volatility': curr_vols})
    
    # Combine stock and currency statistics into one dataframe
    final_stats = stock_stats_by_currency.join(currency_stats, how='inner')
    
    # Define the desired order of currencies
    currency_order = ['AUD', 'CAD', 'DKK', 'EUR', 'HKD', 'JPY', 'NZD', 'NOK', 'SGD', 'ZAR', 'SEK', 'CHF', 'GBP']
    
    # Reorder final_stats according to the desired currency order
    final_stats = final_stats.reindex(currency_order)
    
    # Print out the results matching the table format
    print(f"{'Currency':<15} | {'Stock Return (%)':<18} | {'Stock Volatility (%)':<22} | {'Currency Return (%)':<22} | {'Currency Volatility (%)'}")
    print("-" * 110)
    print(final_stats.to_string(float_format="{:.2f}".format, header=False))

if __name__ == "__main__":
    main()
