"""

Description: Base script for running the UniversalBacktester with an Equal Weight Optimizer. 
Loads data, sets up the backtester, and plots cumulative returns.

Author: Anej Rozman
Last edited: -

"""

import os
import sys
import matplotlib.pyplot as plt

# Add the src directory to the path so we can import from it
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.backtest.UniversalBacktester import UniversalBacktester
from src.backtest.DataHandler import DataHandler
from src.optimizers.EqualWeightOptimizer import EqualWeightOptimizer

def main():

    processed_data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
    raw_data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
    
    # File paths
    universe_file = os.path.join(raw_data_dir, 'currency_alocation_10_202603081747.csv') # Top 10 assets per currency universe
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
    
    # Set up data bundle
    data_bundle = {
        'R_fh': R_fh,
        'R_c': R_c,
    }
    
    # Backtester configuration
    backtester = UniversalBacktester(
        optimizer_class=EqualWeightOptimizer,
        window_size=2500,  
        val_size=250,      
        step_size=63       
    )
    
    # Run backtest
    oos_returns = backtester.run(data_bundle)
    
    # Cumulative returns
    cumulative_returns = (1 + oos_returns).cumprod()
    
    # Plot
    plt.figure(figsize=(10, 6))
    cumulative_returns.plot(label='Equal Weight Fully Hedged')
    plt.title('Cumulative Returns - Equally Weighted Portfolio')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(os.path.dirname(__file__), '..', 'plots', 'EqualWeightOptimizer_cum_returns_10univ_3m.png'), dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()