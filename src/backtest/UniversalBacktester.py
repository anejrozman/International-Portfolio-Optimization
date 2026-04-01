"""

Description: Modular UniversalBacktester implementation based on "Regularized Multi-Currency Expected Shortfall Portfolios" by Ulrych and Lucescu (2026) and 
"Sparse and stable international portfolio optimization and currency risk management" by Ulrych and Burkhardt (2023). 

Author: Anej Rozman
Last edited: -

"""

import numpy as np
import pandas as pd
from itertools import product

class UniversalBacktester:
    def __init__(self, optimizer_class, param_grid=None, window_size=2500, val_size=250, step_size=63):
        self.optimizer_class = optimizer_class
        self.param_grid = param_grid if param_grid else {}
        self.window_size = window_size
        self.val_size = val_size
        self.step_size = step_size
        
    def _generate_grid(self):
        if not self.param_grid: return [{}]
        keys, values = zip(*self.param_grid.items())
        return [dict(zip(keys, v)) for v in product(*values)]

    def _slice_data(self, bundle, start, end):
        # Slices Pandas DataFrames by integer position while keeping numpy arrays intact
        return {k: (v.iloc[start:end] if isinstance(v, pd.DataFrame) else v) 
                for k, v in bundle.items()}

    def run(self, data_bundle):
        R_fh = data_bundle['R_fh']
        T = len(R_fh)
        
        oos_dates = []
        oos_returns = []

        drifted_weights_a = None

        for start_idx in range(0, T - self.window_size, self.step_size):
            end_idx = start_idx + self.window_size
            
            # Step 1: Train/Validation Split
            val_start = end_idx - self.val_size
            train_bundle = self._slice_data(data_bundle, start_idx, val_start)
            val_bundle = self._slice_data(data_bundle, val_start, end_idx)
            
            # Step 2: Hyperparameter Tuning
            best_params, best_score = None, float('inf')
            for params in self._generate_grid():
                opt = self.optimizer_class(hyperparams=params)
                opt.fit(train_bundle)
                weights = opt.optimize()
                score = opt.score(weights, val_bundle)
                
                if score < best_score:
                    best_score = score
                    best_params = params
                    
            # Step 3: Final Fit
            full_train_bundle = self._slice_data(data_bundle, start_idx, end_idx)
            final_opt = self.optimizer_class(hyperparams=best_params)
            final_opt.fit(full_train_bundle)
            
            # optimize() should return a dictionary if doing currency overlay/joint
            # e.g., {'assets': weights_a, 'currencies': weights_c}. 
            # For 1/N, it can just return asset weights.
            weights_out = final_opt.optimize()
            
            if isinstance(weights_out, dict):
                w_a = weights_out.get('assets')
                w_c = weights_out.get('currencies', np.zeros(data_bundle['R_c'].shape[1]))
            else:
                w_a = weights_out
                w_c = np.zeros(data_bundle['R_c'].shape[1])
            
                        # ---------------------------------------------------------
            # Step 4: Transaction Costs
            # ---------------------------------------------------------
            if drifted_weights_a is None:
                drifted_weights_a = np.zeros_like(w_a)
                
            delta_a = np.sum(np.abs(w_a - drifted_weights_a))
            
            # Note: According to Appendix B, delta_c should ideally also include the turnover 
            # of the implied currency weights of the assets, not just the forwards. 
            # If you have an asset-to-currency mapping, add it here.
            delta_c = np.sum(np.abs(w_c)) 

            cost_a = delta_a * 0.0010  # 10 bps
            cost_c = delta_c * 0.0001  # 1 bp
            total_cost = cost_a + cost_c
            
            # ---------------------------------------------------------
            # Step 5: True Buy-and-Hold Out-of-Sample Projection
            # ---------------------------------------------------------
            oos_bundle = self._slice_data(data_bundle, end_idx, end_idx + self.step_size)
            oos_asset_returns = np.nan_to_num(oos_bundle['R_fh'].values, nan=0.0)
            oos_currency_returns = np.nan_to_num(oos_bundle['R_c'].values, nan=0.0)
            
            # 1. Calculate the cumulative wealth of the assets over the 63 days
            # Starts at the target weights w_a and compounds daily
            asset_wealth = np.cumprod(1 + oos_asset_returns, axis=0) * w_a
            
            # 2. Calculate the cumulative PnL from the zero-investment currency forwards
            curr_pnl = np.cumsum(oos_currency_returns @ w_c)
            
            # 3. Total portfolio wealth index (starts at 1.0)
            total_wealth = np.sum(asset_wealth, axis=1) + curr_pnl
            
            # 4. Derive the true daily portfolio returns from the wealth index
            daily_port_returns = np.zeros(len(total_wealth))
            if len(total_wealth) > 0:
                # First day return is Day 1 Wealth minus the starting capital (1.0)
                daily_port_returns[0] = total_wealth[0] - 1.0 
                
                if len(total_wealth) > 1:
                    # Day 2+ returns are the daily percentage change in total wealth
                    daily_port_returns[1:] = (total_wealth[1:] - total_wealth[:-1]) / total_wealth[:-1]
                
                # Deduct total transaction costs from the FIRST day's return
                daily_port_returns[0] -= total_cost
                
            oos_returns.extend(daily_port_returns)
            oos_dates.extend(oos_bundle['R_fh'].index.tolist())
            
            # ---------------------------------------------------------
            # Step 6: Extract Drifted Weights for the Next Loop
            # ---------------------------------------------------------
            if len(total_wealth) > 0:
                # The real weight of the assets at the end of the 63 days is their 
                # ending wealth divided by the total portfolio equity
                drifted_weights_a = asset_wealth[-1] / total_wealth[-1]
            else:
                drifted_weights_a = w_a
            
        return pd.Series(oos_returns, index=oos_dates)

