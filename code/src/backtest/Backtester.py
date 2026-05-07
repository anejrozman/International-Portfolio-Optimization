"""

Description: Modular Backtester implementation based on "Regularized Multi-Currency Expected Shortfall Portfolios" by Ulrych and Lucescu (2026) and 
"Sparse and stable international portfolio optimization and currency risk management" by Ulrych and Burkhardt (2023). 

Author: Anej Rozman
Last edited: -

"""

import pandas as pd
import os
import sys
import math
from itertools import product
from typing import Type, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.optimizers.BaseOptimizer import BaseOptimizer

class Backtester:

    def __init__(self, 
                 optimizer: Type[BaseOptimizer], 
                 param_grid: Optional[dict] = None,         # Grid of hyperparameters to search over during model selection. Depends on optimizer type. 
                 window_years: int = 10,                    # Total length of rolling lookcback window used for model estimaton, validation and selection at each rebalancing date.
                 val_years: int = 1,                        # Length of validation window used for model selection at each rebalancing date. Must be less than window_years.
                 rebalancing_frequency: str = "quarterly"   # Options: daily, weekly, monthly, quarterly, semi-annually, annually
                 ):

            self.optimizer = optimizer
            self.param_grid = param_grid if param_grid else {}
            self.window_years = window_years
            self.val_years = val_years 
            self.rebalancing_frequency = rebalancing_frequency
    
    def _get_rebalance_dates(self, dates: pd.DatetimeIndex) -> list[pd.Timestamp]:
        """
        Build rebalancing dates from a trading-day index.

        For non-daily frequencies, the rebalance date is the last available
        trading day in each period.
        """
        if dates is None or len(dates) == 0:
            return []

        trading_dates = pd.DatetimeIndex(dates).sort_values().unique()
        freq = self.rebalancing_frequency.strip().lower()

        if freq == "daily":
            return list(trading_dates)
        if freq == "weekly":
            period_key = trading_dates.to_period("W-FRI")
        elif freq == "monthly":
            period_key = trading_dates.to_period("M")
        elif freq == "quarterly":
            period_key = trading_dates.to_period("Q")
        elif freq == "semi-annually":
            period_key = pd.Index(
                [(d.year, 1 if d.month <= 6 else 2) for d in trading_dates],
                dtype="object"
            )
        elif freq == "annually":
            period_key = trading_dates.to_period("Y")
        else:
            raise ValueError(
                "Unsupported rebalancing_frequency. "
                "Use one of: daily, weekly, monthly, quarterly, semi-annually, annually."
            )

        helper = pd.DataFrame({"date": trading_dates, "period": period_key})
        rebalance_dates = (
            helper.groupby("period", sort=False)["date"]
            .max()
            .sort_values()
            .tolist()
        )
        return rebalance_dates
    
    def _get_estimation_and_validation_splits(self, 
                                              dates: pd.DatetimeIndex,              # Full index of trading dates.
                                              rebalancing_date: pd.Timestamp,       # Current rebalancing date for which we want to compute the splits.
                                              next_rebalancing_date: pd.Timestamp   # Next rebalancing date, needed to compute the holding period end boundary with execution lag.
                                              ) -> dict[str, pd.Timestamp]:
        """
        Compute period boundaries for one rebalancing cycle.

        Returns a dictionary with:
        - train_start, train_end
        - validation_start, validation_end
        - full_estimation_start, full_estimation_end
        - holding_start, holding_end
        """

        trading_dates = pd.DatetimeIndex(dates).sort_values().unique()
        t = pd.Timestamp(rebalancing_date)
        t_next = pd.Timestamp(next_rebalancing_date)

        if t not in trading_dates:
            raise ValueError("'rebalancing_date' must be present in 'dates'.")
        if t_next not in trading_dates:
            raise ValueError("'next_rebalancing_date' must be present in 'dates'.")
        if t_next <= t:
            raise ValueError("'next_rebalancing_date' must be strictly after 'rebalancing_date'.")

        # Calendar anchors based on year offsets.
        estimation_anchor = t - pd.DateOffset(years=self.window_years)
        validation_anchor = t - pd.DateOffset(years=self.val_years)

        def first_on_or_after(ts: pd.Timestamp) -> pd.Timestamp:
            idx = trading_dates.searchsorted(ts, side="left")
            if idx >= len(trading_dates):
                raise ValueError("Cannot align start boundary: anchor is after available dates.")
            return pd.Timestamp(trading_dates[idx])

        def last_on_or_before(ts: pd.Timestamp) -> pd.Timestamp:
            idx = trading_dates.searchsorted(ts, side="right") - 1
            if idx < 0:
                raise ValueError("Cannot align end boundary: anchor is before available dates.")
            return pd.Timestamp(trading_dates[idx])

        # Estimation/validation boundaries.
        train_start = first_on_or_after(estimation_anchor)
        train_end = last_on_or_before(validation_anchor)
        train_end_idx = trading_dates.get_loc(train_end)
        validation_start = pd.Timestamp(trading_dates[train_end_idx + 1])
        validation_end = t

        if train_start > train_end:
            raise ValueError("Invalid training window: start is after end.")
        if validation_start > validation_end:
            raise ValueError("Invalid validation window: start is after end.")

        # Full estimation window for final fit after model selection.
        full_estimation_start = train_start
        full_estimation_end = t

        # Execution lag:
        # - Weights computed at close of t
        # - Orders executed at close of t+1
        # - OOS returns start at t+2 and end at t_next+1
        t_idx = trading_dates.get_loc(t)
        t_next_idx = trading_dates.get_loc(t_next)

        if isinstance(t_idx, slice) or isinstance(t_next_idx, slice):
            raise ValueError("Duplicate rebalancing dates detected in trading index.")

        holding_start_idx = int(t_idx) + 2
        holding_end_idx = int(t_next_idx) + 1

        if holding_start_idx >= len(trading_dates):
            raise ValueError("Cannot compute holding start (t+2): insufficient future dates.")
        if holding_end_idx >= len(trading_dates):
            raise ValueError("Cannot compute holding end (t_next+1): insufficient future dates.")
        if holding_start_idx > holding_end_idx:
            raise ValueError("Invalid holding window: start is after end.")

        holding_start = pd.Timestamp(trading_dates[holding_start_idx])
        holding_end = pd.Timestamp(trading_dates[holding_end_idx])

        return {
            "train_start": train_start,
            "train_end": train_end,
            "validation_start": validation_start,
            "validation_end": validation_end,
            "full_estimation_start": full_estimation_start,
            "full_estimation_end": full_estimation_end,
            "holding_start": holding_start,
            "holding_end": holding_end,
        }
    
    def _filter_universe(self, data_slice: pd.DataFrame) -> list[str]:
        """
        Takes a sliced dataframe of returns.
        Returns a list of column names that have no missing values 
        in this specific time window.
        """
        valid_assets = data_slice.columns[data_slice.notna().all()].tolist()
        if len(valid_assets) == 0:
            raise ValueError("No assets have a complete price history for this window.")
        return valid_assets

    
    def _generate_grid(self) -> list:
        """
        Generates a list of hyperparameter combinations from the provided parameter grid.
        """
        if not self.param_grid: 
             return [{}]
        keys, values = zip(*self.param_grid.items())
        return [dict(zip(keys, v)) for v in product(*values)]

    def _compute_drifted_weights(self, initial_weights: pd.Series, returns_slice: pd.DataFrame) -> pd.DataFrame:
        """
        Given the target weights at the start of the holding period, 
        compound them daily using the asset returns to find the drifted 
        weights at the end of the period.
            
        Returns:
            pd.DataFrame: A dataframe of the same shape as returns_slice containing the daily drifted weights.
        """
        
        shifted_returns = returns_slice.shift(1).fillna(0)
        cumulative_growth = (1 + shifted_returns).cumprod()
        unnormalized_weights = cumulative_growth * initial_weights
        drifted_weights = unnormalized_weights.div(unnormalized_weights.sum(axis=1), axis=0)

        return drifted_weights
    
    def _compute_asset_transaction_costs(self, 
                                         drifted_weights: pd.Series, 
                                         new_weights: pd.Series, 
                                         asset_cost_bps: float = 10.0   # Default to 10 bps per unit traded, assumption in Appendix C of the thesis.
                                         ) -> float:
        """
        Calculate the transaction costs incurred from rebalancing individual asset holdings.
        """
        chi = asset_cost_bps / 10000.0
        tau = math.log((1 + chi) / (1 - chi))
        trade_volume = (new_weights - drifted_weights).abs()
        
        return float((tau * trade_volume).sum())

    def _compute_currency_transaction_costs(self,
                                            drifted_forward_weights: pd.Series,
                                            new_forward_weights: pd.Series,
                                            drifted_currency_weights: pd.Series,
                                            new_currency_weights: pd.Series,
                                            full_forward_turnover: bool,
                                            base_currency: str = "USD",
                                            fx_cost_bps: float = 1.0        # Default to 1 bp per unit traded, assumption in Appendix C of the thesis.
                                            ) -> float:
        """
        Calculate the transaction costs incurred from currency spot conversions and forward hedges.
        """

        chi = fx_cost_bps / 10000.0
        tau = math.log((1 + chi) / (1 - chi))
        
        # Helper to zero out the base currency costs
        def _exclude_base(s: pd.Series) -> pd.Series:
            s_out = s.copy()
            if base_currency in s_out.index:
                s_out[base_currency] = 0.0
            return s_out

        new_fwds = _exclude_base(new_forward_weights)
        drifted_fwds = _exclude_base(drifted_forward_weights)
        new_spot = _exclude_base(new_currency_weights)
        drifted_spot = _exclude_base(drifted_currency_weights)
        
        if full_forward_turnover:
            # Old contracts expire, new contracts are fully traded
            fwd_volume = new_fwds.abs()
        else:
            # Contracts are kept open, only incremental adjustments are traded
            fwd_volume = (new_fwds - drifted_fwds).abs()
            
        fwd_costs = (tau * fwd_volume).sum()
        
        # Exchanges required to fund purchases of assets denominated in different currencies
        spot_volume = (new_spot - drifted_spot).abs()
        spot_costs = (tau * spot_volume).sum()
        
        return float(fwd_costs + spot_costs)

    def run(self, data_bundle: dict) -> pd.DataFrame:
        return pd.DataFrame()
    