import numpy as np
import pandas as pd
import cvxpy as cp
import warnings

from .BaseOptimizer import BaseOptimizer

class RegularizedExpectedShortfallAssetZeroOptimizer(BaseOptimizer):
    """
    Asset-only Regularized Expected Shortfall Optimizer (ASSETS-ZERO strategy).

    Optimizes portfolio weights to minimize Expected Shortfall at a given confidence
    level, plus L1 and L2 regularization penalties, subject to a leverage limit
    (L1 norm bound) and a full investment constraint. Uses unhedged asset returns
    (R_u): the investor bears full currency risk and holds no forward positions.
    """

    def __init__(self, hyperparams=None):
        super().__init__(hyperparams)
        self.alpha = self.hyperparams.get('alpha', 0.80)
        self.lambda_l1 = self.hyperparams.get('lambda_l1', 0.0)
        self.lambda_l2 = self.hyperparams.get('lambda_l2', 0.0)
        self.leverage_limit = self.hyperparams.get('leverage_limit', 2.0)
        self.solver = self.hyperparams.get('solver', cp.GUROBI)

        self.Y = None
        self.N = 0
        self.q = 0

    def fit(self, train_bundle):
        """
        Extracts and stores historical unhedged returns for optimization.
        """
        if 'R_u' not in train_bundle:
            raise ValueError("train_bundle must contain 'R_u'")

        R_u = train_bundle['R_u']
        if R_u.empty:
            raise ValueError("R_u in train_bundle is empty.")

        self.Y = R_u.values
        self.q, self.N = self.Y.shape

    def optimize(self):
        """
        Formulates and solves the regularized Expected Shortfall minimization problem.
        Returns a 1D NumPy array of asset weights.
        """
        if self.Y is None or self.N == 0:
            raise RuntimeError("fit() must be called before optimize().")

        x = cp.Variable(self.N)
        u = cp.Variable(self.q)
        eta = cp.Variable()

        # Reformulate L1 norm for constraints and objective
        l1_penalty = cp.norm(x, 1)
        l2_penalty_sq = cp.sum_squares(x)

        tail_prob = 1.0 - self.alpha
        if tail_prob <= 0 or tail_prob >= 1:
            raise ValueError("alpha must be in (0, 1)")

        # Objective: ES + L1 + L2
        es_term = eta + (1.0 / (self.q * tail_prob)) * cp.sum(u)
        objective = cp.Minimize(es_term + self.lambda_l1 * l1_penalty + self.lambda_l2 * l2_penalty_sq)

        constraints = [
            u >= 0,
            -self.Y @ x - eta <= u,
            cp.sum(x) == 1,
            l1_penalty <= self.leverage_limit
        ]

        problem = cp.Problem(objective, constraints)

        try:
            if self.solver == cp.GUROBI:
                # Method=2 forces Barrier, Crossover=0 massively speeds up Expected Shortfall
                problem.solve(
                    solver=self.solver,
                    Method=2,
                    verbose=True,
                    Crossover=0,
                    BarHomogeneous=1  # Better numerical stability for financial decimals
                )
            else:
                problem.solve(solver=self.solver)
        except cp.error.SolverError as e:
            raise RuntimeError(f"Solver error: {str(e)}")

        if problem.status not in ["optimal", "optimal_inaccurate"]:
            raise RuntimeError(f"Optimization failed with status: {problem.status}")

        weights = x.value

        if weights is None:
            raise RuntimeError("Solver finished but weights are None.")

        # Ensure exact sum to 1 to correct minor numerical imprecision
        weights = weights / np.sum(weights)

        return weights.flatten()

    def score(self, weights, val_bundle):
        """
        Evaluates the validation Expected Shortfall of the candidate weights.
        Lower score is better.
        """
        if 'R_u' not in val_bundle:
            raise ValueError("val_bundle must contain 'R_u'")

        R_u_val = val_bundle['R_u']
        if R_u_val.empty:
            return float('inf')

        Y_val = R_u_val.values
        port_rets = Y_val @ weights

        tail_prob = 1.0 - self.alpha
        threshold = np.quantile(port_rets, tail_prob)

        tail_returns = port_rets[port_rets <= threshold]

        if len(tail_returns) == 0:
            # Fallback if numerical issues lead to empty slice
            return -np.mean(port_rets)

        return -np.mean(tail_returns)
