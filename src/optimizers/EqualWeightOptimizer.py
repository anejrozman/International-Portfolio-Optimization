from .BaseOptimizer import BaseOptimizer
import numpy as np

class EqualWeightOptimizer(BaseOptimizer):
    """
    1/N benchmark portfolio.
    Allocates wealth equally across all available assets and 
    assumes zero active currency overlay (fully hedged baseline).
    """
    
    def __init__(self, hyperparams=None):
        super().__init__(hyperparams)
        self.num_assets = 0
        self.valid_mask = None
        
    def fit(self, train_bundle):
        """
        The 1/N portfolio doesn't need historical returns to train.
        It only needs to know how many assets (N) exist in the universe.
        """
        R_fh = train_bundle.get('R_fh')
        if R_fh is None:
            raise ValueError("train_bundle must contain 'R_fh' (asset returns).")
            
        self.valid_mask = ~R_fh.iloc[-1].isna().values
        self.num_assets = R_fh.shape[1]
        
    def optimize(self):
        """
        Returns a vector where every asset gets exactly 1/N of the capital.
        """
        if self.num_assets == 0:
            raise ValueError("Optimizer must be fitted before calling optimize().")
            
        weights = np.zeros(self.num_assets)
        valid_count = self.valid_mask.sum()
        print(f"DEBUG: Allocating 1/N weight to {valid_count} assets.")

        if valid_count > 0:
            weights[self.valid_mask] = 1.0 / valid_count
        
        return weights
        
    def score(self, weights, val_bundle):
        """
        Because 1/N has no hyperparameters.
        """
        return 0.0
