"""

Description: Abstract Base Class for all portfolio optimizers in the framework.
Standard structure for any optimizer interacting with the UniversalBacktester. 

Author: Anej Rozman
Last edited: 2026-03-18

"""


from abc import ABC, abstractmethod

class BaseOptimizer(ABC):
    
    def __init__(self, hyperparams=None):
        """
        Args:
            hyperparams (dict): Dictionary of penalty parameters (e.g., lambda_1, lambda_2).
                                Defaults to an empty dict for unregularized models.
        """
        self.hyperparams = hyperparams if hyperparams is not None else {}
        
    @abstractmethod
    def fit(self, train_bundle):
        """
        Extracts and stores necessary components from the training data.
        
        Args:
            train_bundle (dict): Contains required data for optimization. 
                                 e.g., {'R_fh': pd.DataFrame, 'R_c': pd.DataFrame, 'C': np.ndarray}
        """
        pass
        
    @abstractmethod
    def optimize(self):
        """
        Executes the optimization to find the optimal weights.
        
        Returns:
            np.ndarray or dict: The optimal portfolio weights. 
                                (e.g., a single vector for assets, or a dict {'x': x_opt, 'psi': psi_opt})
        """
        pass
        
    @abstractmethod
    def score(self, weights, val_bundle):
        """
        Evaluates the optimizer's performance
        on unseen validation data. Used by the backtester for hyperparameter tuning.
        
        Args:
            weights: The output from the optimize() method.
            val_bundle (dict): Unseen validation data.
            
        Returns:
            float: The risk score.
        """
        pass
