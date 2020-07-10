import pandas as pd
import numpy as np

def get_return(weights: pd.Series, mean_rets: pd.Series) -> float:
    """
    Total Portfolio Return given the respective weights
    """
    return (weights * mean_rets).sum() * 252


def get_std(weights: pd.Series, cov_matrix: pd.DataFrame) -> float:
    """
    Portfolio Variance given the respective covariance Matrix
    """
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)


def get_sharpe_ratio(pf_return: float, pf_std: float) -> float:
    """
    Sharpe ratio assuming risk-free rate is zero
    """
    return pf_return / pf_std