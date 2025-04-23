import numpy as np
from scipy.special import rel_entr
from ._blossum import get_substitution_matrix
from typing import Literal, Union

def batch_pairwise_distance(cols1: np.ndarray,
                           cols2: np.ndarray,
                           metric: Literal['jensen_shannon', 'kl_divergence', 'hybrid'] = 'hybrid',
                           substitution_matrix: Union[np.ndarray, str] = None,
                           alpha: float = 0.5,
                           epsilon: float = 1e-10) -> np.ndarray:
    """
    Calculate pairwise distance matrix between pairs of columns in a
    vectorized operation.

    Parameters
    ----------
    p : np.ndarray 
        The first probability distribution.
    
    q: np.ndarray
        The second probability distributin. 

    substitution_matrix : np.ndarray or str
        The replacement matrix.
    
    alpha : float
        The scalar to to the standard distance metric.
    
    metric : str, default=`jensen_shannon`
        The probability distance metric to use.
    
    epsilon : float
        Small value to stabilize numeric operations.
        
    Returns
    -------
    score : float
        
    Returns
    -------
    hybrid_dist : np.ndarray
        Hybrid distance matrix with shape (L,L). 
    """
    # Preprocess: clip and normalize each row.
    cols1_safe = np.clip(cols1, epsilon, 1.0)
    cols2_safe = np.clip(cols2, epsilon, 1.0)

    cols1_norm = cols1_safe / np.sum(cols1_safe, axis=1, keepdims=True)  # shape (n, d)
    cols2_norm = cols2_safe / np.sum(cols2_safe, axis=1, keepdims=True)  # shape (m, d)
    
    if metric == 'hybrid' and substitution_matrix is not None:
    
        if isinstance(substitution_matrix, str):
            matrix = get_substitution_matrix(substitution_matrix)
        else:
            matrix = substitution_matrix

        p_exp = cols1_norm[:, None, :]  # shape (n, 1, d)
        q_exp = cols2_norm[None, :, :]  # shape (1, m, d)
        m_val = 0.5 * (p_exp + q_exp)    # shape (n, m, d)
        with np.errstate(divide='ignore', invalid='ignore'):

            kl1 = np.sum(p_exp * np.log(p_exp / m_val), axis=2)  # shape (n, m)
            kl2 = np.sum(q_exp * np.log(q_exp / m_val), axis=2)  # shape (n, m)
        js = np.sqrt(0.5 * (kl1 + kl2))
        js = np.nan_to_num(js)
        
        # Einsum to distance matrix.
        sim = np.einsum('id,df,jf->ij', cols1_norm, matrix, cols2_norm)
        max_val = np.max(matrix)
        blosum_dist = 1.0 - sim / max_val
        
        hybrid_dist = alpha * js + (1.0 - alpha) * blosum_dist
        return hybrid_dist

    elif metric == 'jensen_shannon':
        p_exp = cols1_norm[:, None, :]  # (n, 1, d)
        q_exp = cols2_norm[None, :, :]   # (1, m, d)
        m_val = 0.5 * (p_exp + q_exp)
        with np.errstate(divide='ignore', invalid='ignore'):
            kl1 = np.sum(p_exp * np.log(p_exp / m_val), axis=2)  # (n, m)
            kl2 = np.sum(q_exp * np.log(q_exp / m_val), axis=2)  # (n, m)
        js = np.sqrt(0.5 * (kl1 + kl2))
        return np.nan_to_num(js)
    
    elif metric == 'kl_divergence':

        p_exp = cols1_norm[:, None, :]  # (n, 1, d)
        q_exp = cols2_norm[None, :, :]   # (1, m, d)
        divergence = np.sum(rel_entr(p_exp, q_exp), axis=2)  # (n, m)
        return divergence
    else:
        raise ValueError(f"Unknown distance metric: {metric}")


def batch_column_distance(cols1: np.ndarray,
                          cols2: np.ndarray,
                          metric: Literal['jensen_shannon', 'kl_divergence', 'hybrid']='hybrid',
                          substitution_matrix: Union[np.ndarray, str]=None,
                          alpha: float = 0.5,
                          epsilon: float=1e-10) -> np.ndarray:
    """
    Calculate distance vector between multiple pairs of columns in a
    vectorized operation.

    Parameters
    ----------
    p : np.ndarray 
        The first probability distribution.
    
    q: np.ndarray
        The second probability distributin. 

    substitution_matrix : np.ndarray or str
        The replacement matrix.
    
    alpha : float
        The scalar to to the standard distance metric.
    
    metric : str, default=`jensen_shannon`
        The probability distance metric to use.
    
    epsilon : float
        Small value to stabilize numeric operations.
        
    Returns
    -------
    score : float
        
    Returns
    -------
    hybrid_dist : np.ndarray
        Hybrid distance vector with shape (L,). 
    """

    # Clip and normalize each row.
    cols1_safe = np.clip(cols1, epsilon, 1.0)
    cols2_safe = np.clip(cols2, epsilon, 1.0)
    cols1_norm = cols1_safe / np.sum(cols1_safe, axis=1, keepdims=True)
    cols2_norm = cols2_safe / np.sum(cols2_safe, axis=1, keepdims=True)
    
    if metric == 'hybrid' and substitution_matrix is not None:
        # Retrieve substitution matrix if provided as a string.
        if isinstance(substitution_matrix, str):
            matrix = get_substitution_matrix(substitution_matrix)
        else:
            matrix = substitution_matrix

        # For each pair of rows in cols1_norm and cols2_norm, compute:
        #   m = (p + q) / 2
        #   KL divergence parts: kl1 = sum(p * log(p / m)), kl2 = sum(q * log(q / m))
        #   js = sqrt(0.5*(kl1 + kl2))
        
        m = 0.5 * (cols1_norm + cols2_norm)  # (n, d)
        with np.errstate(divide='ignore', invalid='ignore'):
            kl1 = np.sum(cols1_norm * np.log(cols1_norm / m), axis=1)
            kl2 = np.sum(cols2_norm * np.log(cols2_norm / m), axis=1)
        js = np.sqrt(0.5 * (kl1 + kl2))
        js = np.nan_to_num(js)

        # Compute the substitution-based similarity using einsum
        # For each row i, compute: sim[i] = p_i^T * matrix * q_i.
        sim = np.einsum('ij,jk,ik->i', cols1_norm, matrix, cols2_norm)
        # Convert similarity to a distance.
        max_val = np.max(matrix)
        blosum_dist = 1.0 - sim / max_val

        hybrid_dist = alpha * js + (1.0 - alpha) * blosum_dist
        return hybrid_dist

    elif metric == 'jensen_shannon':
        # Vectorized JS
        m = 0.5 * (cols1_norm + cols2_norm)
        with np.errstate(divide='ignore', invalid='ignore'):
            kl1 = np.sum(cols1_norm * np.log(cols1_norm / m), axis=1)
            kl2 = np.sum(cols2_norm * np.log(cols2_norm / m), axis=1)
        js = np.sqrt(0.5 * (kl1 + kl2))
        return np.nan_to_num(js)
    
    elif metric == 'kl_divergence':

        return np.sum(rel_entr(cols1_norm, cols2_norm), axis=1)
    else:
        raise ValueError(f"Unknown distance metric: {metric}")

