import numpy as np
from scipy.special import rel_entr
from ._blossum import get_reordered_matrix
from typing import Literal, Union, List

def batch_pairwise_distance(cols1: np.ndarray,
                           cols2: np.ndarray,
                           alphabet: List[str],
                           metric: Literal['jensen_shannon', 'kl_divergence', 'hybrid'] = 'hybrid',
                           matrix_name: Literal['BLOSUM62'] = 'BLOSUM62',
                           alpha: float = 0.05,
                           epsilon: float = 1e-10) -> np.ndarray:
    """
    Calculate pairwise distance matrix between pairs of columns in a
    vectorized operation.

    Parameters
    ----------
    cols1 : np.ndarray
        The first set of probability distributions (shape: n, alphabet_size).
    cols2: np.ndarray
        The second set of probability distributions (shape: m, alphabet_size).
    alphabet : List[str]
        The alphabet corresponding to the columns of the probability distributions.
    metric : str, default=`hybrid`
        The probability distance metric to use.
    matrix_name : str, default='BLOSUM62'
        The name of the substitution matrix to use for the hybrid metric.
    alpha : float
        The scalar to weight the Jensen-Shannon vs. substitution distance.
    epsilon : float
        Small value to stabilize numeric operations.
        
    Returns
    -------
    np.ndarray
        Distance matrix with shape (n, m). 
    """
    # Preprocess: clip and normalize each row.
    cols1_safe = np.clip(cols1, epsilon, 1.0)
    cols2_safe = np.clip(cols2, epsilon, 1.0)

    cols1_norm = cols1_safe / np.sum(cols1_safe, axis=1, keepdims=True)
    cols2_norm = cols2_safe / np.sum(cols2_safe, axis=1, keepdims=True)
    
    if metric == 'hybrid':
        # Get the substitution matrix, correctly reordered for the given alphabet.
        matrix = get_reordered_matrix(target_alphabet=alphabet, matrix_name=matrix_name)

        p_exp = cols1_norm[:, None, :]  # shape (n, 1, d)
        q_exp = cols2_norm[None, :, :]  # shape (1, m, d)
        m_val = 0.5 * (p_exp + q_exp)    # shape (n, m, d)
        with np.errstate(divide='ignore', invalid='ignore'):
            kl1 = np.sum(p_exp * np.log(p_exp / m_val), axis=2)  # shape (n, m)
            kl2 = np.sum(q_exp * np.log(q_exp / m_val), axis=2)  # shape (n, m)
        js = np.sqrt(0.5 * (kl1 + kl2))
        js = np.nan_to_num(js)
        
        # Einsum to calculate substitution-based similarity matrix.
        sim = np.einsum('id,df,jf->ij', cols1_norm, matrix, cols2_norm)
        max_val = np.max(matrix)
        blosum_dist = 1.0 - sim / max_val
        
        hybrid_dist = alpha * js + (1.0 - alpha) * blosum_dist
        return hybrid_dist

    elif metric == 'jensen_shannon':
        p_exp = cols1_norm[:, None, :]
        q_exp = cols2_norm[None, :, :]
        m_val = 0.5 * (p_exp + q_exp)
        with np.errstate(divide='ignore', invalid='ignore'):
            kl1 = np.sum(p_exp * np.log(p_exp / m_val), axis=2)
            kl2 = np.sum(q_exp * np.log(q_exp / m_val), axis=2)
        js = np.sqrt(0.5 * (kl1 + kl2))
        return np.nan_to_num(js)
    
    elif metric == 'kl_divergence':
        p_exp = cols1_norm[:, None, :]
        q_exp = cols2_norm[None, :, :]
        divergence = np.sum(rel_entr(p_exp, q_exp), axis=2)
        return divergence
    else:
        raise ValueError(f"Unknown distance metric: {metric}")


def batch_column_distance(cols1: np.ndarray,
                          cols2: np.ndarray,
                          alphabet: List[str],
                          metric: Literal['jensen_shannon', 'kl_divergence', 'hybrid']='hybrid',
                          matrix_name: Literal['BLOSUM62'] = 'BLOSUM62',
                          alpha: float = 0.5,
                          epsilon: float=1e-10) -> np.ndarray:
    """
    Calculate distance vector between corresponding pairs of columns in a
    vectorized operation.

    Parameters
    ----------
    cols1 : np.ndarray 
        The first set of probability distributions (shape: n, alphabet_size).
    cols2: np.ndarray
        The second set of probability distributions (shape: n, alphabet_size).
    alphabet : List[str]
        The alphabet corresponding to the columns of the probability distributions.
    metric : str, default=`hybrid`
        The probability distance metric to use.
    matrix_name : str, default='BLOSUM62'
        The name of the substitution matrix to use for the hybrid metric.
    alpha : float
        The scalar to weight the Jensen-Shannon vs. substitution distance.
    epsilon : float
        Small value to stabilize numeric operations.
        
    Returns
    -------
    np.ndarray
        Distance vector with shape (n,). 
    """
    # Clip and normalize each row.
    cols1_safe = np.clip(cols1, epsilon, 1.0)
    cols2_safe = np.clip(cols2, epsilon, 1.0)
    cols1_norm = cols1_safe / np.sum(cols1_safe, axis=1, keepdims=True)
    cols2_norm = cols2_safe / np.sum(cols2_safe, axis=1, keepdims=True)
    
    if metric == 'hybrid':
        # Get the substitution matrix, correctly reordered for the given alphabet.
        matrix = get_reordered_matrix(target_alphabet=alphabet, matrix_name=matrix_name)

        m = 0.5 * (cols1_norm + cols2_norm)
        with np.errstate(divide='ignore', invalid='ignore'):
            kl1 = np.sum(cols1_norm * np.log(cols1_norm / m), axis=1)
            kl2 = np.sum(cols2_norm * np.log(cols2_norm / m), axis=1)
        js = np.sqrt(0.5 * (kl1 + kl2))
        js = np.nan_to_num(js)

        # Compute the substitution-based similarity using einsum
        sim = np.einsum('ij,jk,ik->i', cols1_norm, matrix, cols2_norm)
        max_val = np.max(matrix)
        blosum_dist = 1.0 - sim / max_val

        hybrid_dist = alpha * js + (1.0 - alpha) * blosum_dist
        return hybrid_dist

    elif metric == 'jensen_shannon':
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
