import numpy as np
from typing import Literal, List

# Amino acid order used in BLOSUM and PAM matrices
_aa_order = "ARNDCQEGHILKMFPSTWYV"

# BLOSUM62 matrix
# Source: https://www.ncbi.nlm.nih.gov/IEB/ToolBox/C_DOC/lxr/source/data/BLOSUM62
BLOSUM62 = np.array([
    [ 4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0],  # A
    [-1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3],  # R
    [-2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3],  # N
    [-2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3],  # D
    [ 0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1],  # C
    [-1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2],  # Q
    [-1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2],  # E
    [ 0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3],  # G
    [-2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3],  # H
    [-1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3],  # I
    [-1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1],  # L
    [-1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2],  # K
    [-1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1],  # M
    [-2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1],  # F
    [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2],  # P
    [ 1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2],  # S
    [ 0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0],  # T
    [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3],  # W
    [-2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1],  # Y
    [ 0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4],  # V
])

# BLOSUM50 matrix
# Source: https://www.ncbi.nlm.nih.gov/IEB/ToolBox/C_DOC/lxr/source/data/BLOSUM50
BLOSUM50 = np.array([
    [ 5, -2, -1, -2, -1, -1, -1,  0, -2, -1, -2, -1, -1, -3, -1,  1,  0, -3, -2,  0],  # A
    [-2,  7, -1, -2, -4,  1,  0, -3,  0, -4, -3,  3, -2, -3, -3, -1, -1, -3, -1, -3],  # R
    [-1, -1,  7,  2, -2,  0,  0,  0,  1, -3, -4,  0, -2, -4, -2,  1,  0, -4, -2, -3],  # N
    [-2, -2,  2,  8, -4,  0,  2, -1, -1, -4, -4, -1, -4, -5, -1,  0, -1, -5, -3, -4],  # D
    [-1, -4, -2, -4, 13, -3, -3, -3, -3, -2, -2, -3, -2, -2, -4, -1, -1, -5, -3, -1],  # C
    [-1,  1,  0,  0, -3,  7,  2, -2,  1, -3, -2,  2,  0, -4, -1,  0, -1, -1, -1, -3],  # Q
    [-1,  0,  0,  2, -3,  2,  6, -3,  0, -4, -3,  1, -2, -3, -1, -1, -1, -3, -2, -3],  # E
    [ 0, -3,  0, -1, -3, -2, -3,  8, -2, -4, -4, -2, -3, -4, -2,  0, -2, -3, -3, -4],  # G
    [-2,  0,  1, -1, -3,  1,  0, -2, 10, -4, -3,  0, -1, -1, -2, -1, -2, -3,  2, -4],  # H
    [-1, -4, -3, -4, -2, -3, -4, -4, -4,  5,  2, -3,  2,  0, -3, -3, -1, -3, -1,  4],  # I
    [-2, -3, -4, -4, -2, -2, -3, -4, -3,  2,  5, -3,  3,  1, -4, -3, -1, -2, -1,  1],  # L
    [-1,  3,  0, -1, -3,  2,  1, -2,  0, -3, -3,  6, -2, -4, -1,  0, -1, -3, -2, -3],  # K
    [-1, -2, -2, -4, -2,  0, -2, -3, -1,  2,  3, -2,  7,  0, -3, -2, -1, -1,  0,  1],  # M
    [-3, -3, -4, -5, -2, -4, -3, -4, -1,  0,  1, -4,  0,  8, -4, -3, -2,  1,  4, -1],  # F
    [-1, -3, -2, -1, -4, -1, -1, -2, -2, -3, -4, -1, -3, -4, 10, -1, -1, -4, -3, -3],  # P
    [ 1, -1,  1,  0, -1,  0, -1,  0, -1, -3, -3,  0, -2, -3, -1,  5,  2, -4, -2, -2],  # S
    [ 0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  2,  5, -3, -2,  0],  # T
    [-3, -3, -4, -5, -5, -1, -3, -3, -3, -3, -2, -3, -1,  1, -4, -4, -3, 15,  2, -3],  # W
    [-2, -1, -2, -3, -3, -1, -2, -3,  2, -1, -1, -2,  0,  4, -3, -2, -2,  2,  8, -1],  # Y
    [ 0, -3, -3, -4, -1, -3, -3, -4, -4,  4,  1, -3,  1, -1, -3, -2,  0, -3, -1,  5],  # V
])

# PAM250 matrix
# Source: https://www.ncbi.nlm.nih.gov/IEB/ToolBox/C_DOC/lxr/source/data/PAM250
PAM250 = np.array([
    [ 2, -2,  0,  0, -2,  0,  0,  1, -1, -1, -2, -1, -1, -3,  1,  1,  1, -6, -3,  0],  # A
    [-2,  6,  0, -1, -4,  1, -1, -3,  2, -2, -3,  3,  0, -4,  0,  0, -1,  2, -4, -2],  # R
    [ 0,  0,  2,  2, -4,  1,  1,  0,  2, -2, -3,  1, -2, -3,  0,  1,  0, -4, -2, -2],  # N
    [ 0, -1,  2,  4, -5,  2,  3,  1,  1, -2, -4,  0, -3, -6, -1,  0,  0, -7, -4, -2],  # D
    [-2, -4, -4, -5, 12, -5, -5, -3, -3, -2, -6, -5, -5, -4, -3,  0, -2, -8,  0, -2],  # C
    [ 0,  1,  1,  2, -5,  4,  2, -1,  3, -2, -2,  1, -1, -5,  0, -1, -1, -5, -4, -2],  # Q
    [ 0, -1,  1,  3, -5,  2,  4,  0,  1, -2, -3,  0, -2, -5, -1,  0,  0, -7, -4, -2],  # E
    [ 1, -3,  0,  1, -3, -1,  0,  5, -2, -3, -4, -2, -3, -5,  0,  1,  0, -7, -5, -1],  # G
    [-1,  2,  2,  1, -3,  3,  1, -2,  6, -2, -2,  0, -2, -2,  0, -1, -1, -3,  0, -2],  # H
    [-1, -2, -2, -2, -2, -2, -2, -3, -2,  5,  2, -2,  2,  1, -2, -1,  0, -5, -1,  4],  # I
    [-2, -3, -3, -4, -6, -2, -3, -4, -2,  2,  6, -3,  4,  2, -3, -3, -2, -2, -1,  2],  # L
    [-1,  3,  1,  0, -5,  1,  0, -2,  0, -2, -3,  5,  0, -5, -1,  0,  0, -3, -4, -2],  # K
    [-1,  0, -2, -3, -5, -1, -2, -3, -2,  2,  4,  0,  6,  0, -2, -2, -1, -4, -2,  2],  # M
    [-3, -4, -3, -6, -4, -5, -5, -5, -2,  1,  2, -5,  0,  9, -5, -3, -3,  0,  7, -1],  # F
    [ 1,  0,  0, -1, -3,  0, -1,  0,  0, -2, -3, -1, -2, -5,  6,  1,  0, -6, -5, -1],  # P
    [ 1,  0,  1,  0,  0, -1,  0,  1, -1, -1, -3,  0, -2, -3,  1,  2,  1, -2, -3, -1],  # S
    [ 1, -1,  0,  0, -2, -1,  0,  0, -1,  0, -2,  0, -1, -3,  0,  1,  3, -5, -3,  0],  # T
    [-6,  2, -4, -7, -8, -5, -7, -7, -3, -5, -2, -3, -4,  0, -6, -2, -5, 17,  0, -6],  # W
    [-3, -4, -2, -4,  0, -4, -4, -5,  0, -1, -1, -4, -2,  7, -5, -3, -3,  0, 10, -2],  # Y
    [ 0, -2, -2, -2, -2, -2, -2, -1, -2,  4,  2, -2,  2, -1, -1, -1,  0, -6, -2,  4],  # V
])

def get_substitution_matrix(matrix_name: Literal['BLOSUM62', 'BLOSUM50', 'PAM250'] = "BLOSUM62") -> np.ndarray:
    """
    Get a substitution matrix by name.
    
    Parameters
    ----------
    matrix_name : str, default=`BLOSUM62`
        Name of the substitution matrix ('BLOSUM62', 'BLOSUM50', 'PAM250').
        
    Returns
    -------
    numpy.ndarray
        The substitution matrix as a 2D numpy array.
    """
    if matrix_name.upper() == "BLOSUM62":
        return BLOSUM62
    elif matrix_name.upper() == "BLOSUM50":
        return BLOSUM50
    elif matrix_name.upper() == "PAM250":
        return PAM250
    else:
        raise ValueError(f"Unknown substitution matrix: {matrix_name}")

def get_reordered_matrix(target_alphabet: List[str],
                         matrix_name: Literal['BLOSUM62'] = "BLOSUM62") -> np.ndarray:
    """
    Retrieves a predefined substitution matrix and reorders its rows
    and columns to match the provided target alphabet.

    Parameters
    ----------
    target_alphabet : List
        The alphabet to reorder substitution matrix columns and rows to
        match.
    
    matrix_name :str, default=`BLOSUM62`
        Name of the substitution matrix ('BLOSUM62', 'BLOSUM50', 'PAM250').

    Returns
    -------
    reordered_matrix : np.ndarray
        The reodered substitution matrix.
    """
    original_matrix = get_substitution_matrix(matrix_name)
    canonical_aa_map = {aa: i for i, aa in enumerate(_aa_order)}
    
    alphabet_size = len(target_alphabet)
    reordered_matrix = np.zeros((alphabet_size, alphabet_size), dtype=np.float64)
    
    for i, aa1 in enumerate(target_alphabet):
        for j, aa2 in enumerate(target_alphabet):
            try:
                original_idx1 = canonical_aa_map[aa1]
                original_idx2 = canonical_aa_map[aa2]
            except KeyError as e:
                raise ValueError(f"Amino acid '{e.args[0]}' in target alphabet is not canonical.")
            
            reordered_matrix[i, j] = original_matrix[original_idx1, original_idx2]
            
    return reordered_matrix

def get_substitution_score(aa1: str,
                           aa2: str,
                           matrix: np.ndarray,
                           alphabet: List[str]) -> float:
    """
    Get the substitution score for a pair of amino acids from a given
    matrix and its corresponding alphabet.

    Parameters
    ----------
    aa1 : str
        First amino acid (single-letter code).
    aa2 : str
        Second amino acid (single-letter code).
    matrix : np.ndarray
        The substitution matrix (can be reordered).
    alphabet : List[str]
        The alphabet corresponding to the matrix's order.

    Returns
    -------
    float
        Substitution score.
    """
    try:
        idx_map = {aa: i for i, aa in enumerate(alphabet)}
        i = idx_map[aa1]
        j = idx_map[aa2]
    except KeyError as e:
        raise ValueError(f"Amino acid '{e.args[0]}' not found in the provided alphabet.")
    
    return matrix[i, j]

def normalize_substitution_matrix(matrix: np.ndarray) -> np.ndarray:
    """
    Normalize a substitution matrix to have values between 0 and 1.

    Parameters
    ----------
    matrix : np.ndarray
        The substitution matrix.

    Returns
    -------
    matrix : np.ndarray
        The normalized substitution matrix.
    """
    min_val = np.min(matrix)
    if min_val < 0:
        matrix = matrix - min_val
    
    max_val = np.max(matrix)
    if max_val > 0:
        matrix = matrix / max_val
    
    return matrix