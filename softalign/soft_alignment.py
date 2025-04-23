from __future__ import annotations
from typing import Literal, List, Union, Dict, Any
import numpy as np

from softalign._softalign import nw_affine
from ._blossum import get_substitution_matrix
import numpy as np
from .distance_metrics_with_blosum import batch_column_distance
from ._blossum import get_substitution_matrix
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

def pairwise_align(
    seq1: np.ndarray,
    seq2: np.ndarray,
    gap_open: float = 10.0,
    gap_extend: float = 0.5,
    substitution_matrix: Union[str, np.ndarray] = "BLOSUM62",
    alpha: float = 0.5):
    """
    """
    seq1_f = np.asarray(seq1, dtype=np.float32)
    seq2_f = np.asarray(seq2, dtype=np.float32)

    if isinstance(substitution_matrix, str):
        subst = get_substitution_matrix(substitution_matrix).astype(np.float32)
    else:
        subst = np.asarray(substitution_matrix, dtype=np.float32)

    aligned1, aligned2, score = nw_affine(
        seq1_f, seq2_f, subst, gap_open, gap_extend, alpha
    )
    return aligned1, aligned2, score


def build_guide_tree(distances: np.ndarray) -> Dict:
    """
    Build a guide tree from a distance matrix using UPGMA.
    
    Parameters
    ----------
    distances : numpy.ndarray
        Distance matrix.
        
    Returns
    -------
    Dict
        A guide tree represented as a dictionary with node IDs as keys
              and (left_child, right_child, distance) tuples as values
    """
    n = distances.shape[0]
    condensed = squareform(distances, checks=False)
    
    # Compute the linkage matrix using the "average" method,
    # which corresponds to UPGMA.
    Z = linkage(condensed, method='average')
    
    # Build the guide tree from the linkage matrix (n-1, 4).
    guide_tree = {}
    for i, row in enumerate(Z):
        left_child = int(row[0])
        right_child = int(row[1])
        dist = row[2]
        # New cluster IDs start at n and increase by 1 for each merge.
        new_node_id = n + i
        guide_tree[new_node_id] = (left_child, right_child, dist)
    
    return guide_tree


def profile_align(profile1: np.ndarray,
                  profile2: np.ndarray,
                  pairwise_align: Any = pairwise_align,
                  gap_open: float = 10.0,
                  gap_extend: float = 0.5,
                  substitution_matrix: Literal["BLOSUM62", "BLOSUM50", "PAM250"] = 'BLOSUM62',
                  alpha: float = 0.5) -> tuple:
    """
    Align two profiles (sets of aligned sequences) using dynamic
    programming.
    
    Parameters
    ----------
    profile1 : np.ndarray
        First array of aligned sequences with shape (L1, 21).
    
    profile2 : np.ndarray
        Second array of aligned sequences with shape (L2, 21).

    pairwise_align : callable
        The pairwise alignment function.
        
    gap_open : float
        Gap opening penalty.

    gap_extend : float
        Gap extension penalty.
    
    metric : str, default=`hybrid`
        Distance metric to use ('jensen_shannon', 'kl_divergence', or
        'hybrid').

    substitution_matrix : str or numpy.ndarray, default=`BLOSUM62`
        Substitution matrix name or matrix array
        
    alpha : float
        Weight for the standard distance metric in hybrid mode.
        
    Returns
    -------
    aligned_profile1 : np.ndarray
        Aligned profile 1 with shape (N1, L', 21).
        
    aligned_profile2 : np.ndararay
        Aligned profile 2 with shape (N1, L', 21).
    """
    # Convert profiles to average probability distributions per column
    avg_profile1 = np.mean(profile1, axis=0)
    avg_profile2 = np.mean(profile2, axis=0)
    
    # Align the average profiles
    aligned_avg1, aligned_avg2, score = pairwise_align(avg_profile1[:, :20],
                                                       avg_profile2[:, :20], 
                                                       gap_open,
                                                       gap_extend,
                                                       substitution_matrix,
                                                       alpha)
    
    align_length = aligned_avg1.shape[0]

    aligned_profile1 = np.zeros((profile1.shape[0], align_length, 21))
    aligned_profile2 = np.zeros((profile2.shape[0], align_length, 21))
    
    pos1, pos2 = 0, 0
    for i in range(align_length):
        is_gap1 = aligned_avg1[i, 20] > 0.5
        is_gap2 = aligned_avg2[i, 20] > 0.5
        
        if not is_gap1:
            if pos1 < profile1.shape[1]:
                aligned_profile1[:, i, :] = profile1[:, pos1, :]
            else:
                aligned_profile1[:, i, :] = 0
                aligned_profile1[:, i, 20] = 1.0
            pos1 += 1
        
        if not is_gap2:
            if pos2 < profile2.shape[1]:
                aligned_profile2[:, i, :] = profile2[:, pos2, :]
            else:
                aligned_profile2[:, i, :] = 0
                aligned_profile2[:, i, 20] = 1.0
            pos2 += 1

    return aligned_profile1, aligned_profile2, score


def progressive_alignment(sequences: List,
                          guide_tree: Dict,
                          pairwise_align: Any = pairwise_align,
                          gap_open: float = 10.0,
                          gap_extend: float = 0.5,
                          substitution_matrix: Literal["BLOSUM62", "BLOSUM50", "PAM250"] = 'BLOSUM62',
                          alpha: float = 0.5):
    """
    Perform progressive alignment following a guide tree.
    
    Parameters
    -----------
    sequences : List
        List of soft sequences as arrays of shape (L, 20).
    
    guide_tree : Dict
        Guide tree to use for progressive alignment.
    
    pairwise_align : Callable
        Pairwise alignment function.
    
    gap_open : float
        Gap opening penalty.

    gap_extend : float
        Gap extension penalty.
    
    metric : str, default=`hybrid`
        Distance metric to use ('jensen_shannon', 'kl_divergence', or
        'hybrid').

    substitution_matrix : str or numpy.ndarray, default=`BLOSUM62`
        Substitution matrix name or matrix array
        
    alpha : float
        Weight for the standard distance metric in hybrid mode.
        
    Returns
    -------
    List
        List of aligned sequences as arrays of shape (L', 21)
    """
    n = len(sequences)
    
    # Add gap dimension to sequences
    seqs_with_gaps = []
    for seq in sequences:
        seq_with_gap = np.zeros((seq.shape[0], 21))
        seq_with_gap[:, :20] = seq
        seqs_with_gaps.append(seq_with_gap)
    
    profiles = {i: np.expand_dims(seqs_with_gaps[i], axis=0) for i in range(n)}

    for node_id in sorted(guide_tree.keys()):
        left_child, right_child, _ = guide_tree[node_id]
        
        # Align the profiles of the two children
        aligned_left, aligned_right, score = profile_align(
            profiles[left_child], 
            profiles[right_child],
            pairwise_align,
            gap_open,
            gap_extend,
            substitution_matrix,
            alpha
        )
        
        # Merge the aligned profiles
        profiles[node_id] = np.vstack([aligned_left, aligned_right])
        
        # Remove the children to save memory
        if left_child >= n:  # Only remove internal nodes
            del profiles[left_child]
        if right_child >= n:  # Only remove internal nodes
            del profiles[right_child]
    
    root_id = max(guide_tree.keys())

    return [profiles[root_id][i] for i in range(len(sequences))], score


def remove_all_gap_columns(aligned_sequences: np.ndarray) -> np.ndarray:
    """
    Remove columns that are gaps in all sequences.
    
    Parameters
    ----------
    aligned_sequences : List
        List of aligned sequences as arrays of shape (L, 21).
        
    Returns
    -------
    result : List
        List of sequences with all-gap columns removed, with entries of
        shape (L, 20).
    """
    # Convert to numpy array for easier manipulation
    alignment = np.array(aligned_sequences)
    
    # Find columns that are gaps in all sequences
    all_gap_columns = np.all(alignment[:, :, 20] > 0.5, axis=0)
    
    # Create a mask for columns to keep
    keep_columns = ~all_gap_columns
    
    # Remove all-gap columns
    result = []
    for seq in aligned_sequences:
        result.append(seq[keep_columns])
    
    return result

def refine_alignment(aligned_sequences,
                     alignment_score,
                     pairwise_align: Any = pairwise_align,
                     gap_open=10.0,
                     gap_extend=0.5,
                     substitution_matrix="BLOSUM62",
                     alpha=0.5):
    """
    Refine the alignment through iterative optimization.
    
    Parameters
    ----------
    aligned_sequences : List 
        List of aligned sequences as arrays of shape (L, 21)
    
    pairwise_align : callable
        The pairwise alignment function.
    
    gap_open : float
        Gap opening penalty.

    gap_extend : float
        Gap extension penalty.
    
    metric : str, default=`hybrid`
        Distance metric to use ('jensen_shannon', 'kl_divergence', or
        'hybrid').

    substitution_matrix : str or numpy.ndarray, default=`BLOSUM62`
        Substitution matrix name or matrix array
        
    alpha : float
        Weight for the standard distance metric in hybrid mode.
    
    Returns
    -------
    improved_alignment : np.ndarray
        The refined alignment.
        
    improved : bool
        Boolean for whether the alignment has been improved.
    """
    n = len(aligned_sequences)
    improved = False
    old_score = alignment_score
    
    # Try different ways to split the alignment
    for k in range(1, n):
        # Split into two groups
        group1 = aligned_sequences[:k]
        group2 = aligned_sequences[k:]
        
        # Remove gaps that are in all sequences of a group
        group1_nogaps = remove_all_gap_columns(group1)
        group2_nogaps = remove_all_gap_columns(group2)
        
        # Realign the two groups
        aligned_group1, aligned_group2, new_score = profile_align(
            np.array(group1_nogaps),
            np.array(group2_nogaps),
            pairwise_align,
            gap_open,
            gap_extend,
            substitution_matrix,
            alpha)
        
        # Combine the realigned groups
        new_alignment = list(aligned_group1) + list(aligned_group2)
        
        # If the new alignment is better, keep it
        if new_score > old_score:
            aligned_sequences = new_alignment
            old_score = new_score
            improved = True
            break
    
    return aligned_sequences, improved, new_score


def align_soft_sequences_with_blosum(sequences: List,
                                     pairwise_align: Any = pairwise_align,
                                     gap_open: float = 10.0,
                                     gap_extend: float = 0.5,
                                     distance_metric: Literal['jensen_shannon', 'kl_divergence', 'hybrid'] = 'hybrid',
                                     substitution_matrix: Literal["BLOSUM62", "BLOSUM50", "PAM250"] = 'BLOSUM62',
                                     alpha: float = 0.5,
                                     max_iterations: int = 3):
    """
    Align a list of soft sequences (probability distributions over amino acids)
    using substitution matrices for scoring.
    
    Parameters
    ----------
    sequences : List 
        List of  sequences as arrays of shape (L, 20). To be aligned.
    
    pairwise_align : callable
        The pairwise alignment function.
    
    gap_open : float
        Gap opening penalty.

    gap_extend : float
        Gap extension penalty.
    
    distance_metric : str, default=`hybrid`
        Distance metric to use ('jensen_shannon', 'kl_divergence', or
        'hybrid').

    substitution_matrix : str or numpy.ndarray, default=`BLOSUM62`
        Substitution matrix name or matrix array
        
    alpha : float
        Weight for the standard distance metric in hybrid mode.
    
    max_iterations : int
        Maximum number of refinement iterations.
        
    Returns
    -------
    list
        List of aligned sequences as numpy arrays with shape (L', 21).
    """

    if not sequences:
        raise ValueError("Input sequences list cannot be empty")
    
    for i, seq in enumerate(sequences):
        if not isinstance(seq, np.ndarray):
            raise TypeError(f"Sequence {i} is not a numpy array")
        
        if seq.ndim != 2 or seq.shape[1] != 20:
            raise ValueError(f"Sequence {i} has shape {seq.shape}, expected (L, 20)")
        
        # Check if rows sum to approximately 1
        row_sums = np.sum(seq, axis=1)
        if not np.allclose(row_sums, 1.0, rtol=1e-5):
            raise ValueError(f"Sequence {i} contains rows that do not sum to 1.0")
    
    # Validate parameters
    if distance_metric not in ['hybrid', 'jensen_shannon', 'kl_divergence']:
        raise ValueError(f"Unknown distance metric: {distance_metric}")
    
    if alpha < 0 or alpha > 1:
        raise ValueError(f"Alpha must be between 0 and 1, got {alpha}")
    
    if gap_open < 0:
        raise ValueError(f"Gap opening penalty must be non-negative, got {gap_open}")
    
    if gap_extend < 0:
        raise ValueError(f"Gap extension penalty must be non-negative, got {gap_extend}")
    
    if max_iterations < 0:
        raise ValueError(f"Maximum iterations must be non-negative, got {max_iterations}")

    n = len(sequences)
    if n < 2:
        # If only one sequence, just add the gap dimension and return
        if n == 1:
            seq_with_gap = np.zeros((sequences[0].shape[0], 21))
            seq_with_gap[:, :20] = sequences[0]
            return [seq_with_gap]
        return []
    
    # Get the substitution matrix if a name is provided
    if isinstance(substitution_matrix, str):
        matrix = get_substitution_matrix(substitution_matrix)
    else:
        matrix = substitution_matrix
    
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):

            min_len = min(sequences[i].shape[0], sequences[j].shape[0])
            
            if min_len > 0:
                
                batch_dists = batch_column_distance(
                    sequences[i][:min_len], 
                    sequences[j][:min_len], 
                    distance_metric, matrix, alpha
                )
                dist = np.mean(batch_dists)
            else:
                dist = 0.0
            
            # Add penalty for length difference
            len_diff = abs(sequences[i].shape[0] - sequences[j].shape[0])
            if len_diff > 0:
                dist += (gap_open + (len_diff - 1) * gap_extend) / max(sequences[i].shape[0], sequences[j].shape[0])
            
            distances[i, j] = distances[j, i] = dist
    
    guide_tree = build_guide_tree(distances)
    
    alignment, alignment_sore = progressive_alignment(sequences,
                                                      guide_tree,
                                                      pairwise_align,
                                                      gap_open,
                                                      gap_extend,
                                                      matrix,
                                                      alpha)
    
    for _ in range(max_iterations):
        alignment, improved, score = refine_alignment(alignment,
                                                      alignment_sore,
                                                      pairwise_align,
                                                      gap_open,
                                                      gap_extend,
                                                      matrix,
                                                      alpha)
        if not improved:
            break
    
    return alignment