import pytest
import numpy as np
from softalign.soft_alignment import (
    pairwise_align,
    build_guide_tree,
    profile_align,
    progressive_alignment,
    remove_all_gap_columns,
    refine_alignment,
    align_soft_sequences,
)
from softalign.distance_metrics_with_blosum import (
    batch_pairwise_distance,
    batch_column_distance,
)
from softalign._blosum import get_reordered_matrix

# Define a standard alphabet for testing
ALPHABET = list("ARNDCQEGHILKMFPSTWYV")
ALPHABET_SIZE = len(ALPHABET)


@pytest.fixture
def soft_sequence_factory():
    def _create_soft_sequence(length, alphabet_size):
        seq = np.random.rand(length, alphabet_size).astype(np.float32)
        return seq / seq.sum(axis=1, keepdims=True)
    return _create_soft_sequence

@pytest.fixture
def identical_soft_sequences(soft_sequence_factory):
    np.random.seed(0) 
    seq = soft_sequence_factory(10, ALPHABET_SIZE)
    return [seq, seq.copy()]

def test_pairwise_align(identical_soft_sequences):
    """
    Test pairwise alignment of identical sequences.
    """
    seq1, seq2 = identical_soft_sequences
    aligned1, aligned2, score = pairwise_align(seq1, seq2, ALPHABET)
    # Use allclose for floating-point comparison
    assert np.allclose(aligned1, aligned2)
    assert score < 0 # Alignment score should be negative

def test_build_guide_tree():
    """
    Test the construction of a guide tree from a distance matrix.
    """
    distances = np.array([[0, 0.1, 0.5], [0.1, 0, 0.4], [0.5, 0.4, 0]])
    guide_tree = build_guide_tree(distances)
    assert isinstance(guide_tree, dict)
    assert len(guide_tree) == 2

def test_profile_align(identical_soft_sequences):
    """
    Test the alignment of identical profiles.
    """
    seq1, seq2 = identical_soft_sequences

    profile1_with_gap = np.zeros((seq1.shape[0], ALPHABET_SIZE + 1), dtype=np.float32)
    profile1_with_gap[:, :ALPHABET_SIZE] = seq1
    profile1 = np.expand_dims(profile1_with_gap, axis=0)

    profile2_with_gap = np.zeros((seq2.shape[0], ALPHABET_SIZE + 1), dtype=np.float32)
    profile2_with_gap[:, :ALPHABET_SIZE] = seq2
    profile2 = np.expand_dims(profile2_with_gap, axis=0)
    
    aligned_profile1, aligned_profile2, score = profile_align(
        profile1, profile2, ALPHABET
    )
    # Use allclose for floating-point comparison
    assert np.allclose(aligned_profile1, aligned_profile2)
    assert score < 0

def test_progressive_alignment(identical_soft_sequences):
    """
    Test progressive alignment of identical sequences.
    """
    distances = np.array([[0, 0.01], [0.01, 0]]) # Use a small non-zero distance
    guide_tree = build_guide_tree(distances)
    alignment, score = progressive_alignment(
        identical_soft_sequences, guide_tree, ALPHABET
    )
    # Use allclose for floating-point comparison
    assert np.allclose(alignment[0], alignment[1])
    assert score < 0

def test_remove_all_gap_columns():
    """
    Test the removal of all-gap columns from an alignment.
    """
    # Create a dummy alignment with a gap column
    alignment = np.zeros((2, 5, ALPHABET_SIZE + 1))
    alignment[:, 2, ALPHABET_SIZE] = 1.0  # All-gap column
    result = remove_all_gap_columns(alignment, ALPHABET)
    assert result[0].shape[0] == 4

def test_refine_alignment(identical_soft_sequences):
    """
    Test the refinement of an alignment.
    """
    # Create a dummy alignment and score
    seq1, seq2 = identical_soft_sequences
    alignment = [
        np.zeros((10, ALPHABET_SIZE + 1)),
        np.zeros((10, ALPHABET_SIZE + 1)),
    ]
    alignment[0][: , :ALPHABET_SIZE] = seq1
    alignment[1][: , :ALPHABET_SIZE] = seq2
    
    alignment_score = -100.0
    _, improved, new_score = refine_alignment(
        alignment, alignment_score, ALPHABET
    )
    assert isinstance(improved, bool)
    assert isinstance(new_score, float)


def test_align_soft_sequences(identical_soft_sequences):
    """
    Test the main alignment function with identical sequences.
    """
    alignment, score = align_soft_sequences(identical_soft_sequences, ALPHABET)
    # Use allclose for floating-point comparison
    assert np.allclose(alignment[0], alignment[1])
    assert score < 0

def test_batch_pairwise_distance():
    """
    Test the calculation of pairwise distances between columns.
    """
    cols1 = np.random.rand(5, ALPHABET_SIZE)
    cols2 = np.random.rand(3, ALPHABET_SIZE)
    distances = batch_pairwise_distance(cols1, cols2, ALPHABET)
    assert distances.shape == (5, 3)

def test_batch_column_distance():
    """
    Test the calculation of distances between corresponding columns.
    """
    cols1 = np.random.rand(5, ALPHABET_SIZE)
    cols2 = np.random.rand(5, ALPHABET_SIZE)
    distances = batch_column_distance(cols1, cols2, ALPHABET)
    assert distances.shape == (5,)

def test_get_reordered_matrix():
    """
    Test the reordering of a substitution matrix.
    """
    reordered_matrix = get_reordered_matrix(ALPHABET, "BLOSUM62")
    assert reordered_matrix.shape == (ALPHABET_SIZE, ALPHABET_SIZE)