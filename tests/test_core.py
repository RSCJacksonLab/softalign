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
        seq = np.random.rand(length, alphabet_size)
        return seq / seq.sum(axis=1, keepdims=True)
    return _create_soft_sequence

@pytest.fixture
def identical_soft_sequences(soft_sequence_factory):
    seq = soft_sequence_factory(10, ALPHABET_SIZE)
    return [seq, seq.copy()]

def test_pairwise_align(identical_soft_sequences):
    """
    Test pairwise alignment of identical sequences.
    """
    seq1, seq2 = identical_soft_sequences
    aligned1, aligned2, score = pairwise_align(seq1, seq2, ALPHABET)
    assert np.array_equal(aligned1, aligned2)
    assert score > 0

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
    profile1 = np.expand_dims(identical_soft_sequences[0], axis=0)
    profile2 = np.expand_dims(identical_soft_sequences[1], axis=0)
    aligned_profile1, aligned_profile2, score = profile_align(
        profile1, profile2, ALPHABET
    )
    assert np.array_equal(aligned_profile1, aligned_profile2)
    assert score > 0

def test_progressive_alignment(identical_soft_sequences):
    """
    Test progressive alignment of identical sequences.
    """
    distances = np.array([[0, 0.1], [0.1, 0]])
    guide_tree = build_guide_tree(distances)
    alignment, score = progressive_alignment(
        identical_soft_sequences, guide_tree, ALPHABET
    )
    assert np.array_equal(alignment[0], alignment[1])
    assert score > 0

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
    alignment = [
        np.zeros((10, ALPHABET_SIZE + 1)),
        np.zeros((10, ALPHABET_SIZE + 1)),
    ]
    alignment_score = -10.0
    improved_alignment, improved, new_score = refine_alignment(
        alignment, alignment_score, ALPHABET
    )
    assert improved
    assert new_score > alignment_score

def test_align_soft_sequences(identical_soft_sequences):
    """
    Test the main alignment function with identical sequences.
    """
    alignment, score = align_soft_sequences(identical_soft_sequences, ALPHABET)
    assert np.array_equal(alignment[0], alignment[1])
    assert score > 0

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