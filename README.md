# softalign
A high-performance Python package for aligning "soft" biological sequences (probability distributions) using a C++ backend accelerated with AVX and OpenMP.

## Installation

Install directly from the source repository:

```bash
pip install git+https://github.com/RSCJacksonLab/softalign.git
```

## Usage Example

```python
import numpy as np
from softalign import align_soft_sequences

# Define the standard 20-amino-acid alphabet
ALPHABET = list("ARNDCQEGHILKMFPSTWYV")
ALPHABET_SIZE = len(ALPHABET)

seq1 = np.random.rand(10, ALPHABET_SIZE)
seq1 = seq1 / seq1.sum(axis=1, keepdims=True)

seq2 = np.random.rand(12, ALPHABET_SIZE)
seq2 = seq2 / seq2.sum(axis=1, keepdims=True)

alignment, score = align_soft_sequences(
    sequences=[seq1, seq2],
    alphabet=ALPHABET,
    gap_open=10.0,
    gap_extend=0.5
)

aligned_seq1 = alignment[0]
aligned_seq2 = alignment[1]

print(f"Alignment Score: {score:.2f}")
print(f"Alignment Length: {aligned_seq1.shape[0]}")
```

## Testing
To run the test suite, clone the repository and install the testing dependencies.

```bash
git clone https://github.com/RSCJacksonLab/softalign.git
cd softalign

# Install in editable mode with testing dependencies
# Use quotes for zsh compatibility
pip install -e '.[test]'

# Run tests
pytest
```

## License
This project is licensed under the MIT License. See the LICENSE file for details.