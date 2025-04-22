from importlib import import_module
import numpy as np 

import_module("softalign._softalign")

from .soft_alignment import align_soft_sequences_with_blosum  # noqa: F401