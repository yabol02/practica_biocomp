import random
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ProblemConfig:
    """Configuration for optimization problems."""

    max_evaluations: int
    seed: Optional[int] = None
    output_dir: str = "results"

    def initialize_random_state(self):
        """Initialize random seeds for reproducibility."""
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
