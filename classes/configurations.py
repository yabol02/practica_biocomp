import csv
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np


@dataclass
class ProblemConfig:
    """Configuration for optimization problems."""

    max_evaluations: int
    seed: Optional[int] = None
    output_dir: str = "results"

    def __post_init__(self):
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

@dataclass
class SingleObjectiveResult:
    """Results for single-objective optimization."""
    problem_name: str
    best_fitness: float
    best_solution: List
    evaluations_used: int
    history: List[float] = field(default_factory=list)
    
    def save_csv(self, filepath: str) -> None:
        """
        Save single-objective results to CSV.
        
        :param filepath: Output CSV path
        :type filepath: str
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['generation', 'min_fitness'])
            for gen, fitness in enumerate(self.history):
                writer.writerow([gen, fitness])
