"""
Individual class for genetic algorithms.
"""

import random
from abc import ABC
from typing import List, Optional, Tuple


class Individual(ABC):
    """Represents an individual in an evolutionary algorithm."""

    def __init__(self, genotype: Optional[List] = None):
        """
        Initialize an individual.

        :param genotype: Individual's genetic representation
        :type genotype: Optional[List]
        """
        self.genotype: List = genotype if genotype is not None else []
        self.fitness_value: float = 0.0
        self.is_evaluated: bool = False

    @property
    def fitness(self) -> float:
        """Get fitness value."""
        return self.fitness_value

    @fitness.setter
    def fitness(self, value: float) -> None:
        """Set fitness value and mark as evaluated."""
        self.fitness_value = value
        self.is_evaluated = True

    def __repr__(self) -> str:
        return f"Individual(fitness={self.fitness_value:.4f}, genotype={self.genotype})"

    def __str__(self) -> str:
        return f"Individual with fitness: {self.fitness_value:.4f}"
