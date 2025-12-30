"""
Individual class for genetic algorithms.
"""

import random
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np


class Individual(ABC):
    """Represents an individual in an evolutionary algorithm."""

    def __init__(self, genotype: Optional[List] = None, bounds: Optional[List] = None):
        """
        Initialize an individual.

        :param genotype: Individual's genetic representation
        :type genotype: Optional[List]
        """
        self.genotype: List = genotype if genotype is not None else []
        self.bounds: List = bounds
        self.fitness_value: float = 0.0
        self.is_evaluated: bool = False
        self._check_genotype_validity(bounds)

    @property
    def fitness(self) -> float:
        """Get fitness value."""
        return self.fitness_value

    @fitness.setter
    def fitness(self, value: float) -> None:
        """Set fitness value and mark as evaluated."""
        self.fitness_value = value
        self.is_evaluated = True

    @abstractmethod
    def _check_genotype_validity(self, bounds) -> bool:
        """
        Check if the genotype is valid according to problem constraints.

        :return: True if valid, False otherwise
        :rtype: bool
        """
        pass

    def __repr__(self) -> str:
        return f"Individual(fitness={self.fitness_value:.4f}, genotype={self.genotype})"

    def __str__(self) -> str:
        return f"Individual with fitness: {self.fitness_value:.4f}"


class RealIndividual(Individual):
    """Individual with real-valued genotype for continuous optimization."""

    def __init__(
        self,
        genotype: Optional[List[float]] = None,
        bounds: List[Tuple[float, float]] = None,
    ):
        """
        Initialize real-valued individual.

        :param genotype: Real-valued genes
        :type genotype: Optional[List[float]]
        :param bounds: Min/max bounds for each gene [(min, max), ...]
        :type bounds: List[Tuple[float, float]]
        """
        if genotype is None and bounds is not None:
            genotype = [random.uniform(low, high) for low, high in bounds]

        super().__init__(genotype, bounds)

    def _check_genotype_validity(self, bounds: List[Tuple[float, float]]) -> bool:
        """
        Check if genotype values are within specified bounds.

        :param bounds: Min/max bounds for each gene [(min, max), ...]
        :type bounds: List[Tuple[float, float]]
        :return: True if valid, False otherwise
        """

        if bounds is None or self.genotype is None:
            return True

        genes = np.asanyarray(self.genotype)
        array_bounds = np.asanyarray(bounds)
        low_limits = array_bounds[:, 0]
        high_limits = array_bounds[:, 1]

        if len(genes) != len(bounds):
            raise ValueError(
                f"Differnce of dimensions: Genotype({len(genes)}) != Bounds({len(bounds)})"
            )

        out_low = genes < low_limits
        out_high = genes > high_limits

        if np.any(out_low | out_high):
            indices = np.where(out_low | out_high)[0]
            raise ValueError(
                f"Genes out of bounds at indices {indices}: "
                f"Values {genes[indices]} out of [{low_limits[indices]}, {high_limits[indices]}]"
            )

        return True

