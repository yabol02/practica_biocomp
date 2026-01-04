"""
Individual class for genetic algorithms.
"""

import random
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

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
        self._fitness_value: Optional[np.ndarray[float]] = None
        self.is_evaluated: bool = False
        self._check_genotype_validity(bounds)

    @property
    def fitness(self) -> Union[float, np.ndarray]:
        """
        Get fitness value.
        Returns a scalar if it's mono-objective (1x1) or the array if it's multi-objective.

        :return: Fitness value
        :rtype: Union[float, np.ndarray]
        """
        if self._fitness_value is None:
            return 0.0

        if self._fitness_value.size == 1:
            return self._fitness_value.item()

        return self._fitness_value

    @fitness.setter
    def fitness(self, value: float) -> None:
        """
        Set fitness value and mark as evaluated.
        Ensures the input is stored as a numpy array internally.

        :param value: Fitness value
        :type value: float
        """
        self._fitness_value = np.atleast_1d(value)
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
        fitness_str = (
            f"{self.fitness:.4f}"
            if isinstance(self.fitness, (float, int))
            else np.array2string(self.fitness, precision=3)
        )
        fitness_str = fitness_str if self.is_evaluated else "None"
        geno = str(self.genotype)
        if len(geno) > 30:
            geno = f"{geno[:10]}...{geno[-10:]}"
        return f"{self.__class__.__name__}(fitness={fitness_str}, genotype={geno})"

    def __str__(self) -> str:
        fitness_str = (
            f"{self.fitness:.4f}"
            if isinstance(self.fitness, (float, int))
            else np.array2string(self.fitness, precision=3)
        )
        fitness_str = fitness_str if self.is_evaluated else "None"
        return f"<{self.__class__.__name__} | fitness: {fitness_str}>"


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


class PermutationIndividual(Individual):
    """Individual with integer-valued genotype for permutation problems."""

    def __init__(
        self,
        genotype: Optional[np.ndarray] = None,
        bounds: Tuple[int, int] = None,
    ):
        """
        :param genotype: Optional predefined permutation.
        :type genotype: Optional[List[int]]
        :param bounds: Tuple defining the range of integers (min, max).
        :type bounds: Tuple[int, int]
        """
        if genotype is None and bounds is not None:
            genotype = np.arange(bounds[0], bounds[1] + 1, dtype=int)
            np.random.shuffle(genotype)

        super().__init__(genotype, bounds)

    def _check_genotype_validity(self, bounds: Tuple[int, int]) -> bool:
        """
        Check if genotype is a valid permutation within bounds.

        :param bounds: Tuple defining the range of integers (min, max).
        :type bounds: Tuple[int, int]
        :return: True if valid, raises ValueError otherwise.
        :rtype: bool
        """
        if bounds is None or self.genotype is None:
            return True

        low, high = bounds
        expected_len = high - low + 1
        gen = np.asanyarray(self.genotype)

        if len(gen) != expected_len:
            raise ValueError(f"Incorrect length: {len(gen)} (expected: {expected_len})")

        if np.any((gen < low) | (gen > high)):
            raise ValueError(f"Values out of range [{low}, {high}]: {gen}")

        if len(np.unique(gen)) != expected_len:
            vals, counts = np.unique(gen, return_counts=True)
            raise ValueError(f"Duplicate genes detected: {vals[counts > 1]}")
        return True
