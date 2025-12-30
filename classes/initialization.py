from abc import ABC, abstractmethod
from typing import List, Tuple

from .individual import PermutationIndividual, RealIndividual
from .population import Population
from .problems import Problem


class Initialization(ABC):
    """Abstract base class for population initialization strategies."""

    @abstractmethod
    def initialize(
        self, population_size: int, bounds: List, problem: Problem
    ) -> Population:
        """
        Initialize population.

        :param population_size: Number of individuals
        :type population_size: int
        :param bounds: Variable bounds
        :type bounds: List
        :param problem: Problem instance for evaluation tracking
        :type problem: Problem
        :return: Initialized population
        :rtype: Population
        """
        pass

    def __call__(self, *args, **kwargs) -> Population:
        """Alias for `initialize` method."""
        return self.initialize(*args, **kwargs)


class RandomInitialization(Initialization):
    """Random uniform initialization within bounds."""

    def initialize(
        self,
        population_size: int,
        bounds: List[Tuple[float, float]],
        problem: Problem,
    ) -> Population:
        """
        Create random population.

        :param population_size: Number of individuals
        :type population_size: int
        :param bounds: Variable bounds
        :type bounds: List
        :param problem: Problem instance
        :type problem: Problem
        :return: Random population
        :rtype: Population
        """
        individuals = [RealIndividual(bounds=bounds) for _ in range(population_size)]
        return Population(individuals, minimize=True)


class PermutationInitialization(Initialization):
    """Random permutation initialization."""

    def initialize(
        self,
        population_size: int,
        bounds: Tuple[int, int],
        problem: Problem,
    ) -> Population:
        """
        Create a population of individuals with permuted genotypes.

        :param population_size: Number of individuals to create.
        :param bounds: Tuple defining the range of integers (min, max).
        :param problem: Problem instance.
        :return: Population of PermutationIndividuals.
        """
        individuals = [
            PermutationIndividual(bounds=bounds) for _ in range(population_size)
        ]
        return Population(individuals, minimize=True)
