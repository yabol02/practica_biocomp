import random
from abc import ABC, abstractmethod
from typing import List, Tuple

from .individual import RealIndividual
from .population import Population
from .problems import Problem


class Initialization(ABC):
    """Abstract base class for population initialization strategies."""

    @abstractmethod
    def initialize(
        self, population_size: int, bounds: List[Tuple[float, float]], problem: Problem
    ) -> Population:
        """
        Initialize population.

        :param population_size: Number of individuals
        :type population_size: int
        :param bounds: Variable bounds
        :type bounds: List[Tuple[float, float]]
        :param problem: Problem instance for evaluation tracking
        :type problem: Problem
        :return: Initialized population
        :rtype: Population
        """
        pass

    def __call__(
        self, population_size: int, bounds: List[Tuple[float, float]], problem: Problem
    ) -> Population:
        return self.initialize(population_size, bounds, problem)


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
        :type bounds: List[Tuple[float, float]]
        :param problem: Problem instance
        :type problem: Problem
        :return: Random population
        :rtype: Population
        """
        individuals = [RealIndividual(bounds=bounds) for _ in range(population_size)]
        return Population(individuals, minimize=True)
