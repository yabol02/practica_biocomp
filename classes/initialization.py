import random
from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np

from .individual import PermutationIndividual, RealIndividual
from .population import Population
from .problems import SingleObjectiveProblem, TSProblem


class Initialization(ABC):
    """Abstract base class for population initialization strategies."""

    @abstractmethod
    def initialize(
        self, population_size: int, bounds: List, problem: SingleObjectiveProblem
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
        problem: SingleObjectiveProblem,
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
        return Population(individuals, minimize=problem.minimize)


class PermutationInitialization(Initialization):
    """Random permutation initialization."""

    def initialize(
        self,
        population_size: int,
        bounds: Tuple[int, int],
        problem: SingleObjectiveProblem,
    ) -> Population:
        """
        Create a population of individuals with permuted genotypes.

        :param population_size: Number of individuals to create.
        :type population_size: int
        :param bounds: Tuple defining the range of integers (min, max).
        :type bounds: Tuple[int, int]
        :param problem: Problem instance.
        :type problem: SingleObjectiveProblem
        :return: Population of PermutationIndividuals.
        """
        individuals = [
            PermutationIndividual(bounds=bounds) for _ in range(population_size)
        ]
        return Population(individuals, minimize=problem.minimize)


class NeighborInitialization(Initialization):
    """Population initialization for TSP using a stochastic Nearest Neighbor heuristic."""

    def __init__(self, k_best: int = 3):
        """
        Initialize the operator.

        :param k_best: Number of nearest neighbors considered at each step.
                       If k_best = 1, the heuristic reduces to the classical
                       deterministic Nearest Neighbor.
        :type k_best: int
        """
        if k_best < 1:
            raise ValueError("k_best must be >= 1")

        self.k_best = k_best

    def initialize(
        self,
        population_size: int,
        bounds: Tuple[int, int],
        problem: TSProblem,
    ) -> Population:
        """
        Generate an initial population using a k-Nearest Neighbor heuristic.

        :param population_size: Number of individuals to generate.
        :type population_size: int
        :param bounds: Genotype bounds used by permutation individuals.
        :type bounds: Tuple[int, int]
        :param problem: Traveling Salesman Problem instance.
        :type problem: TSProblem
        :return: Initialized population.
        :rtype: Population
        """
        individuals = []
        n_cities = problem.dist_matrix.shape[0]

        for _ in range(population_size):
            unvisited = list(range(n_cities))
            current = random.choice(unvisited)
            unvisited.remove(current)

            genotype = [current]

            while unvisited:
                distances = [problem.dist_matrix[current, city] for city in unvisited]

                k = min(self.k_best, len(unvisited))
                nearest_indices = np.argsort(distances)[:k]
                chosen_idx = random.choice(nearest_indices)
                next_city = unvisited[chosen_idx]

                genotype.append(next_city)
                unvisited.remove(next_city)
                current = next_city

            individuals.append(PermutationIndividual(genotype=genotype, bounds=bounds))

        return Population(individuals, minimize=problem.minimize)
