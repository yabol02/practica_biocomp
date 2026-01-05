import random
from abc import ABC, abstractmethod
from typing import List, Union

import numpy as np

from .population import Population


class Selection(ABC):
    """Abstract base class for selection operators."""

    @abstractmethod
    def select(self, population: Population, size: int) -> Population:
        """
        Select individuals from population.

        :param population: Population to select from
        :type population: Population
        :param size: Number of individuals to select
        :type size: int
        :return: Selected population
        :rtype: Population
        """
        pass

    def __call__(self, *args, **kwargs) -> Population:
        """Alias for `select` method."""
        return self.select(*args, **kwargs)

    def __repr__(self) -> str:
        attrs = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"{self.__class__.__name__}({attrs})"

    def __str__(self) -> str:
        return f"<{self.__class__.__name__}>"


class TournamentSelection(Selection):
    """Tournament selection operator."""

    def __init__(self, tournament_size: int = 3):
        """
        Initialize tournament selection.

        :param tournament_size: Number of individuals per tournament
        :type tournament_size: int
        """
        self.tournament_size = tournament_size

    def select(self, population: Population, size: int) -> Population:
        """
        Select individuals via tournament.

        :param population: Population to select from
        :type population: Population
        :param size: Number of individuals to select
        :type size: int
        :return: Selected population
        :rtype: Population
        """
        selected = []
        class_ind = population.ind_class
        bounds = population.bounds

        for _ in range(size):
            tournament = random.sample(population.individuals, self.tournament_size)
            winner = min(
                tournament,
                key=lambda ind: ind.fitness if population.minimize else -ind.fitness,
            )
            selected.append(class_ind(genotype=winner.genotype.copy(), bounds=bounds))

        return Population(
            selected,
            minimize=population.minimize,
            multiobjective=population.multiobjective,
        )


class WeightedSelection(Selection):
    """
    Selects individuals based on a weighted sum of their objectives.
    Useful when some objectives are more important than others.
    """

    def __init__(self, weights: Union[List[float], np.ndarray], minimize: bool = True):
        """
        :param weights: Importance of each objective.
        :type weights: Union[List[float], np.ndarray]
        :param minimize: True if lower weighted scores are better.
        :type minimize: bool
        """
        self.weights = np.asarray(weights)
        self.minimize = minimize

    def select(self, population: Population, size: int) -> Population:
        """
        Selects the top N individuals based on their weighted fitness.

        :param population: Population to select from
        :type population: Population
        :param size: Number of individuals to select
        :type size: int
        :return: Selected population
        :rtype: Population
        """
        scores = [self._calculate_weighted_score(ind.fitness) for ind in population]
        indices = np.argsort(scores)
        if not self.minimize:
            indices = indices[::-1]

        selected_indices = indices[:size]
        return Population(
            [population[i] for i in selected_indices],
            minimize=True,
            multiobjective=True,
        )

    def _calculate_weighted_score(self, fitness: Union[float, np.ndarray]) -> float:
        """
        Reduces a multi-objective vector to a single scalar value.

        :param fitness: Fitness value(s) of an individual
        :type fitness: Union[float, np.ndarray]
        :return: Weighted scalar fitness
        :rtype: float
        """
        f_vector = np.atleast_1d(fitness)

        if len(f_vector) != len(self.weights):
            raise ValueError(
                f"Dimension mismatch: Fitness has {len(f_vector)} objectives, "
                f"but weights have {len(self.weights)}."
            )

        return float(np.dot(f_vector, self.weights))


class ParetoSelection(Selection):
    """Pareto-based selection operator for multi-objective optimization."""

    def select(self, population: Population, size: int) -> Population:
        """
        Search for non-dominated individuals and select them. If more individuals are needed
        to reach the desired size, it selects the non-dominated individual of the first ones
        not yet selected until the population size is met.

        :param population: Population to select from
        :type population: Population
        :param size: Number of individuals to select
        :type size: int
        :return: Selected population
        :rtype: Population
        """
        fitness_matrix = np.array([ind.fitness for ind in population])
        remaining_indices = list(range(len(population)))
        selected_indices = []

        while len(selected_indices) < size and remaining_indices:
            # 1. Find the current front
            front_indices = self._find_non_dominated_front(
                remaining_indices, fitness_matrix
            )

            # 2. Handle size (truncate if front is too large)
            if len(selected_indices) + len(front_indices) > size:
                needed = size - len(selected_indices)
                front_indices = front_indices[:needed]

            selected_indices.extend(front_indices)
            for idx in front_indices:
                remaining_indices.remove(idx)

        return Population(
            [population[i] for i in selected_indices],
            minimize=True,
            multiobjective=True,
        )

    def _find_non_dominated_front(
        self, indices: List[int], fitness_matrix: np.ndarray
    ) -> List[int]:
        """
        Find non-dominated individuals among the given indices.

        :param indices: Indices of individuals to consider
        :type indices: List[int]
        :param fitness_matrix: Fitness values of the population
        :type fitness_matrix: np.ndarray
        :return: Indices of non-dominated individuals
        :rtype: List[int]
        """
        front = []
        for i in indices:
            is_dominated = False
            for j in indices:
                if i != j and self._dominates(fitness_matrix[j], fitness_matrix[i]):
                    is_dominated = True
                    break
            if not is_dominated:
                front.append(i)
        return front

    def _dominates(self, fitness_a: np.ndarray, fitness_b: np.ndarray) -> bool:
        """
        Returns True if fitness_a dominates fitness_b (Minimization).
        A dominates B if A is better or equal in all objectives AND better in at least one.

        :param fitness_a: Fitness of individual A
        :type fitness_a: np.ndarray
        :param fitness_b: Fitness of individual B
        :type fitness_b: np.ndarray
        :return: True if A dominates B
        :rtype: bool
        """
        return np.all(fitness_a <= fitness_b) and np.any(fitness_a < fitness_b)
