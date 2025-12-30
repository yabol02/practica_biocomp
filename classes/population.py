"""
Population class for genetic algorithms.
"""

from typing import Any, Callable, Iterator, List, Optional, Type

import numpy as np

from .individual import Individual


class Population:
    """Represents a population of individuals."""

    def __init__(
        self, individuals: Optional[List[Individual]] = None, minimize: bool = True
    ):
        """
        Initialize population.

        :param individuals: Initial list of individuals
        :type individuals: Optional[List[Individual]]
        :param minimize: True to minimize fitness, False to maximize
        :type minimize: bool
        """
        self.individuals: List[Individual] = (
            individuals if individuals is not None else []
        )
        self.minimize: bool = minimize
        self._best_individual: Optional[Individual] = None
        self._is_sorted: bool = False

    def evaluate_population(self, fitness_function: Callable[[Any], float]) -> None:
        """
        Evaluate all unevaluated individuals.

        :param fitness_function: Function mapping genotype to fitness
        :type fitness_function: Callable[[Any], float]
        """
        unevaluated = [ind for ind in self.individuals if not ind.is_evaluated]

        if not unevaluated:
            return

        for ind in unevaluated:
            ind.fitness = fitness_function(ind.genotype)
        self._best_individual = self.best_individual
        self._is_sorted = False

    @property
    def best_individual(self) -> Individual:
        """
        Get best individual (lazy evaluation).

        :return: Best individual in population
        :rtype: Individual
        :raises ValueError: If population is empty
        """
        if not self.individuals:
            raise ValueError("Population is empty")

        if self._best_individual is None:
            self._best_individual = min(
                self.individuals,
                key=lambda ind: ind.fitness if self.minimize else -ind.fitness,
            )
        return self._best_individual

    @property
    def bounds(self) -> Optional[List]:
        """
        Get bounds of individuals in the population.

        :return: Bounds of individuals or None if population is empty
        :rtype: Optional[List]
        """
        if not self.individuals:
            return None
        return self.individuals[0].bounds

    @property
    def ind_class(self) -> Type[Individual]:
        """
        Get the class of individuals in this population.

        :return: Class of individuals
        :rtype: type
        :raises ValueError: If population is empty
        """
        if not self.individuals:
            raise ValueError("No se puede determinar la clase de una población vacía.")
        return self.individuals[0].__class__

    def add(self, individual: Individual) -> None:
        """
        Add individual to population.

        :param individual: Individual to add
        :type individual: Individual
        """
        self.individuals.append(individual)
        self._best_individual = None

    def extend(self, new_individuals: List[Individual]) -> None:
        """
        Add multiple individuals to population.

        :param new_individuals: Individuals to add
        :type new_individuals: List[Individual]
        """
        self.individuals.extend(new_individuals)
        self._best_individual = None

    def get_top_n(self, n: int) -> List[Individual]:
        """
        Get top N individuals sorted by fitness.

        :param n: Number of top individuals
        :type n: int
        :return: Top N individuals
        :rtype: List[Individual]
        """
        return sorted(
            self.individuals, key=lambda ind: ind.fitness, reverse=not self.minimize
        )[:n]

    def __iter__(self) -> Iterator[Individual]:
        return iter(self.individuals)

    def __len__(self) -> int:
        return len(self.individuals)

    def __getitem__(self, key: int) -> Individual:
        return self.individuals[key]

    def __repr__(self) -> str:
        return (
            f"Population(size={len(self)}, minimize={self.minimize}, "
            f"best_fitness={self.best_individual.fitness if self._best_individual else 'Unevaluated'})"
        )
