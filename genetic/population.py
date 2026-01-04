"""
Population class for genetic algorithms.
"""

from typing import Any, Callable, Iterator, List, Optional, Type

import numpy as np

from .individual import Individual


class Population:
    """Represents a population of individuals."""

    def __init__(
        self,
        individuals: Optional[List[Individual]] = None,
        minimize: bool = True,
        multiobjective: bool = False,
    ):
        """
        Initialize population.

        :param individuals: Initial list of individuals
        :type individuals: Optional[List[Individual]]
        :param minimize: True to minimize fitness, False to maximize. Default is True
        :type minimize: bool
        :param multiobjective: True if the problem is multi-objective. Default is False
        :type multiobjective: bool
        """
        self.individuals: List[Individual] = (
            individuals if individuals is not None else []
        )
        self.minimize: bool = minimize
        self._best_individual: Optional[Individual] = None
        self._is_sorted: bool = False
        self.multiobjective: bool = multiobjective

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

        if not self.multiobjective:
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

        if self.minimize:
            self._best_individual = min(self.individuals, key=lambda ind: ind.fitness)
        else:
            self._best_individual = max(self.individuals, key=lambda ind: ind.fitness)

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
        best = "None"
        if np.any(self.individuals) and self._best_individual is not None:
            val = self.best_individual.fitness
            best = (
                f"{val:.4f}"
                if isinstance(val, float)
                else np.array2string(val, precision=3)
            )
            if len(best) > 20:
                best = best[:15] + "..."
        return f"Population(size={len(self)}, mode={'min' if self.minimize else 'max'}, best={best})"

    def __str__(self) -> str:
        status = (
            "Evaluated"
            if all(i.is_evaluated for i in self.individuals)
            else "Unevaluated"
        )
        return f"<Population size={len(self)} ({status})>"
