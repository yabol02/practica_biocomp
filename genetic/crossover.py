import random
from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np

from .individual import Individual, RealIndividual
from .population import Population


class Crossover(ABC):
    """Abstract base class for crossover operators."""

    @abstractmethod
    def cross(self, population: Population, n_offspring: int) -> Population:
        """
        Perform crossover on population.

        :param population: Population to apply crossover
        :type population: Population
        :param n_offspring: Number of offspring to generate
        :type n_offspring: int
        :return: New population after crossover
        :rtype: Population
        """
        pass

    def __call__(self, *args, **kwargs) -> Population:
        """Alias for `cross` method."""
        return self.cross(*args, **kwargs)

    def __repr__(self) -> str:
        attrs = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"{self.__class__.__name__}({attrs})"

    def __str__(self) -> str:
        return f"<{self.__class__.__name__}>"


class OrderCrossover(Crossover):
    """Single-point order crossover for permutation problems."""

    def cross(self, population: Population, n_offspring: int) -> Population:
        """
        Apply single-point order crossover to population pairs.

        :param population: Population to apply crossover
        :type population: Population
        :param n_offspring: Number of offspring to generate
        :type n_offspring: int
        :return: Population with offspring
        :rtype: Population
        """

        new_individuals: List[Individual] = []
        genome_length = len(population[0].genotype)
        pop_size = len(population)

        while len(new_individuals) < n_offspring:
            indices = random.sample(range(pop_size), 2)
            parent1 = population[indices[0]]
            parent2 = population[indices[1]]

            cross_point = random.randint(1, genome_length - 1)

            child1, child2 = self._crossover_individuals(parent1, parent2, cross_point)
            new_individuals.append(child1)

            if len(new_individuals) < n_offspring:
                new_individuals.append(child2)

        population = Population(
            new_individuals[:n_offspring],
            minimize=population.minimize,
            multiobjective=population.multiobjective,
        )
        return population

    def _crossover_individuals(
        self, parent1: Individual, parent2: Individual, cross_point: int
    ) -> Tuple[Individual, Individual]:
        """
        Perform single-point order crossover between two parents.

        :param parent1: First parent
        :type parent1: Individual
        :param parent2: Second parent
        :type parent2: Individual
        :param cross_point: Crossover point
        :type cross_point: int
        :return: Two offspring individuals
        :rtype: Tuple[Individual, Individual]
        """

        def build_genotype(p1_gen, p2_gen, point):
            child_gen = np.full_like(p1_gen, -1)
            child_gen[:point] = p1_gen[:point]
            not_used_genes = np.setdiff1d(p2_gen, child_gen)
            child_gen[point:] = not_used_genes
            return child_gen

        g1, g2 = parent1.genotype, parent2.genotype

        child1_gen = build_genotype(g1, g2, cross_point)
        child2_gen = build_genotype(g2, g1, cross_point)

        child_class = parent1.__class__
        bounds = parent1.bounds
        return child_class(genotype=child1_gen, bounds=bounds), child_class(
            genotype=child2_gen, bounds=bounds
        )


class BlendCrossover(Crossover):
    """Blend crossover (BLX-α) for real-valued genes."""

    def __init__(self, alpha: float = 0.5):
        """
        Initialize blend crossover.

        :param alpha: Extension factor beyond parent range
        :type alpha: float
        """
        self.alpha = alpha

    def cross(self, population: Population, n_offspring: int) -> Population:
        """
        Apply blend crossover to population pairs.

        :param population: Population to apply crossover
        :type population: Population
        :param n_offspring: Number of offspring to generate
        :type n_offspring: int
        :return: Population with offspring
        :rtype: Population
        """

        new_individuals = []
        pop_size = len(population)

        while len(new_individuals) < n_offspring:
            indices = random.sample(range(pop_size), 2)
            parent1 = population[indices[0]]
            parent2 = population[indices[1]]

            child1, child2 = self._blend_individuals(parent1, parent2)
            new_individuals.append(child1)

            if len(new_individuals) < n_offspring:
                new_individuals.append(child2)

        population = Population(
            new_individuals[:n_offspring],
            minimize=population.minimize,
            multiobjective=population.multiobjective,
        )
        return population

    def _blend_individuals(
        self, parent1: Individual, parent2: Individual
    ) -> Tuple[Individual, Individual]:
        """
        Create offspring using blend crossover.

        :param parent1: First parent
        :type parent1: Individual
        :param parent2: Second parent
        :type parent2: Individual
        :return: Two offspring
        :rtype: Tuple[Individual, Individual]
        """
        child1_genotype = []
        child2_genotype = []

        bounds = parent1.bounds

        for i, (g1, g2) in enumerate(zip(parent1.genotype, parent2.genotype)):
            min_val = min(g1, g2)
            max_val = max(g1, g2)
            range_val = max_val - min_val

            low = min_val - self.alpha * range_val
            high = max_val + self.alpha * range_val

            gene_low_bound = bounds[i][0]
            gene_high_bound = bounds[i][1]

            low = max(low, gene_low_bound)
            high = min(high, gene_high_bound)

            child1_genotype.append(random.uniform(low, high))
            child2_genotype.append(random.uniform(low, high))

        return RealIndividual(
            genotype=child1_genotype, bounds=parent1.bounds
        ), RealIndividual(genotype=child2_genotype, bounds=parent2.bounds)
