import random
from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np

from .population import Population


class Mutation(ABC):
    """Abstract base class for mutation operators."""

    @abstractmethod
    def mutate(self, population: Population) -> Population:
        """
        Apply mutation to population.

        :param population: Population to mutate
        :type population: Population
        :return: Mutated population
        :rtype: Population
        """
        pass

    def __call__(self, *args, **kwargs) -> Population:
        """Alias for `mutate` method."""
        return self.mutate(*args, **kwargs)


class UniformMutation(Mutation):
    """Uniform mutation for real-valued individuals."""

    def __init__(self, mutation_rate: float, bounds: List[Tuple[float, float]]):
        """
        Initialize uniform mutation.

        :param mutation_rate: Probability of mutating each gene
        :type mutation_rate: float
        :param bounds: Min/max bounds for each gene [(min, max), ...]
        :type bounds: list[tuple[float, float]]
        """
        self.mutation_rate = mutation_rate
        self.bounds = np.asanyarray(bounds)

    def mutate(self, population: Population) -> Population:
        """
        Apply uniform mutation to population.

        :param population: Population to mutate
        :type population: Population
        :return: Mutated population
        :rtype: Population
        """
        if not population.individuals:
            return population

        n_individuals = len(population)
        n_genes = len(self.bounds)
        ind_class = population.ind_class
        bounds = population.bounds

        matrix = np.array([ind.genotype for ind in population.individuals])
        mutation_mask = np.random.random((n_individuals, n_genes)) < self.mutation_rate
        lows = self.bounds[:, 0]
        highs = self.bounds[:, 1]
        random_values = np.random.uniform(lows, highs, size=(n_individuals, n_genes))
        matrix[mutation_mask] = random_values[mutation_mask]

        any_mutation = mutation_mask.any(axis=1)
        new_individuals = []
        for i, mutated in enumerate(any_mutation):
            if mutated:
                new_ind = ind_class(genotype=matrix[i].tolist(), bounds=bounds)
                new_individuals.append(new_ind)
            else:
                new_individuals.append(population.individuals[i])

        population = Population(new_individuals, minimize=population.minimize)
        return population


class SwapMutation(Mutation):
    """Swap mutation for permutation-based individuals."""

    def __init__(self, mutation_rate: float):
        """
        Initialize swap mutation.

        :param mutation_rate: Probability of an individual undergoing mutation.
        :type mutation_rate: float
        """
        self.mutation_rate = mutation_rate

    def mutate(self, population: Population) -> Population:
        """
        Apply swap mutation to population.

        :param population: Population to mutate
        :type population: Population
        :return: Mutated population
        :rtype: Population
        """
        if not population.individuals:
            return population

        n_individuals = len(population)
        ind_class = population.ind_class
        bounds = population.bounds

        to_mutate = np.random.random(n_individuals) < self.mutation_rate

        new_individuals = []
        for i, mutated in enumerate(to_mutate):
            if mutated:
                genotype = list(population.individuals[i].genotype)
                n_genes = len(genotype)

                idx1, idx2 = random.sample(range(n_genes), 2)
                genotype[idx1], genotype[idx2] = genotype[idx2], genotype[idx1]

                new_ind = ind_class(genotype=genotype, bounds=bounds)
                new_individuals.append(new_ind)
            else:
                new_individuals.append(population.individuals[i])

        population = Population(new_individuals, minimize=population.minimize)
        return population
