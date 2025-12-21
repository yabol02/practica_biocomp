import random
from abc import ABC, abstractmethod
from typing import List, Tuple

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
        self.bounds = bounds

    def mutate(self, population: Population) -> Population:
        """
        Apply uniform mutation to population.

        :param population: Population to mutate
        :type population: Population
        :return: Mutated population
        :rtype: Population
        """
        for individual in population.individuals:
            for i in range(len(individual.genotype)):
                if random.random() < self.mutation_rate:
                    low, high = self.bounds[i]
                    individual.genotype[i] = random.uniform(low, high)
                    individual.is_evaluated = False
        return population
