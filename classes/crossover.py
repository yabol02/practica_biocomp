import random
from abc import ABC, abstractmethod
from typing import List, Tuple

from .individual import Individual, RealIndividual
from .population import Population


class Crossover(ABC):
    """Abstract base class for crossover operators."""

    @abstractmethod
    def cross(self, population: Population) -> Population:
        """
        Perform crossover on population.

        :param population: Population to apply crossover
        :type population: Population
        :return: New population after crossover
        :rtype: Population
        """
        pass

    def __call__(self, *args, **kwargs) -> Population:
        """Alias for `cross` method."""
        return self.cross(*args, **kwargs)


class BasicTSPCrossover(Crossover):
    """Order crossover for TSP problems."""

    def cross(self, population: Population) -> Population:
        """
        Apply order crossover to population pairs.

        :param population: Population to apply crossover
        :type population: Population
        :return: Population with offspring
        :rtype: Population
        """
        new_individuals: List[Individual] = []
        genome_length = len(population[0].genotype)

        for i in range(0, len(population), 2):
            parent1 = population[i]
            parent2 = population[i + 1] if i + 1 < len(population) else population[i]

            cross_point = random.randint(1, genome_length - 1)
            child1, child2 = self._crossover_individuals(parent1, parent2, cross_point)

            new_individuals.extend([child1, child2])

        population.individuals = new_individuals
        return population

    def _crossover_individuals(
        self, parent1: Individual, parent2: Individual, cross_point: int
    ) -> Tuple[Individual, Individual]:
        """
        Perform order crossover between two parents.

        :param parent1: First parent
        :type parent1: Individual
        :param parent2: Second parent
        :type parent2: Individual
        :param cross_point: Crossover point
        :type cross_point: int
        :return: Two offspring individuals
        :rtype: Tuple[Individual, Individual]
        """

        def fill_genotype(child_genotype: List, parent_genotype: List) -> List:
            child_set = set(child_genotype)
            for gene in parent_genotype:
                if gene not in child_set:
                    child_genotype.append(gene)
            return child_genotype

        child1_genotype = parent1.genotype[:cross_point]
        child2_genotype = parent2.genotype[:cross_point]

        child1_genotype = fill_genotype(child1_genotype, parent2.genotype)
        child2_genotype = fill_genotype(child2_genotype, parent1.genotype)

        return Individual(genotype=child1_genotype), Individual(
            genotype=child2_genotype
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

    def cross(self, population: Population) -> Population:
        """
        Apply blend crossover to population pairs.

        :param population: Population to apply crossover
        :type population: Population
        :return: Population with offspring
        :rtype: Population
        """
        new_individuals = []
        for i in range(0, len(population), 2):
            parent1 = population[i]
            parent2 = population[i + 1] if i + 1 < len(population) else population[i]
            child1, child2 = self._blend_individuals(parent1, parent2)
            new_individuals.extend([child1, child2])
        population.individuals = new_individuals
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

        for g1, g2 in zip(parent1.genotype, parent2.genotype):
            min_val = min(g1, g2)
            max_val = max(g1, g2)
            range_val = max_val - min_val

            low = min_val - self.alpha * range_val
            high = max_val + self.alpha * range_val

            child1_genotype.append(random.uniform(low, high))
            child2_genotype.append(random.uniform(low, high))

        return RealIndividual(genotype=child1_genotype), RealIndividual(
            genotype=child2_genotype
        )
