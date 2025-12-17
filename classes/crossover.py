from abc import ABC, abstractmethod
import random

from .individual import Individual

class Crossover(ABC):
    @abstractmethod
    def cross(self, population):      
        """
        Perform crossover on the given population.
        """  
        pass

class BasicTSPCrossover(Crossover):
    def cross(self, population):
        """
        Given one parent, gets a random number from 0 to number of total genes.
        From 0 to that number, get genes from parent 1, then creates a set and the rest of the genotipe is covered by the mother.
        """
        new_individuals = []
        len_genome = len(population[0].genotype)
        for i in range(0, len(population), 2):
            parent1 = population[i]
            parent2 = population[i + 1]
            child1, child2 = self.crossover_individuals(parent1, parent2, cross_point=random.randint(1, len_genome - 1))
            new_individuals.append(child1)
            new_individuals.append(child2)
        population.indivisuals = new_individuals
        return population

    def crossover_individuals(self, parent1, parent2, cross_point):
        """
        Performs crossover between two individuals at the specified crossover point.
        """
        child1_genotype = parent1.genotype[:cross_point]
        child2_genotype = parent2.genotype[:cross_point]

        def fill_genotype(child_genotype, parent_genotype):
            child_set = set(child_genotype)
            for gene in parent_genotype:
                if gene not in child_set:
                    child_genotype.append(gene)
            return child_genotype

        child1_genotype = fill_genotype(child1_genotype, parent2.genotype)
        child2_genotype = fill_genotype(child2_genotype, parent1.genotype)

        child1 = Individual(genotype=child1_genotype)
        child2 = Individual(genotype=child2_genotype)

        return child1, child2

            
            
    