"""
Evo class for evolutionary genetic algorithms.
"""
from typing import List, Callable, Optional
import random
from .individual import Individual


class Evo:
    """
    Class that implements a generic evolutionary algorithm.
    
    Attributes:
        population: List of individuals in the population
        best_indiv: Best individual found
        crossover_function: Crossover function to use
    """
    
    def __init__(
        self,
        population: Optional[List[Individual]] = None,
        crossover_function: Optional[Callable] = None,
        population_size: Optional[int] = None,
        genotype_size: Optional[int] = None
    ):
        """
        Constructor for the Evo class.
        
        Args:
            population: Initial population of individuals. If None, it's generated randomly.
            crossover_function: Custom crossover function.
            population_size: Size of the population to generate if population is None.
            genotype_size: Size of the genotype for random individuals.
        """
        if population is not None:
            self.population = population
        elif population_size is not None and genotype_size is not None:
            # Generate random population
            self.population = [
                Individual(random_size=genotype_size) 
                for _ in range(population_size)
            ]
        else:
            self.population = []
        
        self.best_indiv: Optional[Individual] = None
        self.crossover_function = crossover_function if crossover_function else self._default_crossover
    
    def _default_crossover(self, parent1: Individual, parent2: Individual) -> tuple[Individual, Individual]:
        """
        Default crossover function (single-point crossover).
        
        Args:
            parent1: First parent
            parent2: Second parent
            
        Returns:
            Tuple with two offspring resulting from the crossover
        """
        if len(parent1.genotype) == 0 or len(parent2.genotype) == 0:
            return parent1, parent2
        
        crossover_point = random.randint(1, len(parent1.genotype) - 1)
        
        offspring1_genotype = parent1.genotype[:crossover_point] + parent2.genotype[crossover_point:]
        offspring2_genotype = parent2.genotype[:crossover_point] + parent1.genotype[crossover_point:]
        
        offspring1 = Individual(genotype=offspring1_genotype)
        offspring2 = Individual(genotype=offspring2_genotype)
        
        return offspring1, offspring2
    
    def crossover(self, parent1: Individual, parent2: Individual) -> tuple[Individual, Individual]:
        """
        Performs crossover between two parent individuals.
        
        Args:
            parent1: First parent
            parent2: Second parent
            
        Returns:
            Tuple with two offspring resulting from the crossover
        """
        return self.crossover_function(parent1, parent2)
    
    def mutation(self, individual: Individual, mutation_rate: float = 0.1, mutation_strength: float = 0.1) -> Individual:
        """
        Applies mutation to an individual.
        
        Args:
            individual: Individual to mutate
            mutation_rate: Probability of mutation for each gene
            mutation_strength: Magnitude of the mutation
            
        Returns:
            Mutated individual
        """
        mutated_genotype = []
        
        for gene in individual.genotype:
            if random.random() < mutation_rate:
                # Apply mutation
                if isinstance(gene, (int, float)):
                    gene = gene + random.gauss(0, mutation_strength)
                else:
                    # If not numeric, keep the original gene
                    pass
            mutated_genotype.append(gene)
        
        mutated_individual = Individual(genotype=mutated_genotype)
        return mutated_individual
    
    def update_best_individual(self) -> None:
        """
        Updates the best individual in the population based on fitness.
        """
        if not self.population:
            self.best_indiv = None
            return
        
        self.best_indiv = max(self.population, key=lambda ind: ind.fitness_value)
    
    def __repr__(self) -> str:
        """String representation of the Evo class."""
        return f"Evo(population_size={len(self.population)}, best_fitness={self.best_indiv.fitness_value if self.best_indiv else 'N/A'}"
