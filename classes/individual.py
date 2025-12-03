"""
Individual class for genetic algorithms.
"""
from typing import List, Optional


class Individual:
    """
    Represents an individual in a genetic algorithm.
    
    Attributes:
        genotype: List representing the individual's genotype
        fitness_value: Fitness value of the individual
    """
    
    def __init__(self, genotype: Optional[List] = None):
        """
        Constructor for the individual.
        
        Args:
            genotype: Initial genotype of the individual. If None, it's generated randomly.
            random_size: Size of the random genotype to generate if genotype is None.
        """
        if genotype is not None:
            self.genotype = genotype
        else:
            self.genotype = []
        
        self.fitness_value: float = 0.0
    
    def __repr__(self) -> str:
        """String representation of the individual."""
        return f"Individuo(genotype={self.genotype}, fitness={self.fitness_value:.4f})"
    
    def __str__(self) -> str:
        """Human-readable string of the individual."""
        return f"Individual with fitness: {self.fitness_value:.4f}"
