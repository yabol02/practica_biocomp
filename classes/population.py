"""
Population class for genetic algorithms.
"""
from typing import Callable, List, Optional
from .individual import Individual


class Population():
    """
    Represents a population of individuals in a genetic algorithm.
    Inherits from list to hold Individual objects.
    
    Attributes:
        best_individual: The individual with the highest fitness in the population.
    """

    def __init__(self, individuals: List[Individual] = None, quality_function: Optional[Callable] = None, get_min = True):
        """
        Constructor for the Population class.

        Args:
            individuals: A list of Individual objects to initialize the population.
                         If None, an empty population is created.
        """
        self.quality_function = quality_function
        self.counter_quality = 0
        self.individuals = individuals
        self.get_min = get_min
        self.update_all_qualities()

    def update_all_qualities(self):
        [self.update_quality(i) for i in self.individuals]


    def update_quality(self, individual) -> None:
        """
        Calculates the firnss pf and individual
        """

        individual.fitness_value = self.quality_function(individual.genotype)
        self.counter_quality += 1


    def _update_best_individual(self) -> None:
        """
        Updates the best individual in the population based on fitness.
        This method should be called whenever the population or individual fitnesses change.
        """
        if self.get_min:
            self.best_individual = min(self.individuals, key=lambda ind: ind.fitness_value)
        else:
            self.best_individual = min(self.individuals, key=lambda ind: ind.fitness_value)

    def append(self, individual: Individual) -> None:
        """
        Appends an individual to the population and updates the best individual.

        Args:
            individual: The Individual object to append.
        """
        super().append(individual)
        self._update_best_individual()

    def extend(self, individuals: List[Individual]) -> None:
        """
        Extends the population with a list of individuals and updates the best individual.

        Args:
            individuals: A list of Individual objects to extend the population with.
        """
        super().extend(individuals)
        self._update_best_individual()

    def __setitem__(self, key, value):
        """
        Sets an item in the population and updates the best individual.
        """
        super().__setitem__(key, value)
        self._update_best_individual()

    def __delitem__(self, key):
        """
        Deletes an item from the population and updates the best individual.
        """
        pass
