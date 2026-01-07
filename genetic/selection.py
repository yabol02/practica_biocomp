import random
from abc import ABC, abstractmethod

from .individual import RealIndividual
from .population import Population


class Selection(ABC):
    """Abstract base class for selection operators."""

    @abstractmethod
    def select(self, population: Population, size: int) -> Population:
        """
        Select individuals from population.

        :param population: Population to select from
        :type population: Population
        :param size: Number of individuals to select
        :type size: int
        :return: Selected population
        :rtype: Population
        """
        pass

    def __call__(self, *args, **kwargs) -> Population:
        """Alias for `select` method."""
        return self.select(*args, **kwargs)


class TournamentSelection(Selection):
    """Tournament selection operator."""

    def __init__(self, tournament_size: int = 3):
        """
        Initialize tournament selection.

        :param tournament_size: Number of individuals per tournament
        :type tournament_size: int
        """
        self.tournament_size = tournament_size

    def select(self, population: Population, size: int) -> Population:
        """
        Select individuals via tournament.

        :param population: Population to select from
        :type population: Population
        :param size: Number of individuals to select
        :type size: int
        :return: Selected population
        :rtype: Population
        """
        selected = []
        class_ind = population.ind_class
        bounds = population.bounds

        for _ in range(size):
            tournament = random.sample(population.individuals, self.tournament_size)
            winner = min(
                tournament,
                key=lambda ind: ind.fitness if population.minimize else -ind.fitness,
            )
            selected.append(class_ind(genotype=winner.genotype.copy(), bounds=bounds))

        return Population(selected, minimize=population.minimize)
