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
