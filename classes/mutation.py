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
