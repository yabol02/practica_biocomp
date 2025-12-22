from abc import ABC, abstractmethod

from .population import Population


class Replacement(ABC):
    """Abstract base class for replacement/survival strategies."""

    @abstractmethod
    def replace(self, parents: Population, offspring: Population) -> Population:
        """
        Decide new population from parents and offspring.

        :param parents: Current parent population
        :type parents: Population
        :param offspring: Offspring population
        :type offspring: Population
        :return: New population for next generation
        :rtype: Population
        """
        pass

    def __call__(self, **kwargs) -> Population:
        """Alias for `replace` method."""
        return self.replace(**kwargs)

    def __str__(self) -> str:
        return self.__class__.__name__

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class GenerationalReplacement(Replacement):
    """Generational replacement: offspring completely replace parents."""

    def replace(self, parents: Population, offspring: Population) -> Population:
        """
        Replace parents with offspring.

        :param parents: Parent population (ignored)
        :type parents: Population
        :param offspring: Offspring population
        :type offspring: Population
        :return: Offspring as new population
        :rtype: Population
        """
        return offspring
