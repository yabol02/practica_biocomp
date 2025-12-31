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

    def __call__(self, *args, **kwargs) -> Population:
        """Alias for `replace` method."""
        return self.replace(*args, **kwargs)

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


class ElitistReplacement(Replacement):
    """Elitist replacement: keep best from parents + offspring."""

    def __init__(self, elite_size: int = 1):
        """
        Initialize elitist replacement.

        :param elite_size: Number of best parents to preserve
        :type elite_size: int
        """
        self.elite_size = elite_size

    def replace(self, parents: Population, offspring: Population) -> Population:
        """
        Keep elite parents and fill rest with best offspring.

        :param parents: Parent population
        :type parents: Population
        :param offspring: Offspring population
        :type offspring: Population
        :return: New population with elites
        :rtype: Population
        """
        # Get elite from parents
        if self.elite_size == 1:
            elite = [parents.best_individual]
        else:
            elite = sorted(
                parents.individuals,
                key=lambda ind: ind.fitness if parents.minimize else -ind.fitness,
            )[: self.elite_size]

        # Get best offspring to fill remaining slots
        remaining_size = len(parents) - self.elite_size
        best_offspring = sorted(
            offspring.individuals,
            key=lambda ind: ind.fitness if offspring.minimize else -ind.fitness,
        )[:remaining_size]

        new_individuals = elite + best_offspring
        return Population(new_individuals, minimize=parents.minimize)


class MuPlusLambdaReplacement(Replacement):
    """(μ+λ) replacement: select best from parents + offspring."""

    def replace(self, parents: Population, offspring: Population) -> Population:
        """
        Select best individuals from combined parents and offspring.

        :param parents: Parent population (μ)
        :type parents: Population
        :param offspring: Offspring population (λ)
        :type offspring: Population
        :return: Best μ individuals
        :rtype: Population
        """
        combined = parents.individuals + offspring.individuals
        target_size = len(parents)

        best = sorted(
            combined, key=lambda ind: ind.fitness if parents.minimize else -ind.fitness
        )[:target_size]

        return Population(best, minimize=parents.minimize)
