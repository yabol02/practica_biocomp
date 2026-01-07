import random
from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np

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

    def __call__(self, *args, **kwargs) -> Population:
        """Alias for `mutate` method."""
        return self.mutate(*args, **kwargs)

    def __repr__(self) -> str:
        attrs = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"{self.__class__.__name__}({attrs})"

    def __str__(self) -> str:
        return f"<{self.__class__.__name__}>"


class UniformMutation(Mutation):
    """Uniform mutation for real-valued individuals."""

    def __init__(self, mutation_rate: float, bounds: List[Tuple[float, float]]):
        """
        Initialize uniform mutation.

        :param mutation_rate: Probability of mutating each gene
        :type mutation_rate: float
        :param bounds: Min/max bounds for each gene [(min, max), ...]
        :type bounds: list[tuple[float, float]]
        """
        self.mutation_rate = mutation_rate
        self.bounds = np.asanyarray(bounds)

    def mutate(self, population: Population) -> Population:
        """
        Apply uniform mutation to population.

        :param population: Population to mutate
        :type population: Population
        :return: Mutated population
        :rtype: Population
        """
        if not population.individuals:
            return population

        n_individuals = len(population)
        n_genes = len(self.bounds)
        ind_class = population.ind_class
        bounds = population.bounds

        matrix = np.array([ind.genotype for ind in population])
        mutation_mask = np.random.random((n_individuals, n_genes)) < self.mutation_rate
        lows = self.bounds[:, 0]
        highs = self.bounds[:, 1]
        random_values = np.random.uniform(lows, highs, size=(n_individuals, n_genes))
        matrix[mutation_mask] = random_values[mutation_mask]

        any_mutation = mutation_mask.any(axis=1)
        new_individuals = []
        for i, mutated in enumerate(any_mutation):
            if mutated:
                new_ind = ind_class(genotype=matrix[i].tolist(), bounds=bounds)
                new_individuals.append(new_ind)
            else:
                new_individuals.append(population[i])

        population = Population(
            new_individuals,
            minimize=population.minimize,
            multiobjective=population.multiobjective,
        )
        return population


class SwapMutation(Mutation):
    """Swap mutation for permutation-based individuals."""

    def __init__(self, mutation_rate: float):
        """
        Initialize swap mutation.

        :param mutation_rate: Probability of an individual undergoing mutation.
        :type mutation_rate: float
        """
        self.mutation_rate = mutation_rate

    def mutate(self, population: Population) -> Population:
        """
        Apply swap mutation to population.

        :param population: Population to mutate
        :type population: Population
        :return: Mutated population
        :rtype: Population
        """
        if not population.individuals:
            return population

        n_individuals = len(population)
        ind_class = population.ind_class
        bounds = population.bounds

        to_mutate = np.random.random(n_individuals) < self.mutation_rate

        new_individuals = []
        for i, mutated in enumerate(to_mutate):
            if mutated:
                genotype = list(population[i].genotype)
                n_genes = len(genotype)

                idx1, idx2 = random.sample(range(n_genes), 2)
                genotype[idx1], genotype[idx2] = genotype[idx2], genotype[idx1]

                new_ind = ind_class(genotype=genotype, bounds=bounds)
                new_individuals.append(new_ind)
            else:
                new_individuals.append(population[i])

        population = Population(
            new_individuals,
            minimize=population.minimize,
            multiobjective=population.multiobjective,
        )
        return population
class InversionMutation(Mutation):
    """
    Inversion Mutation (equivalente a una operación 2-opt).

    Selecciona dos posiciones i < j e invierte el segmento intermedio.
    Esta mutación elimina cruces y mejora la estructura geométrica del tour.

    Propiedades:
    - Muy adecuada para TSP métrico.
    - Preserva subrutas contiguas.
    - Mucho más efectiva que swap para reducir distancia total.
    """
    def __init__(self, mutation_rate: float):
        self.mutation_rate = mutation_rate

    def mutate(self, population: Population) -> Population:
        new_inds = []
        for ind in population.individuals:
            if random.random() < self.mutation_rate:
                genotype = ind.genotype.copy().tolist()
                i, j = sorted(random.sample(range(len(genotype)), 2))
                genotype[i:j+1] = reversed(genotype[i:j+1])
                new_inds.append(ind.__class__(genotype=genotype, bounds=ind.bounds))
            else:
                new_inds.append(ind)
        return Population(new_inds, minimize=population.minimize)

class ScrambleMutation(Mutation):
    """
    Scramble Mutation para TSP.

    Selecciona un subsegmento del tour y baraja aleatoriamente
    el orden de las ciudades dentro del segmento.

    Propiedades:
    - Introduce diversidad fuerte.
    - Puede destruir subrutas buenas.
    - Útil como operador exploratorio ocasional.
    """
    def __init__(self, mutation_rate: float):
        self.mutation_rate = mutation_rate

    def mutate(self, population: Population) -> Population:
        new_inds = []
        for ind in population.individuals:
            if random.random() < self.mutation_rate:
                genotype = ind.genotype.copy().tolist()
                i, j = sorted(random.sample(range(len(genotype)), 2))
                segment = genotype[i:j+1]
                random.shuffle(segment)
                genotype[i:j+1] = segment
                new_inds.append(ind.__class__(genotype=genotype, bounds=ind.bounds))
            else:
                new_inds.append(ind)
        return Population(new_inds, minimize=population.minimize)

class TwoOptSearchMutation(Mutation):
    """
    Aplicar una mejora 2-opt sencilla a cada individuo.
    
    
    [!] Esto lo vuelve un algoritmo memético (Habría que aumentar el contador de evaluaciones para usarlo):    

    Pasos:
    - Explora todo el vecindario 2-opt del tour.
    - Evalúa sistemáticamente todos los pares (i, j).
    - Aplica una inversión solo si reduce la distancia.
    - Repite hasta que no exista ninguna mejora posible.
    """
    def __init__(self, mutation_rate: float = 1.0):
        self.mutation_rate = mutation_rate

    def mutate(self, population: Population) -> Population:
        new_inds = []
        for ind in population.individuals:
            if random.random() < self.mutation_rate:
                genotype = ind.genotype.copy().tolist()
                n = len(genotype)
                improved = True
                while improved:
                    improved = False
                    for i in range(n-1):
                        for j in range(i+1, n):
                            # Calcular cambio de costo
                            a, b = genotype[i], genotype[(i+1)%n]
                            c, d = genotype[j], genotype[(j+1)%n]
                            # Asumiendo distancia euclidiana o conocida en el individuo
                            delta = (ind.dist(a, c) + ind.dist(b, d)) - (ind.dist(a, b) + ind.dist(c, d))
                            if delta < 0:  # mejora
                                genotype[i+1:j+1] = reversed(genotype[i+1:j+1])
                                improved = True
                                break
                        if improved:
                            break
                new_inds.append(ind.__class__(genotype=genotype, bounds=ind.bounds))
            else:
                new_inds.append(ind)
        return Population(new_inds, minimize=population.minimize)