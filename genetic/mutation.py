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
        """
        Initialize inversion mutation.

        :param mutation_rate: Probability of an individual undergoing mutation.
        :type mutation_rate: float
        """
        self.mutation_rate = mutation_rate

    def mutate(self, population: Population) -> Population:
        """
        Apply inversion mutation to population.

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

                idx1, idx2 = sorted(random.sample(range(n_genes), 2))
                genotype[idx1 : idx2 + 1] = reversed(genotype[idx1 : idx2 + 1])

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
        """
        Initialize scramble mutation.

        :param mutation_rate: Probability of an individual undergoing mutation.
        :type mutation_rate: float
        """
        self.mutation_rate = mutation_rate

    def mutate(self, population: Population) -> Population:
        """
        Apply scramble mutation to population.

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

                idx1, idx2 = sorted(random.sample(range(n_genes), 2))
                segment = genotype[idx1 : idx2 + 1]
                random.shuffle(segment)
                genotype[idx1 : idx2 + 1] = segment

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


class TwoOptSearchMutation(Mutation):
    """
    Aplicar una mejora 2-opt sencilla a cada individuo.

    [!] Esto lo vuelve un algoritmo memético (Habría que aumentar el contador de evaluaciones para usarlo):

    Pasos:
    - Explora todo el vecindario 2-opt del tour.
    - Evalúa sistemáticamente todos los pares (i, j).
    - Aplica una inversión solo si reduce la distancia.
    - Repite hasta que no exista ninguna mejora posible.

    Nota: Requiere que el individuo tenga una función fitness evaluable.
    Evalúa el fitness antes y después de cada inversión para determinar mejoras.
    """

    def __init__(self, mutation_rate: float = 1.0, distance_matrix: np.ndarray = None):
        """
        Initialize 2-opt search mutation.

        :param mutation_rate: Probability of an individual undergoing mutation.
        :type mutation_rate: float
        :param distance_matrix: Optional distance matrix for TSP. If None, uses fitness evaluation.
        :type distance_matrix: np.ndarray
        """
        self.mutation_rate = mutation_rate
        self.distance_matrix = distance_matrix

    def mutate(self, population: Population) -> Population:
        """
        Apply 2-opt search mutation to population.

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
                individual = population[i]
                genotype = list(individual.genotype)
                n_genes = len(genotype)

                improved = True
                while improved:
                    improved = False
                    for idx1 in range(n_genes - 1):
                        for idx2 in range(idx1 + 1, n_genes):
                            # Calcular cambio de distancia si tenemos matriz de distancias
                            if self.distance_matrix is not None:
                                improvement = self._calculate_improvement_with_matrix(
                                    genotype, idx1, idx2
                                )
                            else:
                                # Evaluar fitness antes y después
                                improvement = self._calculate_improvement_with_fitness(
                                    individual, genotype, idx1, idx2, ind_class, bounds
                                )

                            if improvement > 0:  # hay mejora
                                genotype[idx1 + 1 : idx2 + 1] = reversed(
                                    genotype[idx1 + 1 : idx2 + 1]
                                )
                                improved = True
                                break
                        if improved:
                            break

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

    def _calculate_improvement_with_matrix(
        self, genotype: List, idx1: int, idx2: int
    ) -> float:
        """
        Calculate improvement using distance matrix.

        :param genotype: Current genotype
        :type genotype: List
        :param idx1: First index
        :type idx1: int
        :param idx2: Second index
        :type idx2: int
        :return: Improvement value (positive if beneficial)
        :rtype: float
        """
        n = len(genotype)
        a = genotype[idx1]
        b = genotype[(idx1 + 1) % n]
        c = genotype[idx2]
        d = genotype[(idx2 + 1) % n]

        # Distancia actual: a-b + c-d
        # Distancia nueva: a-c + b-d
        current_dist = (
            self.distance_matrix[a, b] + self.distance_matrix[c, d]
        )
        new_dist = self.distance_matrix[a, c] + self.distance_matrix[b, d]

        return current_dist - new_dist

    def _calculate_improvement_with_fitness(
        self, individual, genotype: List, idx1: int, idx2: int, ind_class, bounds
    ) -> float:
        """
        Calculate improvement by evaluating fitness before and after inversion.

        :param individual: Current individual
        :param genotype: Current genotype
        :type genotype: List
        :param idx1: First index
        :type idx1: int
        :param idx2: Second index
        :type idx2: int
        :param ind_class: Individual class
        :param bounds: Bounds for individual
        :return: Improvement value (positive if beneficial)
        :rtype: float
        """
        # Obtener fitness actual
        current_fitness = individual.fitness if hasattr(individual, 'fitness') else None
        
        if current_fitness is None:
            # Si no hay fitness previo, no podemos determinar mejora
            return 0.0

        # Crear nuevo genotipo con inversión
        new_genotype = genotype.copy()
        new_genotype[idx1 + 1 : idx2 + 1] = reversed(new_genotype[idx1 + 1 : idx2 + 1])
        
        # Crear individuo temporal y evaluar
        temp_individual = ind_class(genotype=new_genotype, bounds=bounds)
        
        # Si el individuo no tiene fitness calculado, retornar 0 (sin mejora)
        if not hasattr(temp_individual, 'fitness') or temp_individual.fitness is None:
            return 0.0
        
        new_fitness = temp_individual.fitness

        # Para minimización: mejora = fitness_actual - fitness_nuevo
        # Si fitness_nuevo < fitness_actual -> mejora positiva
        return current_fitness - new_fitness


class CombinedMutation(Mutation):
    """
    Mutación combinada que aplica secuencialmente múltiples operadores de mutación.
    
    Útil para TSP multiobjetivo donde queremos:
    - InversionMutation para optimización local (2-opt)
    - SwapMutation para mantener diversidad
    """

    def __init__(self, mutations: List[Mutation]):
        """
        Initialize combined mutation.

        :param mutations: List of mutation operators to apply sequentially.
        :type mutations: List[Mutation]
        """
        self.mutations = mutations

    def mutate(self, population: Population) -> Population:
        """
        Apply all mutations sequentially to population.

        :param population: Population to mutate
        :type population: Population
        :return: Mutated population
        :rtype: Population
        """
        for mutation in self.mutations:
            population = mutation.mutate(population)
        return population