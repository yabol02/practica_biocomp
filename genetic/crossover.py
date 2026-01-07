import random
from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np

from .individual import Individual, RealIndividual
from .population import Population


class Crossover(ABC):
    """Abstract base class for crossover operators."""

    @abstractmethod
    def cross(self, population: Population) -> Population:
        """
        Perform crossover on population.

        :param population: Population to apply crossover
        :type population: Population
        :return: New population after crossover
        :rtype: Population
        """
        pass

    def __call__(self, *args, **kwargs) -> Population:
        """Alias for `cross` method."""
        return self.cross(*args, **kwargs)


class OrderCrossover(Crossover):
    """Single-point order crossover for permutation problems."""

    def cross(self, population: Population) -> Population:
        """
        Apply single-point order crossover to population pairs.

        :param population: Population to apply crossover
        :type population: Population
        :return: Population with offspring
        :rtype: Population
        """
        new_individuals: List[Individual] = []
        genome_length = len(population[0].genotype)

        for i in range(0, len(population), 2):
            parent1 = population[i]
            parent2 = population[i + 1] if i + 1 < len(population) else parent1

            cross_point = random.randint(1, genome_length - 1)

            child1, child2 = self._crossover_individuals(parent1, parent2, cross_point)
            new_individuals.extend([child1, child2])

        population = Population(
            new_individuals[: len(population)], minimize=population.minimize
        )
        return population

    def _crossover_individuals(
        self, parent1: Individual, parent2: Individual, cross_point: int
    ) -> Tuple[Individual, Individual]:
        """
        Perform single-point order crossover between two parents.

        :param parent1: First parent
        :type parent1: Individual
        :param parent2: Second parent
        :type parent2: Individual
        :param cross_point: Crossover point
        :type cross_point: int
        :return: Two offspring individuals
        :rtype: Tuple[Individual, Individual]
        """

        def build_genotype(p1_gen, p2_gen, point):
            child_gen = np.full_like(p1_gen, -1)
            child_gen[:point] = p1_gen[:point]
            not_used_genes = np.setdiff1d(p2_gen, child_gen)
            child_gen[point:] = not_used_genes
            return child_gen

        g1, g2 = parent1.genotype, parent2.genotype

        child1_gen = build_genotype(g1, g2, cross_point)
        child2_gen = build_genotype(g2, g1, cross_point)

        child_class = parent1.__class__
        bounds = parent1.bounds
        return child_class(genotype=child1_gen, bounds=bounds), child_class(
            genotype=child2_gen, bounds=bounds
        )


class BlendCrossover(Crossover):
    """Blend crossover (BLX-α) for real-valued genes."""

    def __init__(self, alpha: float = 0.5):
        """
        Initialize blend crossover.

        :param alpha: Extension factor beyond parent range
        :type alpha: float
        """
        self.alpha = alpha

    def cross(self, population: Population) -> Population:
        """
        Apply blend crossover to population pairs.

        :param population: Population to apply crossover
        :type population: Population
        :return: Population with offspring
        :rtype: Population
        """
        new_individuals = []
        for i in range(0, len(population), 2):
            parent1 = population[i]
            parent2 = population[i + 1] if i + 1 < len(population) else population[i]
            child1, child2 = self._blend_individuals(parent1, parent2)
            new_individuals.extend([child1, child2])
        population = Population(new_individuals, minimize=population.minimize)
        return population

    def _blend_individuals(
        self, parent1: Individual, parent2: Individual
    ) -> Tuple[Individual, Individual]:
        """
        Create offspring using blend crossover.

        :param parent1: First parent
        :type parent1: Individual
        :param parent2: Second parent
        :type parent2: Individual
        :return: Two offspring
        :rtype: Tuple[Individual, Individual]
        """
        child1_genotype = []
        child2_genotype = []

        for g1, g2 in zip(parent1.genotype, parent2.genotype):
            min_val = min(g1, g2)
            max_val = max(g1, g2)
            range_val = max_val - min_val

            low = min_val - self.alpha * range_val
            high = max_val + self.alpha * range_val

            child1_genotype.append(random.uniform(low, high))
            child2_genotype.append(random.uniform(low, high))

        return RealIndividual(genotype=child1_genotype), RealIndividual(
            genotype=child2_genotype
        )
class PMXCrossover(Crossover):
    """
    Partially Mapped Crossover (PMX) para permutaciones TSP.
    
    Copia un segmento de ciudades de un padre y reubica las restantes según la correspondencia del otro padre
    Es una versión más adecuada que nuestro Blend Crossover ajustado a TSP

    Fuente: https://en.wikipedia.org/wiki/Crossover_(evolutionary_algorithm) <- Hay una seccion para PMX

    Pasos:
    1. Seleccionar dos puntos de corte i < j.
    2. Copiar directamente al hijo C el segmento A[i..j].
    3. Para cada gen g en el segmento B[i..j]:
        - Si g ya está presente en C, no se hace nada.
        - Si g NO está en C:
            a) Sea pos_B(g) la posición de g en B.
            b) Sea h = A[pos_B(g)] el gen correspondiente en A.
            c) Si la posición pos_B(g) en C está libre, colocar g allí.
            d) Si está ocupada, repetir el proceso usando h como nuevo gen
               (aplicando el mapeo de forma recursiva) hasta encontrar
               una posición libre.
    4. Una vez resueltos todos los conflictos inducidos por el mapeo,
       rellenar las posiciones restantes de C con los genes de B
       manteniendo su orden relativo.
    """
    def cross(self, population: Population) -> Population:
        new_inds = []
        genome_length = len(population.individuals[0].genotype)
        
        for i in range(0, len(population), 2):
            parent1 = population.individuals[i]
            parent2 = population.individuals[min(i+1, len(population)-1)]
            
            p1_gen = np.array(parent1.genotype) if not isinstance(parent1.genotype, np.ndarray) else parent1.genotype
            p2_gen = np.array(parent2.genotype) if not isinstance(parent2.genotype, np.ndarray) else parent2.genotype
            
            # 1. Seleccionar puntos de corte
            idx1, idx2 = sorted(random.sample(range(genome_length), 2))
            
            child1_gen = np.full(genome_length, -1)
            child2_gen = np.full(genome_length, -1)

            # 2. Copiar segmento central
            child1_gen[idx1:idx2+1] = p1_gen[idx1:idx2+1]
            child2_gen[idx1:idx2+1] = p2_gen[idx1:idx2+1]

            p1_list = p1_gen.tolist()
            p2_list = p2_gen.tolist()

            # 3. Mapeo de segmentos
            for k in range(idx1, idx2+1):
                g1 = p1_gen[k]
                g2 = p2_gen[k]
                
                # Para hijo 1: colocar g2 de parent2
                if g2 not in child1_gen:
                    pos = k
                    while child1_gen[pos] != -1:
                        # Buscamos dónde está en p2 el gen que p1 tiene en esta pos
                        val_at_pos = p1_gen[pos]
                        pos = p2_list.index(val_at_pos)
                    child1_gen[pos] = g2
                
                # Para hijo 2: colocar g1 de parent1
                if g1 not in child2_gen:
                    pos = k
                    while child2_gen[pos] != -1:
                        val_at_pos = p2_gen[pos]
                        pos = p1_list.index(val_at_pos)
                    child2_gen[pos] = g1

            # 4. Llenar huecos restantes
            def fill_remaining(child, donor):
                for idx in range(genome_length):
                    if child[idx] == -1:
                        child[idx] = donor[idx]
            
            fill_remaining(child1_gen, p2_gen)
            fill_remaining(child2_gen, p1_gen)

            child_class = parent1.__class__
            new_inds.append(child_class(genotype=child1_gen, bounds=parent1.bounds))
            new_inds.append(child_class(genotype=child2_gen, bounds=parent1.bounds))

        return Population(new_inds[:len(population)], minimize=population.minimize)
class CycleCrossover(Crossover):
    """
    Cycle Crossover (CX) para permutaciones TSP.
    
    Preserva posiciones absolutas de los genes.

    Fuente: https://www.youtube.com/watch?v=DJ-yBmEEkgA

    1. Se copia en C (hijo) el primer elemento (g1) del padre A
    2. Se busca la posición de g1 en el padre B (el gen que ocupa esa posicion vamos a llamarlo g2)
    3. Se añade el gen g2 al hijo y con la posición que g2 ocupa en el padre se repite el paso 2 hasta que volvamos a encontrar a g1
    4. Marcar todas las posiciones del ciclo como completadas.
    5. Repetir el proceso desde otra posición no asignada, alternando el padre de referencia, hasta completar todo el genoma.

    """
    def cross(self, population: Population) -> Population:
        new_inds = []
        n = len(population.individuals[0].genotype)

        for i in range(0, len(population), 2):
            p1 = population.individuals[i]
            p2 = population.individuals[min(i + 1, len(population) - 1)]

            p1 = np.array(p1.genotype) if not isinstance(p1.genotype, np.ndarray) else p1.genotype
            p2 = np.array(p2.genotype) if not isinstance(p2.genotype, np.ndarray) else p2.genotype

            #p1 = p1.to_list()
            #p2 = p2.to_list()

            c1 = np.full(n, -1)
            c2 = np.full(n, -1)

            visited = [False] * n
            use_parent1 = True  # alternar por ciclo

            for start in range(n):
                if visited[start]:
                    continue

                idx = start
                cycle = []

                while not visited[idx]:
                    visited[idx] = True
                    cycle.append(idx)
                    val = p2[idx]
                    idx = np.where(p1 == val)[0][0]

                # Copiar ciclo completo
                for pos in cycle:
                    if use_parent1:
                        c1[pos] = p1[pos]
                        c2[pos] = p2[pos]
                    else:
                        c1[pos] = p2[pos]
                        c2[pos] = p1[pos]

                use_parent1 = not use_parent1  # alternar padre

            child_class = population.individuals[i].__class__
            bounds = population.individuals[i].bounds

            new_inds.append(child_class(genotype=c1, bounds=bounds))
            new_inds.append(child_class(genotype=c2, bounds=bounds))

        return Population(new_inds[:len(population)], minimize=population.minimize)

class EdgeRecombinationCrossover(Crossover):
    """
    Edge Recombination Crossover (ERX) para TSP.

    Preserva el mayor número posible de aristas (adyacencias entre ciudades) presentes en los padres.
    
    Fuente: https://content.wolfram.com/sites/13/2018/02/13-4-1.pdf (y otras más rollo tutorial)

    Pasos:
    - Cada ciudad mantiene una lista de vecinos observados en ambos padres.
    - El hijo se construye incrementalmente eligiendo ciudades con menor grado (menos opciones restantes), favoreciendo aristas comunes.

    1. Construir una tabla de adyacencias:
        Para cada ciudad, registrar sus vecinos izquierdo y derecho
        en ambos padres (hasta 4 vecinos posibles).
    2. Seleccionar aleatoriamente una ciudad inicial y añadirla a C.
    3. Eliminar la ciudad actual de todas las listas de adyacencia.
    4. Elegir como siguiente ciudad:
        a) Si la ciudad actual tiene vecinos disponibles,
           elegir el vecino con la lista de adyacencias más corta
           (heurística de menor grado).
        b) Si no tiene vecinos disponibles,
           elegir aleatoriamente una ciudad aún no utilizada.
    5. Repetir los pasos 3-4 hasta completar el tour.
    """
    def cross(self, population: Population) -> Population:
        def build_edge_map(p1, p2):
            edge_map = {}
            n = len(p1)
            for i in range(n):
                city = p1[i]
                neighbors = set()
                neighbors.add(p1[(i-1)%n]); neighbors.add(p1[(i+1)%n])
                j = np.where(p2 == city)[0][0]
                neighbors.add(p2[(j-1)%n]); neighbors.add(p2[(j+1)%n])
                edge_map[city] = neighbors
            return edge_map

        def build_child(p1, p2):
            n = len(p1)
            edge_map = build_edge_map(p1, p2)
            # Empezar con ciudad aleatoria
            current = random.choice(p1.tolist())
            child = [current]
            while len(child) < n:
                # Eliminar current de las listas de vecinos
                for neighbors in edge_map.values():
                    neighbors.discard(current)
                # Elegir siguiente ciudad
                if edge_map[current]:
                    # escoger vecino con menor lista
                    nxt = min(edge_map[current], key=lambda x: len(edge_map[x]))
                else:
                    # si está vacío, elegir aleatoriamente de los no usados
                    candidates = [c for c in p1 if c not in child]
                    nxt = random.choice(candidates)
                child.append(nxt)
                current = nxt
            return np.array(child)

        new_inds = []
        for i in range(0, len(population), 2):
            p1 = population.individuals[i]
            p2 = population.individuals[min(i+1, len(population)-1)]

            parent1 = np.array(p1.genotype) if not isinstance(p1.genotype, np.ndarray) else p1.genotype
            parent2 = np.array(p2.genotype) if not isinstance(p2.genotype, np.ndarray) else p2.genotype

            # parent1 = p1.to_list()
            # parent2 = p2.to_list()

            child1_gen = build_child(parent1, parent2)
            child2_gen = build_child(parent2, parent1)
            child_class = population.individuals[i].__class__
            bounds = population.individuals[i].bounds
            new_inds.append(child_class(genotype=child1_gen, bounds=bounds))
            new_inds.append(child_class(genotype=child2_gen, bounds=bounds))
        return Population(new_inds[: len(population)], minimize=population.minimize)
    
    # TODO: Podemos implementar EAX (Edge Assembly Crossover) pero es complicado (aunk bastante estado del arte)