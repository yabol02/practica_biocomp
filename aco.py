"""
Módulo de Optimización por Colonia de Hormigas (ACO) para el problema del Viajante (TSP).

REFFACTORIZACIÓN Y MEJORAS:

1. Mejoras Algorítmicas (ACS & Elite Strategy):
   - Implementación de la regla de transición pseudo-aleatoria proporcional (parámetro q0).
     Esto permite a las hormigas alternar entre "explotación" (elegir el mejor camino conocido)
     y "exploración" (probar caminos nuevos según probabilidad).
   - Estrategia de actualización de feromonas "Elite": Solo las mejores hormigas de cada
     iteración depositan feromonas, reduciendo el ruido y acelerando la convergencia.
   - Mecanismo de recuperación de estancamiento: Reinicio de feromonas si la solución
     no mejora tras N iteraciones.

2. Búsqueda Local (2-opt):
   - Se ha integrado el algoritmo de optimización local "2-opt" al finalizar el tour de
     cada hormiga. Esto desenreda cruces en las rutas y mejora drásticamente la calidad
     de la solución final antes de actualizar las feromonas.

3. Optimización de Rendimiento y Estructura de Datos:
   - Normalización de aristas: Uso de `_make_edge` para garantizar que (A, B) y (B, A)
     sean la misma clave en el mapa, evitando duplicidad y búsquedas inversas.
   - Heurística de inicialización: El mapa de feromonas se inicializa basándose en una
     solución rápida de "Vecino más cercano" (Nearest Neighbor), en lugar de valores arbitrarios.
   - Reducción de complejidad: La selección de nodos ahora itera solo sobre los no visitados,
     en lugar de iterar sobre todas las rutas posibles del mapa global.

4. Concurrencia y Calidad de Código:
   - Thread Safety: Introducción de `threading.Lock` para proteger la lectura y escritura
     compartida del mapa de feromonas, eliminando condiciones de carrera (race conditions).
   - Type Hinting: Añadido tipado estático completo para mejor legibilidad y soporte en IDEs.
   - PEP 8: Normalización de nombres de variables (snake_case) y limpieza de atributos de clase.

Original Author: @zro404
Refactored by: @raguirregabiria, @yabol02, @aestoquera
"""

import math
import random
from threading import Lock, Thread
from typing import Callable, Dict, List, Optional, Tuple


class Ant(Thread):
    def __init__(
        self,
        nodes: List[Tuple[float, float]],
        pheromone_map: Dict,
        lock: Lock,
        start: Tuple[float, float],
        distance_callback: Callable,
        alpha: float,
        beta: float,
        q0: float,
        agent_index: int,
        seed: int = None,
    ):
        Thread.__init__(self)
        self.nodes = nodes
        self.pheromone_map = pheromone_map
        self.lock = lock
        self.alpha = alpha
        self.beta = beta
        self.q0 = q0
        self.initial_node = start
        self.agent_index = agent_index
        self.distance = distance_callback

        self.trip = []
        self.trip_distance = 0
        self.current_node = None

        if seed is not None:
            random.seed(seed + agent_index)

    def run(self):
        self.trip = [self.initial_node]
        self.trip_distance = 0
        self.current_node = self.initial_node

        # Construir tour completo
        while len(self.trip) < len(self.nodes):
            next_node = self.choose_next()
            if next_node is None:
                break

            edge = self._make_edge(self.current_node, next_node)
            self.trip_distance += self.distance(edge)
            self.trip.append(next_node)
            self.current_node = next_node

        # Cerrar el tour
        if len(self.trip) == len(self.nodes):
            edge = self._make_edge(self.current_node, self.initial_node)
            self.trip_distance += self.distance(edge)
            self.trip.append(self.initial_node)

            # Aplicar búsqueda local 2-opt
            self._two_opt_optimization()

    def choose_next(self) -> Tuple[float, float]:
        unvisited = [node for node in self.nodes if node not in self.trip]

        if not unvisited:
            return None

        # ACS: pseudo-random proportional rule
        q = random.random()

        if q < self.q0:
            # Explotación: elegir el mejor
            return self._choose_best(unvisited)
        else:
            # Exploración: elegir probabilísticamente
            return self._choose_probabilistic(unvisited)

    def _choose_best(self, unvisited: List) -> Tuple[float, float]:
        best_node = None
        best_value = -1

        with self.lock:
            for node in unvisited:
                edge = self._make_edge(self.current_node, node)
                pheromone = self.pheromone_map.get(edge, 1e-10)
                distance = self.distance(edge)

                value = (pheromone**self.alpha) * ((1.0 / distance) ** self.beta)

                if value > best_value:
                    best_value = value
                    best_node = node

        return best_node

    def _choose_probabilistic(self, unvisited: List) -> Tuple[float, float]:
        probabilities = []
        total = 0

        with self.lock:
            for node in unvisited:
                edge = self._make_edge(self.current_node, node)
                pheromone = self.pheromone_map.get(edge, 1e-10)
                distance = self.distance(edge)

                prob = (pheromone**self.alpha) * ((1.0 / distance) ** self.beta)
                probabilities.append(prob)
                total += prob

        if total == 0:
            return random.choice(unvisited)

        # Normalizar y seleccionar
        probabilities = [p / total for p in probabilities]

        r = random.random()
        cumulative = 0
        for i, prob in enumerate(probabilities):
            cumulative += prob
            if r <= cumulative:
                return unvisited[i]

        return unvisited[-1]

    def _two_opt_optimization(self):
        """Optimización local 2-opt para mejorar el tour"""
        improved = True
        max_iterations = 100
        iteration = 0

        while improved and iteration < max_iterations:
            improved = False
            iteration += 1

            for i in range(1, len(self.trip) - 2):
                for j in range(i + 1, len(self.trip) - 1):
                    # Calcular distancia actual
                    edge1 = self._make_edge(self.trip[i - 1], self.trip[i])
                    edge2 = self._make_edge(self.trip[j], self.trip[j + 1])
                    current_dist = self.distance(edge1) + self.distance(edge2)

                    # Calcular distancia con swap
                    new_edge1 = self._make_edge(self.trip[i - 1], self.trip[j])
                    new_edge2 = self._make_edge(self.trip[i], self.trip[j + 1])
                    new_dist = self.distance(new_edge1) + self.distance(new_edge2)

                    if new_dist < current_dist:
                        # Realizar el swap
                        self.trip[i : j + 1] = reversed(self.trip[i : j + 1])
                        improved = True

        # Recalcular distancia total
        self.trip_distance = 0
        for i in range(len(self.trip) - 1):
            edge = self._make_edge(self.trip[i], self.trip[i + 1])
            self.trip_distance += self.distance(edge)

    def _make_edge(self, node1, node2):
        """Crear edge normalizado (siempre en el mismo orden)"""
        if node1 < node2:
            return (node1, node2)
        return (node2, node1)


class AntColony:
    def __init__(
        self,
        nodes: List[Tuple[float, float]],
        ant_count: int = 50,
        alpha: float = 1.0,
        beta: float = 5.0,
        rho: float = 0.1,
        q0: float = 0.7,
        iterations: int = 200,
        elite_count: int = 3,
        stagnation_limit: int = 20,
        random_state: Optional[int] = None,
    ):
        self.nodes = nodes
        self.rho = rho
        self.iterations = iterations
        self.alpha = alpha
        self.beta = beta
        self.q0 = q0
        self.ant_count = ant_count
        self.elite_count = elite_count
        self.stagnation_limit = stagnation_limit

        self.best_path = []
        self.best_distance = float("inf")
        self.distance_history = []
        self.stagnation_counter = 0

        self.lock = Lock()
        self.pheromone_map = {}

        self.seed = random_state
        if random_state is not None:
            random.seed(random_state)

        # Calcular distancia inicial heurística
        nn_distance = self._nearest_neighbor_heuristic()

        # Inicializar feromonas
        self._init_pheromone_map(nn_distance)

        # Ejecutar optimización
        self._optimize()

    def _nearest_neighbor_heuristic(self) -> float:
        """Heurística de vecino más cercano para inicialización"""
        start = random.choice(self.nodes)
        unvisited = set(self.nodes)
        current = start
        unvisited.remove(current)
        distance = 0

        while unvisited:
            nearest = min(
                unvisited, key=lambda x: self.distance(self._make_edge(current, x))
            )
            distance += self.distance(self._make_edge(current, nearest))
            current = nearest
            unvisited.remove(current)

        distance += self.distance(self._make_edge(current, start))
        return distance

    def _init_pheromone_map(self, initial_distance: float):
        """Inicializar mapa de feromonas con valor heurístico"""
        tau0 = 1.0 / (len(self.nodes) * initial_distance)

        for i, node1 in enumerate(self.nodes):
            for node2 in self.nodes[i + 1 :]:
                edge = self._make_edge(node1, node2)
                self.pheromone_map[edge] = tau0

    def _optimize(self):
        """Bucle principal de optimización"""
        for iteration in range(self.iterations):
            ants = []

            # Cada hormiga empieza desde un nodo diferente (diversificación)
            start_nodes = random.choices(self.nodes, k=self.ant_count)

            # Crear y ejecutar hormigas
            for i in range(self.ant_count):
                ant = Ant(
                    self.nodes,
                    self.pheromone_map,
                    self.lock,
                    start_nodes[i],  # Nodo inicial aleatorio
                    self.distance,
                    self.alpha,
                    self.beta,
                    self.q0,
                    i,
                    seed=self.seed,
                )
                ants.append(ant)
                ant.start()

            # Esperar a que terminen
            for ant in ants:
                ant.join()

            # Recopilar resultados y ordenar por calidad
            valid_ants = [ant for ant in ants if len(ant.trip) == len(self.nodes) + 1]
            valid_ants.sort(key=lambda x: x.trip_distance)

            if valid_ants:
                iteration_best = valid_ants[0].trip_distance

                # Actualizar mejor solución global
                if iteration_best < self.best_distance:
                    self.best_distance = iteration_best
                    self.best_path = valid_ants[0].trip[:]
                    self.stagnation_counter = 0
                else:
                    self.stagnation_counter += 1

                self.distance_history.append(self.best_distance)

                # Evaporación global
                self._evaporate_pheromones()

                # Actualización de feromonas (elite)
                self._update_pheromones_elite(valid_ants[: self.elite_count])

                # Reiniciar feromonas si hay estancamiento
                if self.stagnation_counter >= self.stagnation_limit:
                    print(f"  → Reiniciando feromonas (estancamiento detectado)")
                    self._reset_pheromones()
                    self.stagnation_counter = 0

                print(
                    f"Iteración {iteration + 1}/{self.iterations}: "
                    f"Mejor = {self.best_distance:.2f}, "
                    f"Actual = {iteration_best:.2f}, "
                    f"Promedio = {sum(a.trip_distance for a in valid_ants[:5])/5:.2f}"
                )

    def _evaporate_pheromones(self):
        """Evaporación global de feromonas"""
        for edge in self.pheromone_map:
            self.pheromone_map[edge] *= 1 - self.rho
            if self.pheromone_map[edge] < 1e-10:
                self.pheromone_map[edge] = 1e-10

    def _update_pheromones_elite(self, elite_ants: List[Ant]):
        """Actualizar feromonas usando estrategia elite con peso"""
        for rank, ant in enumerate(elite_ants):
            # Dar más peso a las mejores hormigas
            weight = self.elite_count - rank
            deposit = weight / ant.trip_distance

            for i in range(len(ant.trip) - 1):
                edge = self._make_edge(ant.trip[i], ant.trip[i + 1])
                self.pheromone_map[edge] += deposit

    def _reset_pheromones(self):
        """Reiniciar feromonas cuando hay estancamiento"""
        tau0 = 1.0 / (len(self.nodes) * self.best_distance)
        for edge in self.pheromone_map:
            self.pheromone_map[edge] = tau0

    def distance(self, edge: Tuple) -> float:
        """Calcular distancia euclidiana entre dos nodos"""
        (c1, c2) = edge
        dx = c1[0] - c2[0]
        dy = c1[1] - c2[1]
        return math.sqrt(dx**2 + dy**2)

    def _make_edge(self, node1, node2):
        """Crear edge normalizado"""
        if node1 < node2:
            return (node1, node2)
        return (node2, node1)

    def get_path(self) -> List[Tuple[float, float]]:
        """Obtener el mejor camino encontrado"""
        print(f"\n{'='*60}")
        print(f"Distancia óptima encontrada: {self.best_distance:.2f}")
        print(
            f"Mejora desde inicio: {(self.distance_history[0] - self.best_distance):.2f}"
        )
        print(
            f"Mejora porcentual: {((self.distance_history[0] - self.best_distance) / self.distance_history[0] * 100):.1f}%"
        )
        print(f"{'='*60}")
        return self.best_path

    def get_distance_history(self) -> List[float]:
        """Obtener historial de distancias para análisis"""
        return self.distance_history
