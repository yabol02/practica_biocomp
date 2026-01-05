from typing import List, Optional

import numpy as np

from .crossover import Crossover
from .initialization import Initialization
from .mutation import Mutation
from .population import Population
from .problems import MultiObjectiveProblem, SingleObjectiveProblem
from .replacement import Replacement
from .results import MultiObjectiveResult, SingleObjectiveResult
from .selection import Selection


class GeneticAlgorithmSO:
    """Basic Genetic Algorithm implementation."""

    def __init__(
        self,
        problem: SingleObjectiveProblem,
        population_size: int,
        initialization: Initialization,
        selection: Selection,
        crossover: Crossover,
        mutation: Mutation,
        replacement: Replacement,
        print_interval: int = 10,
    ):
        """
        Initialize Genetic Algorithm.

        :param problem: Problem to optimize
        :type problem: SingleObjectiveProblem
        :param population_size: Number of individuals
        :type population_size: int
        :param initialization: Initialization strategy
        :type initialization: Initialization
        :param selection: Selection operator
        :type selection: Selection
        :param crossover: Crossover operator
        :type crossover: Crossover
        :param mutation: Mutation operator
        :type mutation: Mutation
        :param replacement: Replacement strategy
        :type replacement: Replacement
        :param print_interval: Interval of generations to print progress
        :type print_interval: int
        """
        self.problem = problem
        self.population_size = population_size
        self.initialization = initialization
        self.selection = selection
        self.crossover = crossover
        self.mutation = mutation
        self.replacement = replacement
        self.print_interval = print_interval

        self._current_population: Optional[Population] = None
        self._generation_count: int = 0

    @property
    def current_population(self) -> Optional[Population]:
        """
        Get current population.

        :return: Current population
        :rtype: Optional[Population]
        """
        return self._current_population

    def initialize(self) -> None:
        """
        Prepares the initial population and evaluates it.
        """
        self.problem.config.initialize_random_state()

        bounds = self.problem.get_bounds()
        self._current_population = self.initialization(
            self.population_size, bounds, self.problem
        )
        self._current_population.evaluate_population(self.problem.evaluate)

        self._generation_count = 0
        self.problem.update_history(self._current_population)

    def step(self) -> None:
        """
        Perform a single generation step of the genetic algorithm.
        1. Selection and Reproduction
        2. Replacement (Generate new population)
        3. Evaluation and Logging
        """
        if self._current_population is None:
            self.initialize()

        selected = self.selection(self._current_population, self.population_size)
        offspring = self.crossover(selected, len(self._current_population))
        offspring = self.mutation(offspring)

        self._current_population = self.replacement(self._current_population, offspring)

        self._current_population.evaluate_population(self.problem.evaluate)
        self.problem.update_history(self._current_population)

        self._generation_count += 1

    def run(self) -> SingleObjectiveResult:
        """
        Complete execution of the algorithm until the budget is exhausted.

        :return: Optimization result
        :rtype: SingleObjectiveResult
        """
        if self._current_population is None:
            self.initialize()

        while not self.problem.reached_budget():
            self.step()

            if self._generation_count % self.print_interval == 0:
                stats = self._current_population.stats
                print(
                    f"Gen {self._generation_count:6d} | "
                    f"Evals: {self.problem.evaluations_count:7d} | "
                    f"Best Global: {stats['best']:.5f} | "
                    f"Current Population: {stats['mean']:.5f} ± {stats['std']:.3e}"
                    )

        self._current_population.evaluate_population(self.problem.evaluate)
        self.problem.update_history(self._current_population)

        return self.problem.get_result()

    def __str__(self) -> str:
        return f"GA(pop={self.population_size})"

    def __repr__(self) -> str:
        return (
            f"GeneticAlgorithm("
            f"gen={self._generation_count}, "
            f"pop={self.population_size}, "
            f"sel={self.selection.__class__.__name__}, "
            f"cx={self.crossover.__class__.__name__}, "
            f"mut={self.mutation.__class__.__name__}, "
            f"rep={self.replacement.__class__.__name__}"
            f")"
        )


class GeneticAlgorithmMO:
    """
    Multi-Objective Genetic Algorithm implementation.

    Supports different optimization strategies:
    - 'pareto': Pareto-based selection (NSGA-II style)
    - 'weighted': Weighted sum approach
    - 'sequential': Sequential optimization of objectives
    """

    def __init__(
        self,
        problem: MultiObjectiveProblem,
        population_size: int,
        initialization: Initialization,
        selection: Selection,
        crossover: Crossover,
        mutation: Mutation,
        replacement: Replacement,
    ):
        """
        Initialize Multi-Objective Genetic Algorithm.

        :param problem: Multi-Objective problem to optimize
        :type problem: MultiObjectiveProblem
        :param population_size: Number of individuals each generation
        :type population_size: int
        :param initialization: Initialization method
        :type initialization: Initialization
        :param selection: Selection method
        :type selection: Selection
        :param crossover: Crossover method
        :type crossover: Crossover
        :param mutation: Mutation method
        :type mutation: Mutation
        :param replacement: Replacement method
        :type replacement: Replacement
        """
        self.problem = problem
        self.population_size = population_size
        self.initialization = initialization
        self.selection = selection
        self.crossover = crossover
        self.mutation = mutation
        self.replacement = replacement

    def _compute_pareto_ranks(self, population: Population) -> np.ndarray:
        """
        Compute Pareto rank for each individual using fast non-dominated sorting.

        :param population: Current population
        :type population: Population
        :return: Array of ranks (0 = non-dominated front)
        :rtype: np.ndarray
        """
        n = len(population)
        objectives = np.array([ind.fitness for ind in population])

        # Domination counts and dominated solutions
        domination_count = np.zeros(n, dtype=int)
        dominated_solutions = [[] for _ in range(n)]
        ranks = np.zeros(n, dtype=int)

        # Find domination relationships
        for i in range(n):
            for j in range(i + 1, n):
                if self.problem.dominates(objectives[i], objectives[j]):
                    dominated_solutions[i].append(j)
                    domination_count[j] += 1
                elif self.problem.dominates(objectives[j], objectives[i]):
                    dominated_solutions[j].append(i)
                    domination_count[i] += 1

        # Assign ranks
        current_front = np.where(domination_count == 0)[0].tolist()
        rank = 0

        while current_front:
            next_front = []
            for i in current_front:
                ranks[i] = rank
                for j in dominated_solutions[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)

            current_front = next_front
            rank += 1

        return ranks

    def _compute_crowding_distance(
        self, population: Population, front_indices: List[int]
    ) -> np.ndarray:
        """
        Compute crowding distance for individuals in a front.

        :param population: Current population
        :type population: Population
        :param front_indices: Indices of individuals in the front
        :type front_indices: List[int]
        :return: Crowding distances
        :rtype: np.ndarray
        """
        n = len(front_indices)
        if n <= 2:
            return np.full(n, np.inf)

        distances = np.zeros(n)
        objectives = np.array([population[i].fitness for i in front_indices])

        # Normalize and compute distances for each objective
        for m in range(self.problem.n_objectives):
            sorted_idx = np.argsort(objectives[:, m])
            obj_range = objectives[sorted_idx[-1], m] - objectives[sorted_idx[0], m]

            if obj_range == 0:
                continue

            # Boundary points get infinite distance
            distances[sorted_idx[0]] = np.inf
            distances[sorted_idx[-1]] = np.inf

            # Interior points
            for i in range(1, n - 1):
                distances[sorted_idx[i]] += (
                    objectives[sorted_idx[i + 1], m] - objectives[sorted_idx[i - 1], m]
                ) / obj_range

        return distances

    def run(self) -> MultiObjectiveResult:
        """
        Execute multi-objective genetic algorithm.

        :return: Optimization result
        :rtype: MultiObjectiveResult
        """
        self.problem.config.initialize_random_state()

        # Initialize population
        bounds = self.problem.get_bounds()
        population: Population = self.initialization(
            self.population_size, bounds, self.problem
        )

        generation = 0

        # Main loop
        while not self.problem.reached_budget():
            population.evaluate_population(self.problem.evaluate)

            all_objectives = [ind.fitness for ind in population]
            all_solutions = [ind.genotype for ind in population]
            self.problem.update_pareto_front(
                new_objectives=all_objectives, new_solutions=all_solutions
            )
            self.problem.update_history(all_objectives)

            if generation % 10 == 0:
                n_pareto = len(self.problem.pareto_front)
                idx = 0
                if population.individuals:
                    some_obj = population[idx].fitness
                    print(
                        f"Gen {generation}: Evals={self.problem.evaluations_count}/{self.problem.config.max_evaluations}, "
                        f"Pareto size={n_pareto}, "
                        f"Sample objectives={[f'{x:.4f}' for x in some_obj]}"
                    )

            if self.problem.reached_budget():
                break

            parents = population

            # Selection
            selected = self.selection(population, int(self.population_size * 0.5))

            # Crossover
            offspring = self.crossover(selected, len(parents))

            # Mutation
            offspring = self.mutation(offspring)

            # Replacement
            population = self.replacement(parents, offspring)
            generation += 1

        # Final evaluation and update
        population.evaluate_population(self.problem.evaluate)
        all_objectives = [ind.fitness for ind in population]
        all_solutions = [ind.genotype for ind in population]
        self.problem.update_pareto_front(all_objectives, all_solutions)

        return self.problem.get_result()

    def __str__(self) -> str:
        return f"MOGA(pop={self.population_size})"

    def __repr__(self) -> str:
        return (
            f"MultiObjectiveGeneticAlgorithm("
            f"pop={self.population_size}, "
            f"sel={self.selection.__class__.__name__}, "
            f"cx={self.crossover.__class__.__name__}, "
            f"mut={self.mutation.__class__.__name__}, "
            f"rep={self.replacement.__class__.__name__}"
            f")"
        )
