from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import numpy as np
from pyswarms.single import GlobalBestPSO

from .configurations import ProblemConfig
from .individual import Individual
from .results import (MultiObjectiveResult, OptimizationResult,
                      SingleObjectiveResult)


class Problem(ABC):
    """Abstract base class for optimization problems."""

    def __init__(self, config: ProblemConfig):
        """
        Initialize problem.

        :param config: Problem configuration
        :type config: ProblemConfig
        """
        self.config = config
        self.evaluations_count = 0
        self.history: List = []

    @abstractmethod
    def evaluate(self, solution: List) -> Union[float, List[float]]:
        """
        Evaluate a solution.

        :param solution: Solution to evaluate
        :type solution: List
        :return: Fitness value(s)
        :rtype: Union[float, List[float]]
        """
        pass

    @abstractmethod
    def get_bounds(self) -> List[Tuple[float, float]]:
        """
        Get variable bounds.

        :return: List of (min, max) bounds for each variable
        :rtype: List[Tuple[float, float]]
        """
        pass

    @abstractmethod
    def get_result(self) -> OptimizationResult:
        """
        Get optimization result.

        :return: Result object
        :rtype: OptimizationResult
        """
        pass

    def reached_budget(self) -> bool:
        """
        Check if evaluation budget is reached.

        :return: True if budget exhausted
        :rtype: bool
        """
        return self.evaluations_count >= self.config.max_evaluations

    def reset(self) -> None:
        """Reset evaluation counter and history."""
        self.evaluations_count = 0
        self.history = []


class SingleObjectiveProblem(Problem):
    """Base class for single-objective problems."""

    def __init__(self, config: ProblemConfig, minimize: bool = True):
        super().__init__(config)
        self.minimize = minimize
        self.best_fitness: float = float("inf") if minimize else float("-inf")
        self.best_solution: Optional[Individual] = None

    def evaluate(self, solution: List) -> float:
        """
        Evaluate solution and update tracking.

        :param solution: Solution to evaluate
        :type solution: List
        :return: Fitness value
        :rtype: float
        """
        self.evaluations_count += 1
        fitness = self._fitness_function(solution)

        if self._is_better(fitness, self.best_fitness):
            self.best_fitness = fitness
            self.best_solution = solution.copy()

        return fitness

    def _is_better(self, new_fitness: float, current_best: float) -> bool:
        """
        Check whether the new fitness is better than current best.

        :param new_fitness: New fitness value
        :type new_fitness: float
        :param current_best: Current best fitness value
        :type current_best: float
        :return: True if new fitness is better
        :rtype: bool
        """
        if self.minimize:
            return new_fitness < current_best
        return new_fitness > current_best

    @abstractmethod
    def _fitness_function(self, solution: List) -> float:
        """
        Problem-specific fitness function.

        :param solution: Solution to evaluate
        :type solution: List
        :return: Fitness value
        :rtype: float
        """
        pass

    def update_history(self, current_best: float) -> None:
        """
        Update history with current best fitness.

        :param current_best: Best fitness in current generation
        :type current_best: float
        """
        self.history.append(current_best)

    def get_result(self) -> SingleObjectiveResult:
        """
        Get single-objective result.

        :return: Result object
        :rtype: SingleObjectiveResult
        """
        return SingleObjectiveResult(
            problem_name=self.__class__.__name__,
            best_fitness=self.best_fitness,
            best_solution=self.best_solution,
            evaluations_used=self.evaluations_count,
            history=self.history,
        )


class MultiObjectiveProblem(Problem):
    """Base class for multi-objective problems."""

    def __init__(self, config: ProblemConfig, n_objectives: int):
        """
        Initialize multi-objective problem.

        :param config: Problem configuration
        :type config: ProblemConfig
        :param n_objectives: Number of objectives
        :type n_objectives: int
        """
        super().__init__(config)
        self.n_objectives = n_objectives
        self.pareto_front: List[List[float]] = []
        self.pareto_solutions: List[List] = []

    def evaluate(self, solution: List) -> List[float]:
        """
        Evaluate solution (multi-objective).

        :param solution: Solution to evaluate
        :type solution: List
        :return: List of objective values
        :rtype: List[float]
        """
        self.evaluations_count += 1
        return self._fitness_function(solution)

    @abstractmethod
    def _fitness_function(self, solution: List) -> List[float]:
        """
        Problem-specific multi-objective fitness function.

        :param solution: Solution to evaluate
        :type solution: List
        :return: List of objective values
        :rtype: List[float]
        """
        pass

    def update_pareto_front(
        self, objectives: List[List[float]], solutions: List[List]
    ) -> None:
        """
        Update Pareto front with new solutions.

        :param objectives: List of objective vectors
        :type objectives: List[List[float]]
        :param solutions: Corresponding solutions
        :type solutions: List[List]
        """
        self.pareto_front = objectives
        self.pareto_solutions = solutions

    def get_result(self) -> MultiObjectiveResult:
        """
        Get multi-objective result.

        :return: Result object
        :rtype: MultiObjectiveResult
        """
        return MultiObjectiveResult(
            problem_name=self.__class__.__name__,
            best_fitness=self.pareto_front,
            best_solution=self.pareto_solutions,
            evaluations_used=self.evaluations_count,
            pareto_front=self.pareto_front,
            metrics={},
        )


class HimmelblauProblem(SingleObjectiveProblem):
    """Himmelblau function optimization problem."""

    def __init__(self, config: ProblemConfig, minimize: bool = True):
        super().__init__(config, minimize)
        self.bounds = [(-5.0, 5.0), (-5.0, 5.0)]

    def _fitness_function(self, solution: List) -> float:
        """
        Himmelblau function: f(x,y) = (x²+y-11)² + (x+y²-7)².

        :param solution: [x, y] coordinates
        :type solution: List
        :return: Function value
        :rtype: float
        """
        if isinstance(solution, np.ndarray) and solution.ndim == 2:
            x = solution[:, 0]
            y = solution[:, 1]
            return (x**2 + y - 11) ** 2 + (x + y**2 - 7) ** 2

        x, y = solution
        return (x**2 + y - 11) ** 2 + (x + y**2 - 7) ** 2

    def get_bounds(self) -> List[Tuple[float, float]]:
        """
        Get Himmelblau bounds.

        :return: [(-5, 5), (-5, 5)]
        :rtype: List[Tuple[float, float]]
        """
        return self.bounds

    def get_pyswarms_result(
        self, c1: float = 0.5, c2: float = 0.3, w: float = 0.9, pbar: bool = True
    ) -> SingleObjectiveResult:
        """
        Optimize using PySwarms GlobalBestPSO.

        :param c1: Cognitive coefficient
        :type c1: float
        c1 ⁠:⁠ float cognitive parameter
        :param c2: Social coefficient
        :type c2: float
        :param w: Inertia coefficient
        :type w: float
        :return: Pyswarms optimization result
        :rtype: SingleObjectiveResult
        """

        lower = [b[0] for b in self.bounds]
        upper = [b[1] for b in self.bounds]
        bounds = (np.array(lower), np.array(upper))

        optimizer = GlobalBestPSO(
            n_particles=10,
            dimensions=2,
            options={"c1": c1, "c2": c2, "w": w},
            bounds=bounds,
            ftol=1e-5,
            ftol_iter=100,
        )

        cost, pos = optimizer.optimize(self._fitness_function, iters=3500, verbose=pbar)

        return SingleObjectiveResult(
            problem_name="PySwarms" + self.__class__.__name__,
            best_fitness=cost,
            best_solution=pos,
            evaluations_used=len(optimizer.cost_history),
            history=optimizer.cost_history,
        )


class TSProblem(SingleObjectiveProblem):
    """Traveling Salesman Problem (TSP)."""

    def __init__(self, config: ProblemConfig, cities: List[Tuple[float, float]], minimize: bool = True):
        super().__init__(config, minimize)
        self.cities = np.asanyarray(cities, dtype=np.float64)
        self.n_cities = self.cities.shape[0]
        self.dist_matrix = self._compute_distance_matrix()

    def get_bounds(self) -> Tuple[int, int]:
        """
        In TSP by permutation, bounds are not used the same way as in continuous problems,
        but it is defined the range of valid indices for compatibility.
        """
        return (0, self.n_cities - 1)

    def _compute_distance_matrix(self) -> np.ndarray:
        """
        Compute distance matrix between all the cities.

        :return: Pairwise distance matrix between cities with shape (n_cities, n_cities)
        :rtype: numpy.ndarray[np.float64]
        """
        coords = np.array(self.cities)
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        return np.sqrt(np.sum(diff**2, axis=-1))

    def _sol_distance(self, solution: np.ndarray, closed: bool = False) -> float:
        """
        Compute the total length of a path defined by an ordered list of cities.

        :param solution: Ordered array of city indices defining the path.
        :type solution: numpy.ndarray
        :param closed: Whether to close the path into a cycle.
        :type closed: bool
        :return: Total path length.
        :rtype: float
        """
        solution = np.asanyarray(solution)

        if not np.all((0 <= solution) & (solution < self.n_cities)):
            raise ValueError("Solution contains invalid city index.")

        if len(solution) != self.n_cities:
            raise ValueError("Solution must include all cities exactly once.")

        from_cities = solution[:-1]
        to_cities = solution[1:]

        total = self.dist_matrix[from_cities, to_cities].sum()

        if closed:
            total += self.dist_matrix[solution[-1], solution[0]]

        return float(total)

    def _fitness_function(
        self, solution: np.ndarray, closed_path: bool = True
    ) -> float:
        """
        Fitness function for TSP: inverse of path length.

        :param solution: Ordered list of city indices defining the path.
        :type solution: np.ndarray
        :param closed_path: Whether the path is closed (cycle) or open.
        :type closed_path: bool
        :return: Fitness value (inverse of path length).
        :rtype: float
        """
        # TODO: We are considering only open paths for now, rewrite super().evaluate to pass this parameter
        return 1 / (1 + self._sol_distance(solution, closed=closed_path))

    def get_aco_results(self):
        raise NotImplementedError("ACO format conversion not implemented yet.")
