from abc import ABC, abstractmethod
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
from pymoo.problems import get_problem
from pymoo.visualization.scatter import Scatter
from pyswarms.single import GlobalBestPSO
from scipy.optimize import minimize

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
        self.pareto_front: np.ndarray = np.empty((0, n_objectives))
        self.pareto_solutions: np.ndarray = np.empty((0, 0))
        self.pareto_history: List[Tuple[np.ndarray, np.ndarray]] = []
        self.all_objectives: List[np.ndarray] = []
        self.all_solutions: List[np.ndarray] = []

    def evaluate(self, solution: List) -> List[float]:
        """
        Evaluate solution (multi-objective).

        :param solution: Solution to evaluate
        :type solution: List
        :return: List of objective values
        :rtype: List[float]
        """
        self.evaluations_count += 1
        objectives = self._fitness_function(solution)

        self.all_objectives.append(np.asanyarray(objectives))
        self.all_solutions.append(np.asanyarray(solution))

        return np.array(objectives)

    @abstractmethod
    def _fitness_function(self, solution: Union[List, np.ndarray]) -> np.ndarray:
        """
        Problem-specific multi-objective fitness function.

        :param solution: Solution to evaluate
        :type solution: List
        :return: List of objective values
        :rtype: List[float]
        """
        pass

    def _calculate_non_dominated(
        self, objectives: np.ndarray, solutions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Auxiliar method to filter dominated solutions from a given set.

        :param objectives: Array of objective vectors
        :type objectives: np.ndarray
        :param solutions: Corresponding solutions
        :type solutions: np.ndarray
        :return: Non-dominated objectives and corresponding solutions
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        is_dominated = np.zeros(len(objectives), dtype=bool)

        for i, obj_i in enumerate(objectives):
            if is_dominated[i]:
                continue
            for j, obj_j in enumerate(objectives):
                if i != j and not is_dominated[j]:
                    if self.dominates(obj_j, obj_i):
                        is_dominated[i] = True
                        break

        return objectives[~is_dominated], solutions[~is_dominated]

    @property
    def true_pareto_front(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Obtain the 'true' Pareto front calculated over ALL historical solutions.

        :return: Non-dominated objectives and corresponding solutions
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        if not self.all_objectives:
            return np.array([]), np.array([])

        all_obj_flat = np.vstack(self.all_objectives)
        all_sol_flat = np.vstack(self.all_solutions)

        return self._calculate_non_dominated(all_obj_flat, all_sol_flat)

    def dominates(self, obj1: np.ndarray, obj2: np.ndarray) -> bool:
        """
        Check if obj1 dominates obj2 (Pareto dominance).

        :param obj1: First objective vector
        :type obj1: np.ndarray
        :param obj2: Second objective vector
        :type obj2: np.ndarray
        :return: True if obj1 dominates obj2
        :rtype: bool
        """
        return np.all(obj1 <= obj2) and np.any(obj1 < obj2)

    def update_pareto_front(
        self, new_objectives: List[np.ndarray], new_solutions: List[np.ndarray]
    ) -> None:
        """
        Update Pareto front with new solutions using non-dominated sorting.

        :param new_objectives: List of objective vectors
        :type new_objectives: List[np.ndarray]
        :param new_solutions: Corresponding solutions
        :type new_solutions: List[np.ndarray]
        """
        if not new_objectives:
            return

        objectives_array = np.asanyarray(new_objectives)
        solutions_array = np.asanyarray(new_solutions)

        if self.pareto_front.size > 0:
            combined_objectives = np.vstack((self.pareto_front, objectives_array))
            combined_solutions = np.vstack((self.pareto_solutions, solutions_array))
        else:
            combined_objectives = objectives_array
            combined_solutions = solutions_array

        best_obj, best_sol = self._calculate_non_dominated(
            combined_objectives, combined_solutions
        )

        self.pareto_front = best_obj
        self.pareto_solutions = best_sol
        self.pareto_history.append((best_obj.copy(), best_sol.copy()))

    def update_history(self, current_objectives: Optional[np.ndarray] = None) -> None:
        """
        Update history with current Pareto front metrics.
        If available, it stores the number of non-dominated solutions and hypervolume.

        :param current_objectives: Current population objectives (optional)
        :type current_objectives: Optional[np.ndarray]
        """
        metrics = {
            "generation": len(self.history),
            "pareto": self.pareto_solutions.copy(),
            "pareto_spread": np.mean(np.var(self.pareto_front, axis=0)),
            "nd_ratio": len(self.pareto_front) / len(current_objectives),
        }

        self.history.append(metrics)

    def get_result(self) -> MultiObjectiveResult:
        """
        Get multi-objective result.

        :return: Result object
        :rtype: MultiObjectiveResult
        """
        final_front, final_sols = self.true_pareto_front

        metrics = {
            "n_pareto_solutions": len(final_front),
            "pareto_spread": (
                np.mean(np.var(final_front, axis=0)) if np.any(final_front) else None
            ),
            "history_length": len(self.pareto_history),
        }

        return MultiObjectiveResult(
            problem_name=self.__class__.__name__,
            best_fitness=final_front,
            best_solution=final_sols,
            evaluations_used=self.evaluations_count,
            pareto_front=final_front,
            metrics=metrics,
        )

    def plot_pareto_front(
        self, show_true_front: bool = False, problem_name: str = ""
    ) -> Scatter:
        """
        Plot the obtained Pareto front.

        :param show_true_front: Whether to show the true Pareto front (if available)
        :type show_true_front: bool
        """
        if not np.any(self.pareto_front):
            print("No Pareto front available yet.")
            return

        pareto = np.asarray(self.pareto_front)

        problem_name = problem_name if problem_name else self.__class__.__name__
        scatter = Scatter(title=f"{problem_name} - Pareto Front")
        scatter.add(pareto, color="blue", label="Obtained")

        if show_true_front and hasattr(self, "get_true_pareto_front"):
            true_front = self.get_true_pareto_front()
            if true_front is not None:
                scatter.add(true_front, color="red", label="True Front", alpha=0.5)

        return scatter


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

    def get_scipy_result(
        self,
        method: str = "L-BFGS-B",
        x0: Optional[List[float]] = None,
        tol: float = 1e-8,
    ) -> SingleObjectiveResult:
        """
        Optimize using scipy.optimize.minimize.

        :param method: Optimization method (e.g., 'L-BFGS-B', 'SLSQP', 'Nelder-Mead')
        :type method: str
        :param x0: Initial guess. If None, uses center of bounds.
        :type x0: Optional[List[float]]
        :param tol: Tolerance for termination.
        :type tol: float
        :return: Scipy optimization result
        :rtype: SingleObjectiveResult
        """

        eval_count = 0
        history = []

        def tracked_fitness(x):
            nonlocal eval_count
            eval_count += 1
            fitness = self._fitness_function(x)
            history.append(fitness)
            return fitness

        if x0 is None:
            x0 = [(b[0] + b[1]) / 2 for b in self.bounds]

        scipy_bounds = [(b[0], b[1]) for b in self.bounds]

        result = minimize(
            tracked_fitness,
            x0=x0,
            method=method,
            bounds=scipy_bounds,
            tol=tol,
        )

        return SingleObjectiveResult(
            problem_name=f"Scipy_{method}_{self.__class__.__name__}",
            best_fitness=result.fun,
            best_solution=result.x.tolist(),
            evaluations_used=eval_count,
            history=history,
        )


class TSProblem(SingleObjectiveProblem):
    """Traveling Salesman Problem (TSP)."""

    def __init__(
        self,
        config: ProblemConfig,
        cities: Iterable[Tuple[float, float]],
        minimize: bool = True,
    ):
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


class PymooProblem(MultiObjectiveProblem):
    """Wrapper for pymoo multi-objective problems."""

    def __init__(
        self, config, problem_name: str, n_var: Optional[int] = None, **problem_kwargs
    ):
        """
        Initialize Pymoo problem.

        :param config: Problem configuration
        :type config: ProblemConfig
        :param problem_name: Name of the pymoo problem (e.g., 'zdt1', 'zdt3', 'mw7', 'mw14')
        :type problem_name: str
        :param n_var: Number of variables (optional, uses Pymoo default if not provided)
        :type n_var: Optional[int]
        :param problem_kwargs: Additional problem-specific arguments
        """
        if n_var is not None:
            problem_kwargs["n_var"] = n_var

        self.pymoo_problem = get_problem(problem_name.lower(), **problem_kwargs)

        super().__init__(config, n_objectives=self.pymoo_problem.n_obj)

        self.problem_name = problem_name.upper()
        self.n_var = self.pymoo_problem.n_var
        self.lower_bound = self.pymoo_problem.xl
        self.upper_bound = self.pymoo_problem.xu
        self.minimize = True

    def _fitness_function(self, solution: Union[List, np.ndarray]) -> np.ndarray:
        """
        Evaluate the problem.

        :param solution: Solution to evaluate
        :type solution: Union[List, np.ndarray]
        :return: Objective values
        :rtype: np.ndarray
        """
        solution = np.asanyarray(solution)

        if solution.ndim == 1:
            solution = solution.reshape(1, -1)

        return self.pymoo_problem.evaluate(solution, return_values_of=["F"]).squeeze()

    def get_bounds(self) -> List[Tuple[float, float]]:
        """
        Get variable bounds.

        :return: List of (min, max) bounds for each variable
        :rtype: List[Tuple[float, float]]
        """
        if isinstance(self.lower_bound, np.ndarray):
            return [
                (float(l), float(u)) for l, u in zip(self.lower_bound, self.upper_bound)
            ]
        else:
            return [(float(self.lower_bound), float(self.upper_bound))] * self.n_var

    def get_true_pareto_front(self, n_points: int = 100) -> Optional[np.ndarray]:
        """
        Get the true Pareto front from Pymoo (if available).

        :param n_points: Number of points to sample
        :type n_points: int
        :return: True Pareto front or None
        :rtype: Optional[np.ndarray]
        """
        return self.pymoo_problem.pareto_front()
