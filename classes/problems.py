from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Tuple, Union

import numpy as np

from .configurations import ProblemConfig
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

    def __init__(self, config: ProblemConfig):
        super().__init__(config)
        self.best_fitness: float = float("inf")
        self.best_solution: Optional[List] = None

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

        if fitness < self.best_fitness:
            self.best_fitness = fitness
            self.best_solution = solution.copy()

        return fitness

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

    def __init__(self, config: ProblemConfig):
        super().__init__(config)
        self.bounds = [(-5.0, 5.0), (-5.0, 5.0)]

    def _fitness_function(self, solution: List) -> float:
        """
        Himmelblau function: f(x,y) = (x²+y-11)² + (x+y²-7)².

        :param solution: [x, y] coordinates
        :type solution: List
        :return: Function value
        :rtype: float
        """
        x, y = solution[0], solution[1]
        return (x**2 + y - 11) ** 2 + (x + y**2 - 7) ** 2

    def get_bounds(self) -> List[Tuple[float, float]]:
        """
        Get Himmelblau bounds.

        :return: [(-5, 5), (-5, 5)]
        :rtype: List[Tuple[float, float]]
        """
        return self.bounds

    def to_pyswarms_format(self) -> Tuple[Callable, List[float], List[float]]:
        """
        Convert to pyswarms-compatible format.

        :return: (fitness_func, lower_bounds, upper_bounds)
        :rtype: Tuple[Callable, List[float], List[float]]
        """

        def fitness_wrapper(X):
            """Wrapper for vectorized evaluation."""
            return np.array([self._fitness_function(x) for x in X])

        lower = [b[0] for b in self.bounds]
        upper = [b[1] for b in self.bounds]
        return fitness_wrapper, lower, upper
