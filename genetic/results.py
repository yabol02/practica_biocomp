import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Union


@dataclass
class OptimizationResult:
    """Results from optimization run."""

    problem_name: str
    best_fitness: Union[float, List[float]]
    best_solution: List
    evaluations_used: int
    history: List[Union[float, List[float]]] = field(default_factory=list)

    def save_csv(self, filepath: str) -> None:
        """Save results to CSV file."""
        raise NotImplementedError("Subclass must implement save_csv")


@dataclass
class SingleObjectiveResult(OptimizationResult):
    """Results for single-objective optimization."""
    best_solution_found_on: int

    def save_csv(self, filepath: str) -> None:
        """
        Save single-objective results to CSV.

        :param filepath: Output CSV path
        :type filepath: str
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["generation", "min_fitness"])
            for gen, fitness in enumerate(self.history):
                writer.writerow([gen, fitness])


@dataclass
class MultiObjectiveResult(OptimizationResult):
    """Results for multi-objective optimization."""

    pareto_front: List[List[float]] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    def save_csv(self, filepath: str) -> None:
        """
        Save multi-objective Pareto front to CSV.

        :param filepath: Output CSV path
        :type filepath: str
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            if self.pareto_front:
                n_objectives = len(self.pareto_front[0])
                headers = [f"objective{i+1}" for i in range(n_objectives)]
                writer.writerow(headers)
                writer.writerows(self.pareto_front)
