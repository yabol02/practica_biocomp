"""
Experiment runner for mono-objective genetic algorithm optimization.

This script manages the execution of mono-objective GA experiments across
different configurations and problem instances, with support for parallel
execution and comprehensive metrics collection.
"""

import csv
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from genetic.algorithms import GeneticAlgorithmSO
from genetic.configurations import ProblemConfig
from genetic.crossover import BlendCrossover, OrderCrossover
from genetic.initialization import (Initialization, NeighborInitialization,
                                    PermutationInitialization,
                                    RandomInitialization)
from genetic.mutation import SwapMutation, UniformMutation
from genetic.problems import HimmelblauProblem, TSProblem
from genetic.replacement import (ElitistReplacement, GenerationalReplacement,
                                 MuPlusLambdaReplacement)
from genetic.selection import Selection, TournamentSelection


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""

    config_id: str
    problem_type: str
    problem_params: Dict
    initialization: Initialization
    selection: Selection
    crossover: object
    mutation: object
    replacement: object
    population_size: int
    max_evaluations: int

    def to_dict(self) -> Dict:
        """
        Convert configuration to dictionary for CSV output.

        :return: Dictionary representation of configuration
        :rtype: Dict
        """
        return {
            "config_id": self.config_id,
            "problem_type": self.problem_type,
            "initialization": self.initialization.__class__.__name__,
            "selection": self.selection.__class__.__name__,
            "crossover": self.crossover.__class__.__name__,
            "mutation": self.mutation.__class__.__name__,
            "replacement": self.replacement.__class__.__name__,
            "population_size": self.population_size,
            "max_evaluations": self.max_evaluations,
        }


@dataclass
class ExperimentResult:
    """Results from a single experiment run."""

    config_id: str
    seed: int
    best_fitness: float
    evals_to_best: int
    execution_time: float
    total_evaluations: int
    final_generation: int
    convergence_rate: float

    def to_dict(self) -> Dict:
        """
        Convert result to dictionary for CSV output.

        :return: Dictionary representation of result
        :rtype: Dict
        """
        return asdict(self)


class ExperimentRunner:
    """
    Manages execution of mono-objective GA experiments.

    This class orchestrates the execution of genetic algorithm experiments
    across multiple configurations and seeds, with support for parallel
    execution and comprehensive metrics collection.
    """

    def __init__(
        self,
        output_dir: str = "experiments/mono/results",
        max_workers: int = 4,
        log_level: int = logging.INFO,
    ):
        """
        Initialize experiment runner.

        :param output_dir: Directory to save experiment results
        :type output_dir: str
        :param max_workers: Maximum number of parallel workers
        :type max_workers: int
        :param log_level: Logging level
        :type log_level: int
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers

        # Configure logging
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self.logger = logging.getLogger(__name__)

    def _run_single_experiment(
        self, config: ExperimentConfig, seed: int
    ) -> ExperimentResult:
        """
        Execute a single experiment with given configuration and seed.

        :param config: Experiment configuration
        :type config: ExperimentConfig
        :param seed: Random seed for reproducibility
        :type seed: int
        :return: Experiment result
        :rtype: ExperimentResult
        """
        self.logger.info(
            f"Starting experiment: config_id={config.config_id}, seed={seed}"
        )

        start_time = time.time()

        # Create problem instance
        problem_config = ProblemConfig(
            max_evaluations=config.max_evaluations, seed=seed
        )

        if config.problem_type == "Himmelblau":
            problem = HimmelblauProblem(problem_config)
        elif config.problem_type == "TSP":
            problem = TSProblem(problem_config, **config.problem_params)
        else:
            raise ValueError(f"Unknown problem type: {config.problem_type}")

        # Create GA instance with print_interval set high to suppress logs
        ga = GeneticAlgorithmSO(
            problem=problem,
            population_size=config.population_size,
            initialization=config.initialization,
            selection=config.selection,
            crossover=config.crossover,
            mutation=config.mutation,
            replacement=config.replacement,
            print_interval=999999,  # Suppress progress prints
        )

        # Run optimization
        result = ga.run()

        execution_time = time.time() - start_time

        # Calculate convergence rate (how quickly best fitness was found)
        # Only calculate if best solution was found (best_solution_found_on >= 0)
        convergence_rate = (
            result.best_solution_found_on / result.evaluations_used
            if result.evaluations_used > 0 and result.best_solution_found_on >= 0
            else 0.0
        )

        # Calculate final generation (approximate)
        final_generation = len(result.history)

        experiment_result = ExperimentResult(
            config_id=config.config_id,
            seed=seed,
            best_fitness=result.best_fitness,
            evals_to_best=result.best_solution_found_on,
            execution_time=execution_time,
            total_evaluations=result.evaluations_used,
            final_generation=final_generation,
            convergence_rate=convergence_rate,
        )

        self.logger.info(
            f"Completed experiment: config_id={config.config_id}, "
            f"seed={seed}, best_fitness={result.best_fitness:.6f}, "
            f"time={execution_time:.2f}s"
        )

        return experiment_result

    def run_experiments(
        self, configs: List[ExperimentConfig], seeds: List[int]
    ) -> Dict[str, List[ExperimentResult]]:
        """
        Run experiments for all configurations and seeds.

        :param configs: List of experiment configurations
        :type configs: List[ExperimentConfig]
        :param seeds: List of random seeds to use
        :type seeds: List[int]
        :return: Dictionary mapping config_id to list of results
        :rtype: Dict[str, List[ExperimentResult]]
        """
        all_results: Dict[str, List[ExperimentResult]] = {
            config.config_id: [] for config in configs
        }

        total_experiments = len(configs) * len(seeds)
        self.logger.info(
            f"Starting {total_experiments} experiments "
            f"({len(configs)} configs × {len(seeds)} seeds)"
        )

        # Create list of all experiment tasks
        tasks = [
            (config, seed) for config in configs for seed in seeds
        ]

        # Execute experiments in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(self._run_single_experiment, config, seed): (
                    config,
                    seed,
                )
                for config, seed in tasks
            }

            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_task):
                config, seed = future_to_task[future]
                try:
                    result = future.result()
                    all_results[config.config_id].append(result)
                    completed += 1
                    self.logger.info(
                        f"Progress: {completed}/{total_experiments} experiments completed"
                    )
                except Exception as exc:
                    self.logger.error(
                        f"Experiment failed: config_id={config.config_id}, "
                        f"seed={seed}, error={exc}"
                    )

        return all_results

    def save_results_csv(
        self,
        config: ExperimentConfig,
        results: List[ExperimentResult],
    ) -> None:
        """
        Save experiment results to CSV file.

        :param config: Experiment configuration
        :type config: ExperimentConfig
        :param results: List of experiment results
        :type results: List[ExperimentResult]
        """
        if not results:
            self.logger.warning(
                f"No results to save for config_id={config.config_id}"
            )
            return

        # Create output file path
        output_file = self.output_dir / f"{config.config_id}.csv"

        # Write results to CSV
        with open(output_file, "w", newline="") as f:
            # Get all field names from result and config
            result_fields = list(results[0].to_dict().keys())
            config_fields = [
                k for k in config.to_dict().keys() if k != "config_id"
            ]

            fieldnames = result_fields + config_fields

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            config_dict = config.to_dict()
            for result in results:
                row = result.to_dict()
                # Add config fields (excluding config_id which is already in result)
                for key in config_fields:
                    row[key] = config_dict[key]
                writer.writerow(row)

        self.logger.info(f"Saved results to {output_file}")

    def save_summary_statistics(
        self, all_results: Dict[str, List[ExperimentResult]]
    ) -> None:
        """
        Save summary statistics across all configurations.

        :param all_results: Dictionary mapping config_id to results
        :type all_results: Dict[str, List[ExperimentResult]]
        """
        summary_file = self.output_dir / "summary_statistics.csv"

        with open(summary_file, "w", newline="") as f:
            fieldnames = [
                "config_id",
                "n_runs",
                "best_fitness_mean",
                "best_fitness_std",
                "best_fitness_min",
                "best_fitness_max",
                "evals_to_best_mean",
                "evals_to_best_std",
                "execution_time_mean",
                "execution_time_std",
                "convergence_rate_mean",
                "convergence_rate_std",
            ]

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for config_id, results in all_results.items():
                if not results:
                    continue

                best_fitness_values = [r.best_fitness for r in results]
                evals_to_best_values = [r.evals_to_best for r in results]
                execution_time_values = [r.execution_time for r in results]
                convergence_rate_values = [r.convergence_rate for r in results]

                summary = {
                    "config_id": config_id,
                    "n_runs": len(results),
                    "best_fitness_mean": np.mean(best_fitness_values),
                    "best_fitness_std": np.std(best_fitness_values),
                    "best_fitness_min": np.min(best_fitness_values),
                    "best_fitness_max": np.max(best_fitness_values),
                    "evals_to_best_mean": np.mean(evals_to_best_values),
                    "evals_to_best_std": np.std(evals_to_best_values),
                    "execution_time_mean": np.mean(execution_time_values),
                    "execution_time_std": np.std(execution_time_values),
                    "convergence_rate_mean": np.mean(convergence_rate_values),
                    "convergence_rate_std": np.std(convergence_rate_values),
                }

                writer.writerow(summary)

        self.logger.info(f"Saved summary statistics to {summary_file}")


def create_himmelblau_configs() -> List[ExperimentConfig]:
    """
    Create experiment configurations for Himmelblau problem.

    :return: List of experiment configurations
    :rtype: List[ExperimentConfig]
    """
    configs = []

    # Define operator variations
    initializations = [RandomInitialization()]
    selections = [TournamentSelection(tournament_size=3)]
    crossovers = [BlendCrossover(alpha=0.5)]
    mutations = [
        UniformMutation(mutation_rate=0.1, bounds=[(-5.0, 5.0), (-5.0, 5.0)])
    ]
    replacements = [
        GenerationalReplacement(),
        ElitistReplacement(elite_size=2),
    ]

    # Generate configurations
    config_id = 0
    for init in initializations:
        for sel in selections:
            for cx in crossovers:
                for mut in mutations:
                    for rep in replacements:
                        configs.append(
                            ExperimentConfig(
                                config_id=f"himmelblau_{config_id:03d}",
                                problem_type="Himmelblau",
                                problem_params={},
                                initialization=init,
                                selection=sel,
                                crossover=cx,
                                mutation=mut,
                                replacement=rep,
                                population_size=50,
                                max_evaluations=5000,
                            )
                        )
                        config_id += 1

    return configs


def create_tsp_configs() -> List[ExperimentConfig]:
    """
    Create experiment configurations for TSP problem.

    :return: List of experiment configurations
    :rtype: List[ExperimentConfig]
    """
    configs = []

    # Define a small TSP instance
    cities = [
        (0.0, 0.0),
        (1.0, 5.0),
        (5.0, 2.0),
        (6.0, 6.0),
        (8.0, 3.0),
        (3.0, 7.0),
        (4.0, 1.0),
        (7.0, 8.0),
    ]

    # Define operator variations
    initializations = [
        PermutationInitialization(),
        NeighborInitialization(k_best=3),
    ]
    selections = [TournamentSelection(tournament_size=3)]
    crossovers = [OrderCrossover()]
    mutations = [SwapMutation(mutation_rate=0.2)]
    replacements = [
        GenerationalReplacement(),
        ElitistReplacement(elite_size=2),
    ]

    # Generate configurations
    config_id = 0
    for init in initializations:
        for sel in selections:
            for cx in crossovers:
                for mut in mutations:
                    for rep in replacements:
                        configs.append(
                            ExperimentConfig(
                                config_id=f"tsp_{config_id:03d}",
                                problem_type="TSP",
                                problem_params={"cities": cities},
                                initialization=init,
                                selection=sel,
                                crossover=cx,
                                mutation=mut,
                                replacement=rep,
                                population_size=30,
                                max_evaluations=3000,
                            )
                        )
                        config_id += 1

    return configs


def main():
    """Main entry point for experiment execution."""
    # Create experiment runner
    runner = ExperimentRunner(
        output_dir="experiments/mono/results", max_workers=4, log_level=logging.INFO
    )

    # Create configurations for both problems
    himmelblau_configs = create_himmelblau_configs()
    tsp_configs = create_tsp_configs()
    all_configs = himmelblau_configs + tsp_configs

    # Define seeds for reproducibility
    seeds = list(range(10))  # 10 independent runs

    runner.logger.info(
        f"Total configurations: {len(all_configs)} "
        f"({len(himmelblau_configs)} Himmelblau + {len(tsp_configs)} TSP)"
    )

    # Run all experiments
    all_results = runner.run_experiments(all_configs, seeds)

    # Save individual configuration results
    for config in all_configs:
        runner.save_results_csv(config, all_results[config.config_id])

    # Save summary statistics
    runner.save_summary_statistics(all_results)

    runner.logger.info("All experiments completed successfully!")


if __name__ == "__main__":
    main()
