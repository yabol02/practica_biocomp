#!/usr/bin/env python
"""
Example script demonstrating how to run mono-objective experiments.

This script shows how to:
1. Create custom experiment configurations
2. Run experiments with the ExperimentRunner
3. Save and analyze results
"""

import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(ROOT_DIR)

from experiments.mono.experiment_runner import (ExperimentRunner,
                                                create_himmelblau_configs,
                                                create_tsp_configs)


def run_minimal_example():
    """Run a minimal example with a single configuration."""
    print("Running minimal example...")

    # Create runner
    runner = ExperimentRunner(
        output_dir="experiments/mono/results/minimal", max_workers=2
    )

    # Get one configuration from each problem type
    himmelblau_configs = create_himmelblau_configs()[:1]
    tsp_configs = create_tsp_configs()[:1]
    configs = himmelblau_configs + tsp_configs

    # Run with 3 seeds
    seeds = [42, 43, 44]

    results = runner.run_experiments(configs, seeds)

    # Save results
    for config in configs:
        runner.save_results_csv(config, results[config.config_id])

    runner.save_summary_statistics(results)

    print(f"\nResults saved to: {runner.output_dir}")


def run_full_experiments():
    """Run all predefined experiments."""
    print("Running full experiment suite...")

    # Create runner with more workers for parallel execution
    runner = ExperimentRunner(output_dir="experiments/mono/results/full", max_workers=4)

    # Get all configurations
    all_configs = create_himmelblau_configs() + create_tsp_configs()

    # Run with 10 independent seeds
    seeds = list(range(10))

    print(
        f"\nRunning {len(all_configs)} configurations with {len(seeds)} seeds each..."
    )
    print(f"Total experiments: {len(all_configs) * len(seeds)}")

    results = runner.run_experiments(all_configs, seeds)

    # Save results
    for config in all_configs:
        runner.save_results_csv(config, results[config.config_id])

    runner.save_summary_statistics(results)

    print(f"\nResults saved to: {runner.output_dir}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "full":
        run_full_experiments()
    else:
        print("Usage:")
        print("  python run_experiments.py          # Run minimal example")
        print("  python run_experiments.py full     # Run full experiment suite")
        print()
        run_minimal_example()
