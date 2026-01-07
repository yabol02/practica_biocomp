#!/usr/bin/env python
"""
Quick start script for running and analyzing experiments.

This script demonstrates a complete workflow:
1. Run a small set of experiments
2. Analyze the results
3. Generate visualizations
"""

import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(ROOT_DIR)

import random

import pandas as pd

from experiments.mono.analyze_experiments import ExperimentAnalyzer
from experiments.mono.experiment_runner import (ExperimentRunner,
                                                create_himmelblau_configs,
                                                create_tsp_configs)


def run_experiments():
    """Run a quick experiment for demonstration."""

    print("=" * 80)
    print("QUICK START: Running Sample Experiments")
    print("=" * 80)
    print()

    # Create runner with moderate parallelism
    runner = ExperimentRunner(
        output_dir="experiments/mono/results/exp_1",
        max_workers=4,
        use_processes=False,  # Set to True for faster execution
    )

    # Get first two configurations from each problem
    himmelblau_configs = create_himmelblau_configs()[:2]
    tsp_configs = create_tsp_configs()
    all_configs = tsp_configs

    # Running always the same 10 random seeds for reproducibility
    generator = random.Random(0)
    seeds = [generator.randint(10000000, 99999999) for _ in range(10)]

    print(f"Running {len(all_configs)} configurations with {len(seeds)} seeds each")
    print(f"Total experiments: {len(all_configs) * len(seeds)}")
    print()

    # Run experiments
    results = runner.run_experiments(all_configs, seeds)

    # Save results
    print("\nSaving results...")
    for config in all_configs:
        runner.save_results_summary(config, results[config.config_id])

    runner.save_summary_statistics(results)

    print(f"\n✓ Results saved to: {runner.output_dir}")
    print("✓ Experiment phase complete!")
    print()

    return runner.output_dir


def analyze_results(results_dir):
    """Analyze the experiment results."""

    print("=" * 80)
    print("QUICK START: Analyzing Results")
    print("=" * 80)
    print()

    analyzer = ExperimentAnalyzer(results_dir)

    # 1. Generate text report
    print("1. Generating comprehensive report...")
    report_file = os.path.join(results_dir, "quick_analysis_report.txt")
    analyzer.generate_report(output_file=report_file)
    print(f"   ✓ Report saved to: {report_file}")
    print()

    # 2. Get best configurations
    print("2. Top performing configurations:")
    table = analyzer.generate_comparison_table(
        metrics=["best_fitness_mean", "evals_to_best_mean", "execution_time_mean"],
        top_n=4,
    )
    print(table.to_string(index=False))
    print()

    # 3. Get all config IDs
    all_configs = [config["config_id"] for config in analyzer.configs]
    himmelblau_configs = [c for c in all_configs if "himmelblau" in c]
    tsp_configs = [c for c in all_configs if "tsp" in c]

    # 4. Plot convergence for best config from each problem
    if himmelblau_configs:
        print(f"3. Plotting convergence for {himmelblau_configs[0]}...")
        analyzer.plot_convergence(
            himmelblau_configs[0], metric="best_fitness", show_all_seeds=True, save=True
        )
        print(f"   ✓ Plot saved")
        print()

    if tsp_configs:
        print(f"4. Plotting convergence for {tsp_configs[0]}...")
        analyzer.plot_convergence(
            tsp_configs[0], metric="best_fitness", show_all_seeds=True, save=True
        )
        print(f"   ✓ Plot saved")
        print()

    # 5. Compare all configurations
    if len(all_configs) > 1:
        print("5. Comparing all configurations...")
        analyzer.plot_multiple_configs_convergence(
            all_configs, metric="best_fitness", save=True
        )
        print(f"   ✓ Comparison plot saved")
        print()

    # 6. Summary statistics by problem type
    if himmelblau_configs:
        print("6. Generating Himmelblau summary statistics...")
        analyzer.plot_summary_statistics(
            metric="best_fitness_mean", problem_type="himmelblau", save=True
        )
        print(f"   ✓ Summary plot saved")
        print()

    if tsp_configs:
        print("7. Generating TSP summary statistics...")
        analyzer.plot_summary_statistics(
            metric="best_fitness_mean", problem_type="tsp", save=True
        )
        print(f"   ✓ Summary plot saved")
        print()

    # 7. Fitness distribution
    if len(all_configs) >= 2:
        print("8. Plotting fitness distribution...")
        analyzer.plot_fitness_distribution(all_configs, save=True)
        print(f"   ✓ Distribution plot saved")
        print()

    # 8. Convergence speed analysis
    if len(all_configs) >= 2:
        print("9. Analyzing convergence speed...")
        analyzer.analyze_convergence_speed(all_configs, save=True)
        print(f"   ✓ Convergence analysis saved")
        print()

    print("✓ Analysis complete!")
    print(f"✓ All plots saved to: {analyzer.plots_dir}")
    print()


def show_summary(results_dir):
    """Show a quick summary of results."""

    print("=" * 80)
    print("QUICK SUMMARY")
    print("=" * 80)
    print()

    summary_file = os.path.join(results_dir, "summary_statistics.csv")

    if os.path.exists(summary_file):
        df = pd.read_csv(summary_file)

        print("Overall Performance:")
        print(f"  Total configurations: {len(df)}")
        print()

        # Best overall
        best_idx = df["best_fitness_mean"].idxmin()
        best_config = df.loc[best_idx]

        print("Best Configuration:")
        print(f"  Config ID: {best_config['config_id']}")
        print(f"  Mean Fitness: {best_config['best_fitness_mean']:.6f}")
        print(f"  Std Fitness: {best_config['best_fitness_std']:.6f}")
        print(f"  Avg Evaluations to Best: {best_config['evals_to_best_mean']:.0f}")
        print(f"  Avg Execution Time: {best_config['execution_time_mean']:.2f}s")
        print()

        # Himmelblau best
        himmelblau_df = df[df["config_id"].str.contains("himmelblau")]
        if not himmelblau_df.empty:
            best_h_idx = himmelblau_df["best_fitness_mean"].idxmin()
            best_h = himmelblau_df.loc[best_h_idx]
            print(f"Best Himmelblau Config: {best_h['config_id']}")
            print(f"  Mean Fitness: {best_h['best_fitness_mean']:.6f}")
            print()

        # TSP best
        tsp_df = df[df["config_id"].str.contains("tsp")]
        if not tsp_df.empty:
            best_t_idx = tsp_df["best_fitness_mean"].idxmin()
            best_t = tsp_df.loc[best_t_idx]
            print(f"Best TSP Config: {best_t['config_id']}")
            print(f"  Mean Fitness: {best_t['best_fitness_mean']:.6f}")
            print()

    print("=" * 80)
    print()


def main():
    """Main entry point."""
    print()
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "GENETIC ALGORITHM EXPERIMENT SUITE" + " " * 24 + "║")
    print("║" + " " * 28 + "Quick Start Guide" + " " * 33 + "║")
    print("╚" + "═" * 78 + "╝")
    print()

    # Step 1: Run experiments
    print("Step 1: Running Experiments")
    print("-" * 80)
    results_dir = run_experiments()

    # Step 2: Analyze results
    print("\nStep 2: Analyzing Results")
    print("-" * 80)
    analyze_results(results_dir)

    # Step 3: Show summary
    print("\nStep 3: Results Summary")
    print("-" * 80)
    show_summary(results_dir)

    # Final instructions
    print("=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print()
    print("1. Check the results directory:")
    print(f"   {results_dir}")
    print()
    print("2. View individual experiment histories:")
    print(f"   {results_dir}/histories/")
    print()
    print("3. Check generated plots:")
    print(f"   {results_dir}/plots/")
    print()
    print("4. Read the comprehensive report:")
    print(f"   {results_dir}/quick_analysis_report.txt")
    print()
    print("5. To run more extensive experiments:")
    print("   python -m experiments.mono.experiment_runner")
    print()
    print("6. To re-analyze existing results:")
    print("   python -m experiments.mono.analyze_experiments")
    print()
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
