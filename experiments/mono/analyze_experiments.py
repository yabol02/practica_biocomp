"""
Analysis script for experiment results.

Provides comprehensive analysis including:
- Statistical summaries
- Convergence visualizations
- Configuration comparisons
- Performance metrics
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class ExperimentAnalyzer:
    """Analyzer for genetic algorithm experiment results."""

    def __init__(self, results_dir: str = "experiments/mono/results"):
        """
        Initialize analyzer.

        :param results_dir: Directory containing experiment results
        """
        self.results_dir = Path(results_dir)
        self.history_dir = self.results_dir / "histories"
        self.summary_dir = self.results_dir / "summaries"
        self.metadata_dir = self.results_dir / "metadata"
        self.plots_dir = self.results_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)

        # Set style for better plots
        plt.style.use("dark_background")
        plt.rcParams["figure.figsize"] = (12, 6)

        self.configs = self._load_configurations()
        self.summary_stats = self._load_summary_statistics()

    def _load_configurations(self) -> Dict:
        """Load configuration metadata."""
        config_file = self.metadata_dir / "configurations.json"
        if config_file.exists():
            with open(config_file, "r") as f:
                return json.load(f)
        return {}

    def _load_summary_statistics(self) -> pd.DataFrame:
        """Load overall summary statistics."""
        summary_file = self.results_dir / "summary_statistics.csv"
        if summary_file.exists():
            return pd.read_csv(summary_file)
        return pd.DataFrame()

    def load_experiment_history(
        self, config_id: str, seed: int
    ) -> Optional[pd.DataFrame]:
        """
        Load history for a specific experiment run.

        :param config_id: Configuration identifier
        :param seed: Random seed
        :return: DataFrame with generation-by-generation history
        """
        filename = f"{config_id}_seed_{seed:03d}_history.csv"
        filepath = self.history_dir / filename

        if filepath.exists():
            return pd.read_csv(filepath)
        return None

    def load_all_histories_for_config(self, config_id: str) -> List[pd.DataFrame]:
        """
        Load all history files for a configuration.

        :param config_id: Configuration identifier
        :return: List of DataFrames, one per seed
        """
        pattern = f"{config_id}_seed_*_history.csv"
        files = sorted(self.history_dir.glob(pattern))

        histories = []
        for file in files:
            df = pd.read_csv(file)
            # Extract seed from filename
            seed = int(file.stem.split("_seed_")[1].split("_")[0])
            df["seed"] = seed
            histories.append(df)

        return histories

    def plot_convergence(
        self,
        config_id: str,
        metric: str = "best_fitness",
        show_all_seeds: bool = False,
        save: bool = True,
    ):
        """
        Plot convergence curves for a configuration.

        :param config_id: Configuration identifier
        :param metric: Metric to plot ('best_fitness', 'mean_fitness', etc.)
        :param show_all_seeds: Show individual seed runs or just mean±std
        :param save: Save plot to file
        """
        histories = self.load_all_histories_for_config(config_id)

        if not histories:
            print(f"No histories found for {config_id}")
            return

        fig, ax = plt.subplots(figsize=(12, 6))

        if show_all_seeds:
            # Plot each seed individually
            for df in histories:
                ax.plot(
                    df["generation"],
                    df[metric],
                    alpha=0.3,
                    linewidth=1,
                    label=f"Seed {df['seed'].iloc[0]}",
                )

        # Always plot mean with confidence interval
        all_data = pd.concat(histories)
        grouped = all_data.groupby("generation")[metric]

        mean = grouped.mean()
        std = grouped.std()

        ax.plot(mean.index, mean.values, linewidth=2, label="Mean", color="red")
        ax.fill_between(
            mean.index,
            mean.values - std.values,
            mean.values + std.values,
            alpha=0.2,
            color="red",
            label="±1 Std Dev",
        )

        ax.set_xlabel("Generation", fontsize=12)
        ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12)
        ax.set_title(f"Convergence: {config_id}", fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            filename = f"{config_id}_convergence_{metric}.png"
            plt.savefig(self.plots_dir / filename, dpi=300, bbox_inches="tight")
            print(f"Saved plot: {filename}")

        plt.show()

    def plot_multiple_configs_convergence(
        self,
        config_ids: List[str],
        metric: str = "best_fitness",
        save: bool = True,
    ):
        """
        Compare convergence curves across multiple configurations.

        :param config_ids: List of configuration identifiers
        :param metric: Metric to plot
        :param save: Save plot to file
        """
        fig, ax = plt.subplots(figsize=(14, 7))

        colors = plt.cm.tab10(np.linspace(0, 1, len(config_ids)))

        for config_id, color in zip(config_ids, colors):
            histories = self.load_all_histories_for_config(config_id)

            if not histories:
                continue

            all_data = pd.concat(histories)
            grouped = all_data.groupby("generation")[metric]
            mean = grouped.mean()

            ax.plot(mean.index, mean.values, linewidth=2, label=config_id, color=color)

        ax.set_xlabel("Generation", fontsize=12)
        ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12)
        ax.set_title("Configuration Comparison", fontsize=14, fontweight="bold")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            filename = f"comparison_{metric}.png"
            plt.savefig(self.plots_dir / filename, dpi=300, bbox_inches="tight")
            print(f"Saved plot: {filename}")

        plt.show()

    def plot_summary_statistics(
        self,
        metric: str = "best_fitness_mean",
        problem_type: Optional[str] = None,
        save: bool = True,
    ):
        """
        Plot summary statistics across configurations.

        :param metric: Metric to visualize
        :param problem_type: Filter by problem type (e.g., "himmelblau", "tsp")
        :param save: Save plot to file
        """
        if self.summary_stats.empty:
            print("No summary statistics available")
            return

        df = self.summary_stats.copy()

        # Filter by problem type if specified
        if problem_type:
            df = df[df["config_id"].str.contains(problem_type, case=False)]

        # Sort by metric
        df = df.sort_values(metric)

        fig, ax = plt.subplots(figsize=(12, max(6, len(df) * 0.3)))

        # Create horizontal bar plot
        bars = ax.barh(df["config_id"], df[metric])

        # Color by performance
        norm = plt.Normalize(df[metric].min(), df[metric].max())
        colors = plt.cm.RdYlGn_r(norm(df[metric].values))
        for bar, color in zip(bars, colors):
            bar.set_color(color)

        ax.set_xlabel(metric.replace("_", " ").title(), fontsize=12)
        ax.set_ylabel("Configuration", fontsize=12)
        ax.set_title(
            f"Configuration Performance: {metric.replace('_', ' ').title()}",
            fontsize=14,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3, axis="x")

        plt.tight_layout()

        if save:
            suffix = f"_{problem_type}" if problem_type else ""
            filename = f"summary_{metric}{suffix}.png"
            plt.savefig(self.plots_dir / filename, dpi=300, bbox_inches="tight")
            print(f"Saved plot: {filename}")

        plt.show()

    def generate_comparison_table(
        self,
        metrics: List[str] = None,
        problem_type: Optional[str] = None,
        top_n: int = 10,
    ) -> pd.DataFrame:
        """
        Generate a comparison table of configurations.

        :param metrics: List of metrics to include
        :param problem_type: Filter by problem type
        :param top_n: Number of top configurations to show
        :return: DataFrame with comparison
        """
        if self.summary_stats.empty:
            print("No summary statistics available")
            return pd.DataFrame()

        df = self.summary_stats.copy()

        if problem_type:
            df = df[df["config_id"].str.contains(problem_type, case=False)]

        if metrics is None:
            metrics = [
                "best_fitness_mean",
                "best_fitness_std",
                "evals_to_best_mean",
                "execution_time_mean",
                "convergence_rate_mean",
            ]

        # Select columns
        columns = ["config_id"] + metrics
        df = df[columns]

        # Sort by best fitness and take top N
        df = df.sort_values("best_fitness_mean").head(top_n)

        # Format numerical columns
        for col in metrics:
            if col in df.columns:
                df[col] = df[col].round(6)

        return df

    def plot_fitness_distribution(self, config_ids: List[str], save: bool = True):
        """
        Plot distribution of final best fitness across seeds.

        :param config_ids: List of configuration identifiers
        :param save: Save plot to file
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        data_to_plot = []
        labels = []

        for config_id in config_ids:
            histories = self.load_all_histories_for_config(config_id)

            if not histories:
                continue

            # Get final best fitness for each seed
            final_fitness = [df["best_fitness"].iloc[-1] for df in histories]
            data_to_plot.append(final_fitness)
            labels.append(config_id)

        if not data_to_plot:
            print("No data to plot")
            return

        # Create box plot
        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)

        # Color boxes
        colors = plt.cm.Set3(np.linspace(0, 1, len(data_to_plot)))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)

        ax.set_xlabel("Configuration", fontsize=12)
        ax.set_ylabel("Final Best Fitness", fontsize=12)
        ax.set_title(
            "Fitness Distribution Across Seeds", fontsize=14, fontweight="bold"
        )
        ax.grid(True, alpha=0.3, axis="y")
        plt.xticks(rotation=45, ha="right")

        plt.tight_layout()

        if save:
            filename = "fitness_distribution.png"
            plt.savefig(self.plots_dir / filename, dpi=300, bbox_inches="tight")
            print(f"Saved plot: {filename}")

        plt.show()

    def analyze_convergence_speed(self, config_ids: List[str], save: bool = True):
        """
        Analyze and visualize convergence speed across configurations.

        :param config_ids: List of configuration identifiers
        :param save: Save plot to file
        """
        convergence_data = []

        for config_id in config_ids:
            histories = self.load_all_histories_for_config(config_id)

            if not histories:
                continue

            for df in histories:
                # Find generation where best fitness is reached
                best_fitness = df["best_fitness"].min()
                gen_to_best = df[df["best_fitness"] == best_fitness]["generation"].iloc[
                    0
                ]

                convergence_data.append(
                    {
                        "config_id": config_id,
                        "generations_to_best": gen_to_best,
                        "best_fitness": best_fitness,
                    }
                )

        if not convergence_data:
            print("No convergence data available")
            return

        df = pd.DataFrame(convergence_data)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot 1: Average generations to best
        grouped = df.groupby("config_id")["generations_to_best"]
        means = grouped.mean().sort_values()

        ax1.barh(range(len(means)), means.values)
        ax1.set_yticks(range(len(means)))
        ax1.set_yticklabels(means.index)
        ax1.set_xlabel("Average Generations to Best", fontsize=12)
        ax1.set_title("Convergence Speed", fontsize=14, fontweight="bold")
        ax1.grid(True, alpha=0.3, axis="x")

        # Plot 2: Scatter of convergence vs quality
        for config_id in config_ids:
            config_data = df[df["config_id"] == config_id]
            ax2.scatter(
                config_data["generations_to_best"],
                config_data["best_fitness"],
                label=config_id,
                alpha=0.6,
                s=50,
            )

        ax2.set_xlabel("Generations to Best", fontsize=12)
        ax2.set_ylabel("Best Fitness", fontsize=12)
        ax2.set_title("Speed vs Quality Trade-off", fontsize=14, fontweight="bold")
        ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            filename = "convergence_analysis.png"
            plt.savefig(self.plots_dir / filename, dpi=300, bbox_inches="tight")
            print(f"Saved plot: {filename}")

        plt.show()

    def generate_report(self, output_file: Optional[str] = None):
        """
        Generate a comprehensive text report of all experiments.

        :param output_file: Path to save report (if None, prints to console)
        """
        lines = []
        lines.append("=" * 80)
        lines.append("EXPERIMENT RESULTS SUMMARY")
        lines.append("=" * 80)
        lines.append("")

        # Overall statistics
        if not self.summary_stats.empty:
            lines.append("OVERALL STATISTICS")
            lines.append("-" * 80)
            lines.append(f"Total configurations: {len(self.summary_stats)}")

            # Best configuration by fitness
            best_config = self.summary_stats.loc[
                self.summary_stats["best_fitness_mean"].idxmin()
            ]
            lines.append(f"\nBest Configuration (by mean fitness):")
            lines.append(f"  Config ID: {best_config['config_id']}")
            lines.append(f"  Mean Fitness: {best_config['best_fitness_mean']:.6f}")
            lines.append(f"  Std Fitness: {best_config['best_fitness_std']:.6f}")
            lines.append(
                f"  Mean Evals to Best: {best_config['evals_to_best_mean']:.0f}"
            )
            lines.append("")

            # Problem type breakdown
            for problem_type in ["himmelblau", "tsp"]:
                prob_configs = self.summary_stats[
                    self.summary_stats["config_id"].str.contains(
                        problem_type, case=False
                    )
                ]
                if not prob_configs.empty:
                    lines.append(f"\n{problem_type.upper()} PROBLEM")
                    lines.append("-" * 80)
                    lines.append(f"Number of configurations: {len(prob_configs)}")

                    best_prob = prob_configs.loc[
                        prob_configs["best_fitness_mean"].idxmin()
                    ]
                    lines.append(f"Best configuration: {best_prob['config_id']}")
                    lines.append(
                        f"  Best fitness: {best_prob['best_fitness_mean']:.6f}"
                    )
                    lines.append("")

        # Configuration details
        if self.configs:
            lines.append("\nCONFIGURATION DETAILS")
            lines.append("-" * 80)
            for config in self.configs[:5]:  # Show first 5
                lines.append(f"\nConfig ID: {config['config_id']}")
                lines.append(f"  Problem: {config['problem_type']}")
                lines.append(f"  Population Size: {config['population_size']}")
                lines.append(f"  Max Evaluations: {config['max_evaluations']}")
                lines.append(f"  Initialization: {config['initialization']}")
                lines.append(f"  Selection: {config['selection']}")
                lines.append(f"  Crossover: {config['crossover']}")
                lines.append(f"  Mutation: {config['mutation']}")
                lines.append(f"  Replacement: {config['replacement']}")

            if len(self.configs) > 5:
                lines.append(f"\n... and {len(self.configs) - 5} more configurations")

        lines.append("\n" + "=" * 80)

        report = "\n".join(lines)

        if output_file:
            with open(output_file, "w") as f:
                f.write(report)
            print(f"Report saved to {output_file}")
        else:
            print(report)


def main():
    """Example usage of the analyzer."""
    analyzer = ExperimentAnalyzer("experiments/mono/results")

    print("Generating analysis...")

    # Generate text report
    analyzer.generate_report(output_file="experiments/mono/results/analysis_report.txt")

    # Get all config IDs
    all_configs = [config["config_id"] for config in analyzer.configs]

    if not all_configs:
        print("No configurations found. Run experiments first.")
        return

    # Himmelblau configs
    himmelblau_configs = [c for c in all_configs if "himmelblau" in c]
    tsp_configs = [c for c in all_configs if "tsp" in c]

    # Plot convergence for first few configs
    if himmelblau_configs:
        print(f"\nPlotting convergence for {himmelblau_configs[0]}...")
        analyzer.plot_convergence(himmelblau_configs[0], show_all_seeds=True)

    if tsp_configs:
        print(f"\nPlotting convergence for {tsp_configs[0]}...")
        analyzer.plot_convergence(tsp_configs[0], show_all_seeds=True)

    # Compare configurations
    if len(himmelblau_configs) > 1:
        print("\nComparing Himmelblau configurations...")
        analyzer.plot_multiple_configs_convergence(himmelblau_configs[:3])

    if len(tsp_configs) > 1:
        print("\nComparing TSP configurations...")
        analyzer.plot_multiple_configs_convergence(tsp_configs[:3])

    # Summary statistics
    print("\nPlotting summary statistics...")
    analyzer.plot_summary_statistics("best_fitness_mean", problem_type="himmelblau")
    analyzer.plot_summary_statistics("best_fitness_mean", problem_type="tsp")

    # Fitness distribution
    if himmelblau_configs:
        print("\nPlotting fitness distribution...")
        analyzer.plot_fitness_distribution(himmelblau_configs[:4])

    # Convergence speed analysis
    if len(all_configs) >= 2:
        print("\nAnalyzing convergence speed...")
        analyzer.analyze_convergence_speed(all_configs[:4])

    # Generate comparison table
    print("\nTop 10 configurations:")
    table = analyzer.generate_comparison_table(top_n=10)
    print(table.to_string(index=False))

    print("\nAnalysis complete! Check the plots directory for visualizations.")


if __name__ == "__main__":
    main()
