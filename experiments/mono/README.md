# Mono-Objective Experiment Runner - Enhanced Version

This directory contains tools for running and managing mono-objective genetic algorithm experiments with complete history tracking and advanced analysis capabilities.

## 📁 Directory Structure

```
experiments/mono/results/
├── histories/                    # Individual experiment histories
│   ├── himmelblau_000_seed_000_history.csv
│   ├── himmelblau_000_seed_001_history.csv
│   └── ...
├── summaries/                    # Per-config summaries
│   ├── himmelblau_000_summary.csv
│   ├── tsp_000_summary.csv
│   └── ...
├── metadata/                     # Configuration metadata
│   └── configurations.json
├── plots/                        # Generated visualizations
│   ├── himmelblau_000_convergence_best_fitness.png
│   ├── comparison_best_fitness.png
│   └── ...
├── summary_statistics.csv        # Overall statistics
└── analysis_report.txt          # Text report
```

## 🔧 Usage

### Running Experiments

**Basic usage:**
```bash
python -m experiments.mono.experiment_runner
```

**With multiprocessing (faster for CPU-bound experiments):**
```python
from experiments.mono.experiment_runner import ExperimentRunner

runner = ExperimentRunner(
    output_dir="experiments/mono/results",
    max_workers=8,
    use_processes=True  # Enable multiprocessing
)
```

### Analyzing Results

**Run complete analysis:**
```bash
python -m experiments.mono.analyze_experiments
```

**Custom analysis in Python:**
```python
from experiments.mono.analyze_experiments import ExperimentAnalyzer

analyzer = ExperimentAnalyzer("experiments/mono/results")

# Plot convergence for a specific configuration
analyzer.plot_convergence("himmelblau_000", show_all_seeds=True)

# Compare multiple configurations
analyzer.plot_multiple_configs_convergence([
    "himmelblau_000",
    "himmelblau_001"
])

# Generate comparison table
table = analyzer.generate_comparison_table(top_n=10)
print(table)

# Create comprehensive report
analyzer.generate_report(output_file="my_report.txt")
```

## 📊 Output Files

### History Files

Each experiment run produces a CSV file: `{config_id}_seed_{seed:03d}_history.csv`

**Format:**
```csv
generation,best_fitness,mean_fitness,std_fitness,worst_fitness
0,15.234,25.123,5.234,45.123
1,12.456,22.345,4.567,42.345
...
```

**Columns:**
- `generation`: Generation number
- `best_fitness`: Best fitness in the population
- `mean_fitness`: Mean fitness across population
- `std_fitness`: Standard deviation of fitness
- `worst_fitness`: Worst fitness in the population

### Summary Files

Per-configuration summary: `{config_id}_summary.csv`

**Contains:**
- Results from all seeds for the configuration
- Best fitness, evaluations to best, execution time
- Configuration parameters

### Summary Statistics

Overall summary: `summary_statistics.csv`

**Aggregated metrics across all seeds:**
- Mean, std, min, max of best fitness
- Mean evaluations to best solution
- Mean execution time
- Convergence rate statistics

### Configuration Metadata

JSON file with complete configuration details: `configurations.json`

**Includes:**
- All operator parameters
- Problem configuration
- Population size and evaluation budget

## 📈 Analysis Features

### 1. Convergence Plots

Visualize how fitness improves over generations:

```python
analyzer.plot_convergence(
    config_id="himmelblau_000",
    metric="best_fitness",
    show_all_seeds=True  # Show individual runs + mean±std
)
```

**Metrics available:**
- `best_fitness`: Best individual in population
- `mean_fitness`: Population average
- `std_fitness`: Population diversity
- `worst_fitness`: Worst individual

### 2. Configuration Comparison

Compare multiple configurations on the same plot:

```python
analyzer.plot_multiple_configs_convergence(
    config_ids=["himmelblau_000", "himmelblau_001", "tsp_000"],
    metric="best_fitness"
)
```

### 3. Summary Statistics

Visualize performance across all configurations:

```python
analyzer.plot_summary_statistics(
    metric="best_fitness_mean",
    problem_type="himmelblau"  # Filter by problem
)
```

### 4. Fitness Distribution

Box plots showing fitness distribution across seeds:

```python
analyzer.plot_fitness_distribution(
    config_ids=["himmelblau_000", "himmelblau_001"]
)
```

### 5. Convergence Speed Analysis

Analyze trade-off between speed and quality:

```python
analyzer.analyze_convergence_speed(
    config_ids=["himmelblau_000", "himmelblau_001"]
)
```

### 6. Comparison Tables

Generate ranked tables of configurations:

```python
table = analyzer.generate_comparison_table(
    metrics=[
        "best_fitness_mean",
        "best_fitness_std",
        "evals_to_best_mean",
        "execution_time_mean"
    ],
    problem_type="himmelblau",
    top_n=10
)
```

### 7. Comprehensive Reports

Generate detailed text reports:

```python
analyzer.generate_report(
    output_file="experiments/mono/results/report.txt"
)
```

## 🎯 Example Workflow

```python
# 1. Run experiments
from experiments.mono.experiment_runner import (
    ExperimentRunner,
    create_himmelblau_configs,
    create_tsp_configs
)

runner = ExperimentRunner(
    output_dir="my_experiments",
    max_workers=4,
    use_processes=False  # True for CPU-intensive tasks
)

configs = create_himmelblau_configs() + create_tsp_configs()
seeds = list(range(10))

results = runner.run_experiments(configs, seeds)

# Save summaries
for config in configs:
    runner.save_results_summary(config, results[config.config_id])
runner.save_summary_statistics(results)

# 2. Analyze results
from experiments.mono.analyze_experiments import ExperimentAnalyzer

analyzer = ExperimentAnalyzer("my_experiments")

# Get best configuration
table = analyzer.generate_comparison_table(top_n=5)
best_config = table.iloc[0]["config_id"]

# Plot its convergence
analyzer.plot_convergence(best_config, show_all_seeds=True)

# Compare top 3 configurations
top_3 = table.head(3)["config_id"].tolist()
analyzer.plot_multiple_configs_convergence(top_3)

# Generate comprehensive report
analyzer.generate_report(output_file="analysis.txt")
```

## ⚡ Performance Tips

### When to Use Multiprocessing

**Use `use_processes=True` when:**
- Experiments are CPU-intensive (complex fitness functions)
- You have multiple CPU cores available
- Running many long experiments

**Use `use_processes=False` when:**
- Experiments are I/O bound
- Debugging (easier stack traces)
- Objects are not easily picklable

### Optimizing Experiments

1. **Batch processing**: Group similar configurations
2. **Incremental analysis**: Analyze results as they complete
3. **Parallel workers**: Match to CPU core count
4. **Memory management**: Process large result sets in chunks

```python
# Example: Process results incrementally
runner = ExperimentRunner(max_workers=8)

for batch in config_batches:
    results = runner.run_experiments(batch, seeds)
    
    # Analyze immediately
    analyzer = ExperimentAnalyzer(runner.output_dir)
    analyzer.generate_comparison_table()
```

## 🐛 Troubleshooting

### Common Issues

**Problem:** Multiprocessing hangs or fails  
**Solution:** Set `use_processes=False` or ensure all objects are picklable

**Problem:** Memory errors with large experiments  
**Solution:** Reduce `max_workers` or process in smaller batches

**Problem:** Plots not showing  
**Solution:** Use `plt.show()` or save with `save=True`

**Problem:** Missing history files  
**Solution:** Check that experiments completed successfully (check logs)

## 📝 Notes

- All plots are saved at 300 DPI for publication quality
- History files use minimal storage (only numeric data)
- Metadata is JSON for easy parsing by other tools
- CSV format is compatible with R, Excel, pandas, etc.

## 🔬 Advanced Usage

### Custom Metrics

Add custom metrics to analysis:

```python
# Load history
df = analyzer.load_experiment_history("himmelblau_000", seed=0)

# Calculate custom metric
df["improvement_rate"] = df["best_fitness"].diff().abs()

# Plot
plt.plot(df["generation"], df["improvement_rate"])
plt.xlabel("Generation")
plt.ylabel("Improvement Rate")
plt.show()
```

### Batch Analysis

Analyze multiple result directories:

```python
import pandas as pd

results = []
for exp_dir in ["exp1", "exp2", "exp3"]:
    analyzer = ExperimentAnalyzer(exp_dir)
    stats = analyzer.summary_stats
    stats["experiment"] = exp_dir
    results.append(stats)

combined = pd.concat(results)
# Now analyze combined results
```

### Export for Papers

```python
# Generate all plots for publication
analyzer = ExperimentAnalyzer("experiments/mono/results")

configs = [c["config_id"] for c in analyzer.configs]

# High-quality plots
for config in configs:
    analyzer.plot_convergence(config, save=True)

analyzer.plot_multiple_configs_convergence(configs, save=True)
analyzer.plot_summary_statistics(save=True)

# LaTeX-ready table
table = analyzer.generate_comparison_table(top_n=10)
print(table.to_latex(index=False))
```