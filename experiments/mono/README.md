# Mono-Objective Experiment Runner

This directory contains tools for running and managing mono-objective genetic algorithm experiments.

## Overview

The experiment runner allows you to:
- Execute genetic algorithm experiments across multiple configurations
- Run multiple independent trials with different random seeds
- Collect comprehensive metrics for analysis
- Execute experiments in parallel using multithreading
- Export results to CSV for further analysis

## Usage

### Basic Usage

To run all predefined experiments:

```bash
cd /path/to/practica_biocomp
python -m experiments.mono.experiment_runner
```

### Custom Configurations

You can create custom experiment configurations programmatically:

```python
from experiments.mono import ExperimentRunner, ExperimentConfig
from genetic.initialization import RandomInitialization
from genetic.selection import TournamentSelection
from genetic.crossover import BlendCrossover
from genetic.mutation import UniformMutation
from genetic.replacement import ElitistReplacement

# Create custom configuration
config = ExperimentConfig(
    config_id="custom_001",
    problem_type="Himmelblau",
    problem_params={},
    initialization=RandomInitialization(),
    selection=TournamentSelection(tournament_size=3),
    crossover=BlendCrossover(alpha=0.5),
    mutation=UniformMutation(mutation_rate=0.1, bounds=[(-5.0, 5.0), (-5.0, 5.0)]),
    replacement=ElitistReplacement(elite_size=2),
    population_size=50,
    max_evaluations=5000,
)

# Run experiments
runner = ExperimentRunner(output_dir="my_results", max_workers=4)
results = runner.run_experiments([config], seeds=list(range(10)))
runner.save_results_csv(config, results[config.config_id])
```

## Experiment Configuration

Each experiment configuration includes:

- **config_id**: Unique identifier for the configuration
- **problem_type**: Problem to optimize ("Himmelblau" or "TSP")
- **problem_params**: Problem-specific parameters (e.g., cities for TSP)
- **initialization**: Population initialization strategy
- **selection**: Parent selection operator
- **crossover**: Crossover operator
- **mutation**: Mutation operator
- **replacement**: Replacement/survival strategy
- **population_size**: Number of individuals in the population
- **max_evaluations**: Maximum number of fitness evaluations

## Supported Problems

### Himmelblau Function

A continuous optimization problem with multiple local minima:
- Bounds: x, y ∈ [-5, 5]
- Global minima: f(x,y) = 0 at four locations

### Traveling Salesman Problem (TSP)

A combinatorial optimization problem:
- Objective: Minimize tour distance visiting all cities
- Representation: Permutation of city indices

## Output Files

The experiment runner generates the following outputs:

### Individual Configuration Results

One CSV file per configuration (`{config_id}.csv`) containing:
- `config_id`: Configuration identifier
- `seed`: Random seed used
- `best_fitness`: Best fitness value achieved
- `evals_to_best`: Number of evaluations to reach best fitness
- `execution_time`: Total execution time in seconds
- `total_evaluations`: Total number of evaluations performed
- `final_generation`: Final generation number
- `convergence_rate`: Rate of convergence (evals_to_best / total_evaluations)
- Configuration parameters (initialization, selection, etc.)

### Summary Statistics

`summary_statistics.csv` containing aggregate statistics across all seeds:
- Mean, standard deviation, min, and max for all metrics
- Useful for comparing configurations

## Metrics

For each experiment run, the following metrics are computed:

1. **Best Fitness**: The best fitness value found during the run
2. **Evaluations to Best**: Number of evaluations required to find the best solution
3. **Execution Time**: Wall-clock time for the entire run
4. **Total Evaluations**: Total number of fitness evaluations performed
5. **Convergence Rate**: Ratio of evals_to_best to total_evaluations (lower is better)

## Parallelism

The experiment runner uses Python's `ThreadPoolExecutor` for parallel execution:
- Each thread runs one complete experiment (one configuration + one seed)
- Number of workers is configurable (default: 4)
- Experiments are executed asynchronously and results are collected as they complete

## Logging

The runner provides comprehensive logging:
- Experiment start and completion notifications
- Progress updates
- Error reporting
- Summary statistics

Logging can be configured via the `log_level` parameter:
```python
runner = ExperimentRunner(log_level=logging.DEBUG)  # More verbose
runner = ExperimentRunner(log_level=logging.WARNING)  # Less verbose
```

## Example Output

Sample CSV output for a configuration:

```csv
config_id,seed,best_fitness,evals_to_best,execution_time,total_evaluations,final_generation,convergence_rate,problem_type,initialization,selection,crossover,mutation,replacement,population_size,max_evaluations
himmelblau_001,0,0.0234,1543,2.45,5000,100,0.3086,Himmelblau,RandomInitialization,TournamentSelection,BlendCrossover,UniformMutation,ElitistReplacement,50,5000
himmelblau_001,1,0.0198,1678,2.52,5000,100,0.3356,Himmelblau,RandomInitialization,TournamentSelection,BlendCrossover,UniformMutation,ElitistReplacement,50,5000
...
```

## Notes

- The GA's internal logging is suppressed during experiments (print_interval=999999)
- Experiments log start and completion events for tracking
- All results are saved to CSV for easy analysis with pandas, R, or other tools
- Random seeds ensure reproducibility across runs
