import random

from genetic.configurations import ProblemConfig
from genetic.problems import HimmelblauProblem

config = ProblemConfig(max_evaluations=3500, seed=42, output_dir="results/himmelblau")
problem = HimmelblauProblem(config)

# Simulate some evaluations
for i in range(10):
    solution = [random.uniform(-5, 5), random.uniform(-5, 5)]
    fitness = problem.evaluate(solution)
    problem.update_history(problem.best_fitness)
    print(f"Eval {i+1}: f={fitness:.4f}, best={problem.best_fitness:.4f}")

# Save results
result = problem.get_result()
result.save_csv("results/himmelblau/history.csv")
print(f"\nBest found: f={result.best_fitness:.6f} at {result.best_solution}")
