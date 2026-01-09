"""
Fichero diseñado para la generación de resultados de la entrega.

Se puede elegir un problema y el fichero generará 10 experimentos y guardará los resultados.

En el notebook `entrega.ipynb` se analizarán los CSVs generados.
"""

import random

from genetic.algorithms import GeneticAlgorithmMO, GeneticAlgorithmSO
from genetic.configurations import ProblemConfig
from genetic.problems import (HimmelblauProblem, MOTSProblem, PymooProblem,
                              TSProblem)


def himmelblau_experiments(seeds: list[int]):
    from genetic.crossover import BlendCrossover
    from genetic.initialization import RandomInitialization
    from genetic.mutation import UniformMutation
    from genetic.replacement import GenerationalReplacement
    from genetic.selection import TournamentSelection

    for s in seeds:
        print(f"Running Himmelblau experiment with seed {s}")
        config = ProblemConfig(
            max_evaluations=3_500, seed=s, output_dir=f"results/himmelblau"
        )
        problem = HimmelblauProblem(config)
        initialization = RandomInitialization()
        selection = TournamentSelection(tournament_size=3)
        crossover = BlendCrossover(alpha=0.5)
        mutation = UniformMutation(mutation_rate=0.15, bounds=problem.get_bounds())
        replacement = GenerationalReplacement()

        ga = GeneticAlgorithmSO(
            problem=problem,
            population_size=50,
            initialization=initialization,
            selection=selection,
            crossover=crossover,
            mutation=mutation,
            replacement=replacement,
            print_interval=15,
        )
        res = ga.run()
        print(
            f"[{s}] Best fitness found: {res.best_fitness} with solution {res.best_solution}\n"
        )
        res.save_csv(f"{config.output_dir}/results_{s}.csv")


def tsp_experiments(seeds: list[int]):
    from experiments.mono.experiment_runner import CITIES_TSP
    from genetic.crossover import EdgeRecombinationCrossover
    from genetic.initialization import NeighborInitialization
    from genetic.mutation import InversionMutation
    from genetic.replacement import GenerationalReplacement
    from genetic.selection import TournamentSelection

    for s in seeds:
        print(f"Running TSP experiment with seed {s}")
        config = ProblemConfig(
            max_evaluations=1_000_000, seed=s, output_dir=f"results/tsp"
        )
        problem = TSProblem(config, CITIES_TSP)
        initialization = NeighborInitialization()
        selection = TournamentSelection(tournament_size=43)
        crossover = EdgeRecombinationCrossover()
        mutation = InversionMutation(mutation_rate=0.2)
        replacement = GenerationalReplacement()

        ga = GeneticAlgorithmSO(
            problem=problem,
            population_size=1000,
            initialization=initialization,
            selection=selection,
            crossover=crossover,
            mutation=mutation,
            replacement=replacement,
            print_interval=50,
        )
        res = ga.run()
        print(
            f"[{s}] Best fitness found: {res.best_fitness} with solution {res.best_solution}\n"
        )
        res.save_csv(f"{config.output_dir}/results_{s}.csv")


def zdt3_experiments(seeds: list[int]):
    from genetic.crossover import BlendCrossover
    from genetic.initialization import RandomInitialization
    from genetic.mutation import UniformMutation
    from genetic.replacement import GenerationalReplacement
    from genetic.selection import ParetoSelection

    for s in seeds:
        print(f"Running Pymoo experiment with seed {s}")
        config = ProblemConfig(
            max_evaluations=10_000, seed=s, output_dir=f"results/zdt3"
        )
        problem = PymooProblem(config, problem_name="ZDT3", n_var=30)
        initialization = RandomInitialization()
        selection = ParetoSelection()
        crossover = BlendCrossover(alpha=1)
        mutation = UniformMutation(mutation_rate=0.01, bounds=problem.get_bounds())
        replacement = GenerationalReplacement()

        ga = GeneticAlgorithmMO(
            problem=problem,
            population_size=50,
            initialization=initialization,
            selection=selection,
            crossover=crossover,
            mutation=mutation,
            replacement=replacement,
            print_interval=40,
        )
        res = ga.run()
        print(f"[{s}] Pareto front size: {len(res.pareto_front)}\n")
        res.save_csv(f"{config.output_dir}/results_{s}.csv")


def mw7_experiments(seeds: list[int]):
    from genetic.crossover import BlendCrossover
    from genetic.initialization import RandomInitialization
    from genetic.mutation import UniformMutation
    from genetic.replacement import GenerationalReplacement
    from genetic.selection import ParetoSelection

    for s in seeds:
        print(f"Running MW7 experiment with seed {s}")
        config = ProblemConfig(
            max_evaluations=10_000, seed=s, output_dir=f"results/mw7"
        )
        problem = PymooProblem(config, problem_name="MW7", n_var=30)
        initialization = RandomInitialization()
        selection = ParetoSelection()
        crossover = BlendCrossover(alpha=0.5)
        mutation = UniformMutation(mutation_rate=0.005, bounds=problem.get_bounds())
        replacement = GenerationalReplacement()

        ga = GeneticAlgorithmMO(
            problem=problem,
            population_size=50,
            initialization=initialization,
            selection=selection,
            crossover=crossover,
            mutation=mutation,
            replacement=replacement,
            print_interval=100,
        )
        res = ga.run()
        print(f"[{s}] Pareto front size: {len(res.pareto_front)}\n")
        res.save_csv(f"{config.output_dir}/results_{s}.csv")


def mw14_experiments(seeds: list[int]):
    from genetic.crossover import BlendCrossover
    from genetic.initialization import RandomInitialization
    from genetic.mutation import UniformMutation
    from genetic.replacement import GenerationalReplacement
    from genetic.selection import ParetoSelection

    for s in seeds:
        print(f"Running MW14 experiment with seed {s}")
        config = ProblemConfig(
            max_evaluations=10_000, seed=s, output_dir=f"results/mw14"
        )
        problem = PymooProblem(config, problem_name="MW14", n_var=30)
        initialization = RandomInitialization()
        selection = ParetoSelection()
        crossover = BlendCrossover(alpha=0.5)
        mutation = UniformMutation(mutation_rate=0.01, bounds=problem.get_bounds())
        replacement = GenerationalReplacement()

        ga = GeneticAlgorithmMO(
            problem=problem,
            population_size=50,
            initialization=initialization,
            selection=selection,
            crossover=crossover,
            mutation=mutation,
            replacement=replacement,
            print_interval=100,
        )
        res = ga.run()
        print(f"[{s}] Pareto front size: {len(res.pareto_front)}\n")
        res.save_csv(f"{config.output_dir}/results_{s}.csv")


def motsp_experiments(seeds: list[int]):
    from experiments.mono.experiment_runner import CITIES_TSP
    from genetic.crossover import EdgeRecombinationCrossover
    from genetic.initialization import DiverseNNInitialization
    from genetic.mutation import (CombinedMutation, InversionMutation,
                                  SwapMutation)
    from genetic.replacement import GenerationalReplacement
    from genetic.selection import ParetoSelection

    for s in seeds:
        print(f"Running MOTSP experiment with seed {s}")
        config = ProblemConfig(
            max_evaluations=100000, seed=42, output_dir="results/tspmo_final"
        )

        problem = MOTSProblem(config, CITIES_TSP, seed=s)

        initialization = DiverseNNInitialization()
        selection = ParetoSelection()
        crossover = EdgeRecombinationCrossover()
        mutation = CombinedMutation(
            [
                InversionMutation(mutation_rate=0.25),
                SwapMutation(mutation_rate=0.08),
            ]
        )
        replacement = GenerationalReplacement()

        ga = GeneticAlgorithmMO(
            problem=problem,
            population_size=200,
            initialization=initialization,
            selection=selection,
            crossover=crossover,
            mutation=mutation,
            replacement=replacement,
            print_interval=50,
        )
        res = ga.run()
        print(f"[{s}] Pareto front size: {len(res.pareto_front)}\n")
        res.save_csv(f"{config.output_dir}/results_{s}.csv")


tsp_experiments(list(range(10)))


def main():
    CHOSEN_EXPERIMENT = "tsp"
    gen = random.Random(0)
    seeds = [gen.randint(10_000, 99_999) for _ in range(10)]

    if CHOSEN_EXPERIMENT == "himmelblau":
        himmelblau_experiments(seeds)
    elif CHOSEN_EXPERIMENT == "tsp":
        tsp_experiments(seeds)
    elif CHOSEN_EXPERIMENT == "zdt3":
        zdt3_experiments(seeds)
    elif CHOSEN_EXPERIMENT == "mw7":
        mw7_experiments(seeds)
    elif CHOSEN_EXPERIMENT == "mw14":
        mw14_experiments(seeds)
    elif CHOSEN_EXPERIMENT == "motsp":
        motsp_experiments(seeds)
    else:
        print(
            f"El experimento {CHOSEN_EXPERIMENT} no es válido (los válidos son 'himmelblau', 'tsp', 'zdt3', 'mw7', 'mw14', 'motsp')."
        )


if __name__ == "__main__":
    main()
