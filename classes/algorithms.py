from .crossover import Crossover
from .individual import RealIndividual
from .initialization import Initialization
from .mutation import Mutation
from .population import Population
from .problems import SingleObjectiveProblem
from .replacement import Replacement
from .results import SingleObjectiveResult
from .selection import Selection


class GeneticAlgorithmSO:
    """Basic Genetic Algorithm implementation."""

    def __init__(
        self,
        problem: SingleObjectiveProblem,
        population_size: int,
        initialization: Initialization,
        selection: Selection,
        crossover: Crossover,
        mutation: Mutation,
        replacement: Replacement,
    ):
        """
        Initialize Genetic Algorithm.

        :param problem: Problem to optimize
        :type problem: SingleObjectiveProblem
        :param population_size: Number of individuals
        :type population_size: int
        :param initialization: Initialization strategy
        :type initialization: Initialization
        :param selection: Selection operator
        :type selection: Selection
        :param crossover: Crossover operator
        :type crossover: Crossover
        :param mutation: Mutation operator
        :type mutation: Mutation
        :param replacement: Replacement strategy
        :type replacement: Replacement
        """
        self.problem = problem
        self.population_size = population_size
        self.initialization = initialization
        self.selection = selection
        self.crossover = crossover
        self.mutation = mutation
        self.replacement = replacement

    def run(self) -> SingleObjectiveResult:
        """
        Execute genetic algorithm.

        :return: Optimization result
        :rtype: SingleObjectiveResult
        """
        # Initialize population (may use evaluation!)
        bounds = self.problem.get_bounds()
        # TODO: Add defaults args to the Initialization class and use here, e.g., if no initialization is provided, use something like:
        # population = Population(
        #     individuals=[
        #         RealIndividual(bounds=bounds) for _ in range(self.population_size)
        #     ],
        #     minimize=True,
        # )
        population = self.initialization.initialize(
            self.population_size, bounds, self.problem
        )

        generation = 0

        # Main loop
        while not self.problem.reached_budget():
            # Evaluate population
            population.evaluate_population(self.problem.evaluate)

            # Track best
            best = population.best_individual
            self.problem.update_history(best.fitness)

            # Print progress
            if generation % 10 == 0:
                print(
                    f"Gen {generation}: Evals={self.problem.evaluations_count}/{self.problem.config.max_evaluations}, "
                    f"Best={best.fitness:.6f}, Solution={[f'{x:.4f}' for x in best.genotype]}"
                )

            # Check budget
            if self.problem.reached_budget():
                break

            parents = population

            # Selection
            selected = self.selection.select(population, self.population_size)

            # Crossover
            offspring = self.crossover.cross(selected)

            # Mutation
            offspring = self.mutation.mutate(offspring)

            # Replace population
            population = self.replacement.replace(parents, offspring)
            generation += 1

        # Final evaluation
        population.evaluate_population(self.problem.evaluate)
        best = population.best_individual
        self.problem.update_history(best.fitness)

        return self.problem.get_result()

    def __str__(self) -> str:
        return f"GA(pop={self.population_size})"

    def __repr__(self) -> str:
        return (
            f"GeneticAlgorithm("
            f"pop={self.population_size}, "
            f"sel={self.selection.__class__.__name__}, "
            f"cx={self.crossover.__class__.__name__}, "
            f"mut={self.mutation.__class__.__name__}, "
            f"rep={self.replacement.__class__.__name__}"
            f")"
        )