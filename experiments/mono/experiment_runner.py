"""
Experiment runner for mono-objective genetic algorithm optimization.

Enhanced version with:
- Improved performance using ProcessPoolExecutor
- Complete history tracking per experiment
- Better organization of results
"""

import csv
import json
import logging
import time
from concurrent.futures import (ProcessPoolExecutor, ThreadPoolExecutor,
                                as_completed)
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from genetic.algorithms import GeneticAlgorithmSO
from genetic.configurations import ProblemConfig
from genetic.crossover import BlendCrossover, Crossover, OrderCrossover
from genetic.initialization import (Initialization, NeighborInitialization,
                                    PermutationInitialization,
                                    RandomInitialization)
from genetic.mutation import Mutation, SwapMutation, UniformMutation
from genetic.problems import HimmelblauProblem, TSProblem
from genetic.replacement import (ElitistReplacement, GenerationalReplacement,
                                 MuPlusLambdaReplacement, Replacement)
from genetic.selection import Selection, TournamentSelection

CITIES_TSP = (
    (0.8744058793117488, 0.30207179832703746),
    (0.6901572276474558, 0.8203362566663311),
    (0.7793458478083026, 0.09728979746351918),
    (0.7483358474304422, 0.4563419482458657),
    (0.4556164527790729, 0.12863149772605698),
    (0.1837236688553453, 0.23773749495644914),
    (0.806151832654163, 0.3288366900529254),
    (0.6222702185205219, 0.6019431085670109),
    (0.21886861543988367, 0.6980104036528004),
    (0.9706096813038716, 0.054871876245520146),
    (0.13881496219356027, 0.048489530185603646),
    (0.08511086492841424, 0.9123075505436051),
    (0.45436952164297595, 0.4731540740573398),
    (0.8986352255038935, 0.5116579750591849),
    (0.9539248970461742, 0.7530784036308166),
    (0.6321102985156373, 0.13575967095580344),
    (0.31130918313591205, 0.8545457319320597),
    (0.025897479791937017, 0.7854526665590987),
    (0.31414095045738066, 0.051421702103483846),
    (0.2739561013272712, 0.1970964027924862),
    (0.22762328443470214, 0.627414210742153),
    (0.44920042912037383, 0.18835427573438068),
    (0.2793138301174244, 0.9103956650199608),
    (0.005128451667189338, 0.4655718085045638),
    (0.9222718335261915, 0.8613103704784812),
    (0.49352498279326595, 0.28652416807644077),
    (0.5560886356057152, 0.034564925228364185),
    (0.0048939219846825255, 0.8635350555457212),
    (0.7824462381771976, 0.09355415747424733),
    (0.2973635133076785, 0.037580527218207815),
    (0.659068210692151, 0.5602321147981546),
    (0.43549418718677035, 0.6241289474799954),
    (0.25580657871099755, 0.519726015381631),
    (0.5950780852425872, 0.3193476592023645),
    (0.15763623266223903, 0.18210618406479095),
    (0.3434173147566393, 0.32992255868483356),
    (0.06451313705472006, 0.12606114349173192),
    (0.23212286400737547, 0.3945559700824013),
    (0.955507285594692, 0.02296933408938162),
    (0.7459955912073274, 0.1385374827130177),
    (0.46365765315595475, 0.6132297411379687),
    (0.20196772698059162, 0.046358747430397584),
    (0.2980093518214506, 0.36041197123227897),
    (0.7533459401199348, 0.6610562232191091),
    (0.323433933838161, 0.5068337090138902),
    (0.062138830138457, 0.7177455044259695),
    (0.47467832142010535, 0.17692718986260936),
    (0.2415634321550142, 0.500852093941769),
    (0.3441878827094532, 0.2604351258948506),
    (0.5445252142594251, 0.7295590645635646),
    (0.25391391610860914, 0.7641244059389684),
    (0.2630560480211699, 0.010670642241309536),
    (0.2092202940037915, 0.7415775443241935),
    (0.0031474687633804566, 0.6974574785842637),
    (0.4952329615559806, 0.9311775894780175),
    (0.1978488658972426, 0.8872693649066419),
    (0.1705364411799497, 0.8846808176103278),
    (0.7634962513234275, 0.4358457309207533),
    (0.9259752834140167, 0.0056226911773098465),
    (0.22090405091557253, 0.14500040130773872),
    (0.4636205151855679, 0.19249574424477278),
    (0.8057632881682811, 0.250168118157608),
    (0.6785236695642043, 0.39119477830554505),
    (0.4837264739606879, 0.23045417711558214),
    (0.8786721718893166, 0.015853186240832207),
    (0.1263309724872509, 0.9185321014517236),
    (0.24969404380600335, 0.9600467657522086),
    (0.5886119367943214, 0.10948123668473264),
    (0.13373527735293878, 0.5336853876899833),
    (0.5642696742735989, 0.9801384844337597),
    (0.646401605512568, 0.3338636388089157),
    (0.05603526365085032, 0.3149530712641142),
    (0.387141492676661, 0.5792852545912592),
    (0.8634070619751958, 0.8206820685065667),
    (0.4662474858565825, 0.32647667516764145),
    (0.01801973379815025, 0.12175434785651695),
    (0.6218125565554127, 0.07860046445444713),
    (0.6573938922925594, 0.5023092428939168),
    (0.2837581848186974, 0.02576704782244421),
    (0.3044774374252185, 0.7310313422019327),
    (0.33131673439274434, 0.693896434962478),
    (0.8523694048117944, 0.2805224723356823),
    (0.6200019812032047, 0.03989814578273998),
    (0.5437394088109648, 0.3258031129949792),
    (0.4386079860370994, 0.6142798815693052),
    (0.7430581106897977, 0.49260737229959284),
    (0.3913628378102181, 0.3185830750836698),
    (0.9065498976196814, 0.02381958527497441),
    (0.6874236611131453, 0.18642068590773597),
    (0.6946411585627993, 0.36991609629423117),
    (0.3784063098306413, 0.01018756575849844),
    (0.43806374889930155, 0.2268325586117821),
    (0.7411632641820634, 0.35513151640087337),
    (0.3884706603234591, 0.34043302784114093),
    (0.08674110453583683, 0.4491174987265194),
    (0.9794883193315694, 0.11207520940235804),
    (0.7496156619797683, 0.23845386610980834),
    (0.4423888847621813, 0.17212722421686055),
    (0.2276570035864689, 0.6515924491551327),
    (0.836939257954547, 0.0499933914625984),
)
    

@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""

    config_id: str
    problem_type: str
    problem_params: Dict
    initialization: Initialization
    selection: Selection
    crossover: Crossover
    mutation: Mutation
    replacement: Replacement
    population_size: int
    max_evaluations: int

    def to_dict(self) -> Dict:
        """Convert configuration to dictionary for CSV output."""
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
    """Results from a single experiment run with complete history."""

    config_id: str
    seed: int
    best_fitness: float
    evals_to_best: int
    execution_time: float
    total_evaluations: int
    final_generation: int
    convergence_rate: float
    # Complete history tracking
    best_history: List[float]
    mean_history: List[float]
    std_history: List[float]
    worst_history: List[float]

    def to_dict(self) -> Dict:
        """Convert result to dictionary (excluding histories for summary)."""
        return {
            "config_id": self.config_id,
            "seed": self.seed,
            "best_fitness": self.best_fitness,
            "evals_to_best": self.evals_to_best,
            "execution_time": self.execution_time,
            "total_evaluations": self.total_evaluations,
            "final_generation": self.final_generation,
            "convergence_rate": self.convergence_rate,
        }


def _run_single_experiment_worker(
    config_dict: Dict, seed: int
) -> Tuple[str, int, ExperimentResult]:
    """
    Worker function for running a single experiment.
    Must be a top-level function for ProcessPoolExecutor pickling.
    """
    # Reconstruct objects from dict
    config_id = config_dict["config_id"]
    problem_type = config_dict["problem_type"]
    problem_params = config_dict["problem_params"]
    
    # Recreate operators
    init_class = config_dict["initialization"]["class"]
    init_params = config_dict["initialization"]["params"]
    initialization = globals()[init_class](**init_params)
    
    sel_class = config_dict["selection"]["class"]
    sel_params = config_dict["selection"]["params"]
    selection = globals()[sel_class](**sel_params)
    
    cx_class = config_dict["crossover"]["class"]
    cx_params = config_dict["crossover"]["params"]
    crossover = globals()[cx_class](**cx_params)
    
    mut_class = config_dict["mutation"]["class"]
    mut_params = config_dict["mutation"]["params"]
    mutation = globals()[mut_class](**mut_params)
    
    rep_class = config_dict["replacement"]["class"]
    rep_params = config_dict["replacement"]["params"]
    replacement = globals()[rep_class](**rep_params)
    
    population_size = config_dict["population_size"]
    max_evaluations = config_dict["max_evaluations"]
    
    # Run experiment
    start_time = time.time()
    
    problem_config = ProblemConfig(max_evaluations=max_evaluations, seed=seed)
    
    if problem_type == "Himmelblau":
        problem = HimmelblauProblem(problem_config)
    elif problem_type == "TSP":
        problem = TSProblem(problem_config, **problem_params)
    else:
        raise ValueError(f"Unknown problem type: {problem_type}")
    
    ga = GeneticAlgorithmSO(
        problem=problem,
        population_size=population_size,
        initialization=initialization,
        selection=selection,
        crossover=crossover,
        mutation=mutation,
        replacement=replacement,
        print_interval=999999,
    )
    
    result = ga.run()
    execution_time = time.time() - start_time
    
    convergence_rate = (
        result.best_solution_found_on / result.evaluations_used
        if result.evaluations_used > 0 and result.best_solution_found_on >= 0
        else 0.0
    )
    
    experiment_result = ExperimentResult(
        config_id=config_id,
        seed=seed,
        best_fitness=result.best_fitness,
        evals_to_best=result.best_solution_found_on,
        execution_time=execution_time,
        total_evaluations=result.evaluations_used,
        final_generation=len(result.history),
        convergence_rate=convergence_rate,
        best_history=problem.best_history,
        mean_history=problem.mean_history,
        std_history=problem.std_history,
        worst_history=problem.worst_history,
    )
    
    return config_id, seed, experiment_result


class ExperimentRunner:
    """Manages execution of mono-objective GA experiments with enhanced tracking."""

    def __init__(
        self,
        output_dir: str = "experiments/mono/results",
        max_workers: int = 4,
        log_level: int = logging.INFO,
        use_processes: bool = False,
    ):
        """
        Initialize experiment runner.

        :param output_dir: Directory to save experiment results
        :param max_workers: Maximum number of parallel workers
        :param log_level: Logging level
        :param use_processes: Use ProcessPoolExecutor instead of ThreadPoolExecutor
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.history_dir = self.output_dir / "histories"
        self.history_dir.mkdir(exist_ok=True)
        
        self.summary_dir = self.output_dir / "summaries"
        self.summary_dir.mkdir(exist_ok=True)
        
        self.metadata_dir = self.output_dir / "metadata"
        self.metadata_dir.mkdir(exist_ok=True)
        
        self.max_workers = max_workers
        self.use_processes = use_processes

        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self.logger = logging.getLogger(__name__)

    def _config_to_serializable(self, config: ExperimentConfig) -> Dict:
        """Convert ExperimentConfig to a fully serializable dictionary."""
        return {
            "config_id": config.config_id,
            "problem_type": config.problem_type,
            "problem_params": config.problem_params,
            "initialization": {
                "class": config.initialization.__class__.__name__,
                "params": config.initialization.__dict__,
            },
            "selection": {
                "class": config.selection.__class__.__name__,
                "params": config.selection.__dict__,
            },
            "crossover": {
                "class": config.crossover.__class__.__name__,
                "params": config.crossover.__dict__,
            },
            "mutation": {
                "class": config.mutation.__class__.__name__,
                "params": config.mutation.__dict__,
            },
            "replacement": {
                "class": config.replacement.__class__.__name__,
                "params": config.replacement.__dict__,
            },
            "population_size": config.population_size,
            "max_evaluations": config.max_evaluations,
        }

    def run_experiments(
        self, configs: List[ExperimentConfig], seeds: List[int]
    ) -> Dict[str, List[ExperimentResult]]:
        """Run experiments for all configurations and seeds."""
        all_results: Dict[str, List[ExperimentResult]] = {
            config.config_id: [] for config in configs
        }

        total_experiments = len(configs) * len(seeds)
        self.logger.info(
            f"Starting {total_experiments} experiments "
            f"({len(configs)} configs × {len(seeds)} seeds)"
        )

        # Save metadata for all configurations
        self._save_configs_metadata(configs)

        # Prepare serializable tasks for ProcessPoolExecutor
        if self.use_processes:
            tasks = [
                (self._config_to_serializable(config), seed)
                for config in configs
                for seed in seeds
            ]
            
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_task = {
                    executor.submit(_run_single_experiment_worker, config_dict, seed): (
                        config_dict["config_id"],
                        seed,
                    )
                    for config_dict, seed in tasks
                }
                
                self._collect_results(future_to_task, all_results, total_experiments)
        else:
            # Use threads (original behavior, simpler for debugging)            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_task = {
                    executor.submit(self._run_single_experiment, config, seed): (
                        config,
                        seed,
                    )
                    for config in configs
                    for seed in seeds
                }
                
                self._collect_results(future_to_task, all_results, total_experiments)

        return all_results

    def _run_single_experiment(
        self, config: ExperimentConfig, seed: int
    ) -> ExperimentResult:
        """Execute a single experiment (for ThreadPoolExecutor)."""
        self.logger.info(f"Starting: config_id={config.config_id}, seed={seed}")
        
        start_time = time.time()
        problem_config = ProblemConfig(max_evaluations=config.max_evaluations, seed=seed)
        
        if config.problem_type == "Himmelblau":
            problem = HimmelblauProblem(problem_config)
        elif config.problem_type == "TSP":
            problem = TSProblem(problem_config, **config.problem_params)
        else:
            raise ValueError(f"Unknown problem type: {config.problem_type}")
        
        ga = GeneticAlgorithmSO(
            problem=problem,
            population_size=config.population_size,
            initialization=config.initialization,
            selection=config.selection,
            crossover=config.crossover,
            mutation=config.mutation,
            replacement=config.replacement,
            print_interval=999999,
        )
        
        result = ga.run()
        execution_time = time.time() - start_time
        
        convergence_rate = (
            result.best_solution_found_on / result.evaluations_used
            if result.evaluations_used > 0 and result.best_solution_found_on >= 0
            else 0.0
        )
        
        experiment_result = ExperimentResult(
            config_id=config.config_id,
            seed=seed,
            best_fitness=result.best_fitness,
            evals_to_best=result.best_solution_found_on,
            execution_time=execution_time,
            total_evaluations=result.evaluations_used,
            final_generation=len(result.history),
            convergence_rate=convergence_rate,
            best_history=problem.best_history,
            mean_history=problem.mean_history,
            std_history=problem.std_history,
            worst_history=problem.worst_history,
        )
        
        self.logger.info(
            f"Completed: config_id={config.config_id}, seed={seed}, "
            f"best={result.best_fitness:.6f}, time={execution_time:.2f}s"
        )
        
        return experiment_result

    def _collect_results(self, future_to_task, all_results, total_experiments):
        """Collect results as they complete."""
        completed = 0
        for future in as_completed(future_to_task):
            task_info = future_to_task[future]
            
            try:
                if self.use_processes:
                    config_id, seed, result = future.result()
                else:
                    result = future.result()
                    config_id = result.config_id
                    seed = result.seed
                
                all_results[config_id].append(result)
                
                # Save history immediately after each experiment
                self._save_experiment_history(result)
                
                completed += 1
                self.logger.info(f"Progress: {completed}/{total_experiments} completed")
                
            except Exception as exc:
                if self.use_processes:
                    config_id, seed = task_info
                else:
                    config, seed = task_info
                    config_id = config.config_id
                
                self.logger.error(
                    f"Experiment failed: config_id={config_id}, seed={seed}, error={exc}"
                )

    def _save_configs_metadata(self, configs: List[ExperimentConfig]) -> None:
        """Save configuration metadata to JSON."""
        metadata_file = self.metadata_dir / "configurations.json"
        
        configs_data = []
        for config in configs:
            config_data = config.to_dict()
            # Add parameter details
            config_data["parameters"] = {
                "initialization": config.initialization.__dict__,
                "selection": config.selection.__dict__,
                "crossover": config.crossover.__dict__,
                "mutation": config.mutation.__dict__,
                "replacement": config.replacement.__dict__,
            }
            configs_data.append(config_data)
        
        with open(metadata_file, "w") as f:
            json.dump(configs_data, f, indent=2, default=str)
        
        self.logger.info(f"Saved configurations metadata to {metadata_file}")

    def _save_experiment_history(self, result: ExperimentResult) -> None:
        """Save complete history for a single experiment run."""
        filename = f"{result.config_id}_seed_{result.seed:03d}_history.csv"
        filepath = self.history_dir / filename
        
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "generation",
                "best_fitness",
                "mean_fitness",
                "std_fitness",
                "worst_fitness",
            ])
            
            # Write generation by generation
            max_len = max(
                len(result.best_history),
                len(result.mean_history),
                len(result.std_history),
                len(result.worst_history),
            )
            
            for gen in range(max_len):
                row = [gen]
                row.append(result.best_history[gen] if gen < len(result.best_history) else "")
                row.append(result.mean_history[gen] if gen < len(result.mean_history) else "")
                row.append(result.std_history[gen] if gen < len(result.std_history) else "")
                row.append(result.worst_history[gen] if gen < len(result.worst_history) else "")
                writer.writerow(row)

    def save_results_summary(
        self, config: ExperimentConfig, results: List[ExperimentResult]
    ) -> None:
        """Save summary results across all seeds for a configuration."""
        if not results:
            self.logger.warning(f"No results to save for config_id={config.config_id}")
            return

        output_file = self.summary_dir / f"{config.config_id}_summary.csv"
        
        with open(output_file, "w", newline="") as f:
            result_fields = list(results[0].to_dict().keys())
            config_fields = [k for k in config.to_dict().keys() if k != "config_id"]
            fieldnames = result_fields + config_fields
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            config_dict = config.to_dict()
            for result in results:
                row = result.to_dict()
                for key in config_fields:
                    row[key] = config_dict[key]
                writer.writerow(row)
        
        self.logger.info(f"Saved summary to {output_file}")

    def save_summary_statistics(
        self, all_results: Dict[str, List[ExperimentResult]]
    ) -> None:
        """Save aggregate statistics across all configurations."""
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
                
                summary = {
                    "config_id": config_id,
                    "n_runs": len(results),
                    "best_fitness_mean": np.mean([r.best_fitness for r in results]),
                    "best_fitness_std": np.std([r.best_fitness for r in results]),
                    "best_fitness_min": np.min([r.best_fitness for r in results]),
                    "best_fitness_max": np.max([r.best_fitness for r in results]),
                    "evals_to_best_mean": np.mean([r.evals_to_best for r in results]),
                    "evals_to_best_std": np.std([r.evals_to_best for r in results]),
                    "execution_time_mean": np.mean([r.execution_time for r in results]),
                    "execution_time_std": np.std([r.execution_time for r in results]),
                    "convergence_rate_mean": np.mean([r.convergence_rate for r in results]),
                    "convergence_rate_std": np.std([r.convergence_rate for r in results]),
                }
                
                writer.writerow(summary)
        
        self.logger.info(f"Saved summary statistics to {summary_file}")


def create_himmelblau_configs() -> List[ExperimentConfig]:
    """Create experiment configurations for Himmelblau problem."""
    configs = []
    
    initializations = [RandomInitialization()]
    selections = [TournamentSelection(tournament_size=3)]
    crossovers = [BlendCrossover(alpha=0.5)]
    mutations = [UniformMutation(mutation_rate=0.1, bounds=[(-5.0, 5.0), (-5.0, 5.0)])]
    replacements = [
        GenerationalReplacement(),
        ElitistReplacement(elite_size=2),
        MuPlusLambdaReplacement()
    ]
    population_sizes = [25, 50, 100]
    
    config_id = 0
    for init in initializations:
        for sel in selections:
            for cx in crossovers:
                for mut in mutations:
                    for rep in replacements:
                        for pop_size in population_sizes:
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
                                    population_size=pop_size,
                                    max_evaluations=3500,
                                )
                            )
                        config_id += 1
    
    return configs


def create_tsp_configs() -> List[ExperimentConfig]:
    """Create experiment configurations for TSP problem."""
    configs = []
    
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
        MuPlusLambdaReplacement()
    ]
    population_sizes = [10, 20, 50, 100, 200]
    
    config_id = 0
    for init in initializations:
        for sel in selections:
            for cx in crossovers:
                for mut in mutations:
                    for rep in replacements:
                        for pop_size in population_sizes:
                            configs.append(
                                ExperimentConfig(
                                    config_id=f"tsp_{config_id:03d}",
                                    problem_type="TSP",
                                    problem_params={"cities": CITIES_TSP},
                                    initialization=init,
                                    selection=sel,
                                    crossover=cx,
                                    mutation=mut,
                                    replacement=rep,
                                    population_size=pop_size,
                                    max_evaluations=1_000_000,
                                )
                            )
                            config_id += 1
    
    return configs


def main():
    """Main entry point for experiment execution."""
    runner = ExperimentRunner(
        output_dir="experiments/mono/results",
        max_workers=4,
        log_level=logging.INFO,
        use_processes=False,  # Set to True for ProcessPoolExecutor
    )
    
    himmelblau_configs = create_himmelblau_configs()
    tsp_configs = create_tsp_configs()
    all_configs = himmelblau_configs + tsp_configs
    
    seeds = list(range(10))
    
    runner.logger.info(
        f"Total configurations: {len(all_configs)} "
        f"({len(himmelblau_configs)} Himmelblau + {len(tsp_configs)} TSP)"
    )
    
    all_results = runner.run_experiments(all_configs, seeds)
    
    for config in all_configs:
        runner.save_results_summary(config, all_results[config.config_id])
    
    runner.save_summary_statistics(all_results)
    
    runner.logger.info("All experiments completed successfully!")


if __name__ == "__main__":
    main()