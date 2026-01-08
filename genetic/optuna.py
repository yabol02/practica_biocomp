import logging
import random
import time
from copy import deepcopy
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .algorithms import GeneticAlgorithmSO
from .configurations import ProblemConfig
from .factories import (crossover_factory, initialization_factory,
                        mutation_factory, replacement_factory)
from .problems import TSProblem
from .selection import *


class OptunaOptimizer:
    DEFAULT_SEARCH_SPACE = {
        "initialization_name": ["permutation", "neighbor", "diverse_nn"],
        "crossover_name": ["order", "pmx", "cycle", "edge_recombination"],
        "mutation_name": ["swap", "inversion", "scramble"],
        "replacement_name": ["elitist", "generational", "mu+lambda"],
        "population_size": {"min": 100, "max": 10_000, "step": 10},
        "mutation_rate": {"min": 0.0, "max": 1.0, "step": 0.05},
        "tournament_size": {"min": 1, "max": 50, "step": 2},
        "elite_size": {"min": 1, "max": 50, "step": 1},
    }

    def __init__(
        self,
        problem_config: ProblemConfig,
        problem: TSProblem,
        total_trials: int,
        prune_frequency: int = 10,
        prune_threshold: Optional[float] = None,
        conf_repetitions: int = 10,
        sampler=None,
        seed: int = 42,
        search_space: Optional[dict] = None,
        max_generations: Optional[int] = None,
        pruner: Optional[optuna.pruners.BasePruner] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Optimizador de hiperparámetros para algoritmos genéticos aplicado a problemas
        de optimización combinatoria (p. ej. TSP), basado en Optuna.

        Esta clase encapsula la lógica necesaria para:
            - Definir y validar un espacio de búsqueda de hiperparámetros del algoritmo genético.
            - Evaluar cada configuración de hiperparámetros mediante múltiples ejecuciones
            independientes del algoritmo genético, con control explícito de semillas para
            garantizar reproducibilidad.
            - Calcular métricas agregadas (media de la función objetivo) que reducen el ruido
            estocástico inherente a los algoritmos genéticos.
            - Integrar mecanismos de pruning de Optuna a nivel de repetición, permitiendo
            descartar configuraciones subóptimas de forma temprana y reducir
            significativamente el coste computacional.
            - Registrar de forma exhaustiva la traza experimental: cada ejecución individual,
            su configuración asociada, la pérdida obtenida, el genotipo resultante y el
            valor promedio por configuración, todo ello almacenado en un DataFrame
            estructurado para análisis posterior.

        :param problem_config: Configuracion del problema
        :type problem_config: ProblemConfig
        :param problem: Objeto que almacena la info relacionada con el problema
        :type problem: TSProblem
        :param total_trials: Trials que se le da a optuna para que realice la optimizacion
        :type total_trials: int
        :param prune_frequency: Numero de trials ejecutados antes de una ronda de pruning
        :type prune_frequency: int
        :param prune_threshold: (opcional) umbral informativo de loss
        :type prune_threshold: float
        :param conf_repetitions: Numero de veces que se evalua cada configuracion
        :type conf_repetitions: int
        :param sampler: Sampler de Optuna
        :param search_space: Espacio de busqueda custom
        :type search_space: dict
        :param seed: Seed para repetibilidad
        :type seed: int
        :param max_generations: Numero maximo de generaciones del GA
        :type max_generations: int
        """

        # ---------------- logging ----------------
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.logger.info("Inicializando OptunaOptimizer")

        # ---------------- core config ----------------
        self.config = problem_config
        self.problem = problem
        self.max_generations = max_generations

        self.search_space = self._merge_search_space(search_space)
        self._validate_search_space()

        self.total_trials = int(total_trials)
        self.prune_frequency = int(prune_frequency)
        self.prune_threshold = prune_threshold
        self.conf_repetitions = int(conf_repetitions)
        self.sampler = sampler
        self.seed = int(seed)

        self.pruner = pruner or optuna.pruners.MedianPruner(
            n_warmup_steps=max(1, self.conf_repetitions // 2)
        )

        self.study = optuna.create_study(
            direction="minimize",
            sampler=self.sampler,
            pruner=self.pruner,
        )

        # ---------------- trazabilidad ----------------
        self.results = []
        self.best_ever_loss = float("inf")
        self.best_ever_solution = None
        self.best_ever_avg_loss = float("inf")
        self.best_ever_conf: Dict[str, Any] = {}

        # DataFrame incremental (buffer + flush)
        self._records = []
        self.trials_df = pd.DataFrame()
        self._flush_every = 200

        self.logger.info("OptunaOptimizer inicializado correctamente")

    # ---------------- utilidades ----------------
    def _merge_search_space(self, search_space):
        """
        Unimos la configuracion por defecto con la que el usuario pasa
        """
        final_sp = deepcopy(self.DEFAULT_SEARCH_SPACE)
        if search_space:
            self._deep_update(final_sp, search_space)
        return final_sp

    def _deep_update(self, base: dict, override: dict):
        """
        Merge recursivo: actualiza `base` con valores de `override`.
        """
        for k, v in override.items():
            if k in base and isinstance(base[k], dict) and isinstance(v, dict):
                self._deep_update(base[k], v)
            else:
                base[k] = v

    def _validate_search_space(self):
        """
        Validación del espacio de búsqueda
        """
        self.logger.debug("Validando search_space")
        sp = self.search_space

        categorical_keys = [
            "initialization_name",
            "crossover_name",
            "mutation_name",
            "replacement_name",
        ]
        range_keys = ["population_size", "mutation_rate", "tournament_size"]

        for key in categorical_keys:
            if key not in sp or not isinstance(sp[key], list) or not sp[key]:
                raise ValueError(f"'{key}' debe ser una lista no vacía")

        for key in range_keys:
            conf = sp.get(key)
            if not isinstance(conf, dict):
                raise TypeError(f"'{key}' debe ser un diccionario")
            for sub in ("min", "max", "step"):
                if sub not in conf:
                    raise KeyError(f"'{key}' requiere '{sub}'")
            if conf["min"] > conf["max"]:
                raise ValueError(f"'{key}': min > max")

    def _run_genetic_algorithm(self, genetic_algorithm):
        """
        Wrapper que gestiona cómo ejecutar el GA
        """
        if self.max_generations:
            return genetic_algorithm.run(int(self.max_generations))
        return genetic_algorithm.run()

    # ---------------- DataFrame / logging helpers ----------------
    def _flush_records(self, force: bool = False):
        """
        Pasa de memoria a un dataframe
        """
        if not self._records:
            return
        if not force and len(self._records) < self._flush_every:
            return

        self.logger.debug("Flush de %d registros al DataFrame", len(self._records))
        df_new = pd.DataFrame(self._records)
        self.trials_df = (
            df_new
            if self.trials_df.empty
            else pd.concat([self.trials_df, df_new], ignore_index=True)
        )
        self._records = []

    def _record_single_evaluation(
        self,
        trial_number: int,
        config: dict,
        repeat_index,
        loss: float,
        genotype,
    ):
        """
        Almacena en memoria una evaluacion

        Parameters
        ----------
        trial_number : int
            Identificador del trial de Optuna al que pertenece la evaluación.
        config : dict
            Diccionario con la configuración de hiperparámetros evaluada
            (por ejemplo, tamaño de población, operadores genéticos, etc.).
        repeat_index : int or str
            Índice de la repetición dentro del trial. Puede ser un entero para
            ejecuciones individuales o un valor especial (p. ej. "avg") para
            indicar un registro agregado.
        loss : float
            Valor de la función objetivo (fitness o pérdida) obtenido en esta
            ejecución.
        genotype : Any
            Genotipo asociado a la mejor solución encontrada en la ejecución.
            Se almacena como una representación resumida para evitar un uso
            excesivo de memoria.

        """
        row = {
            "trial_number": trial_number,
            "repeat_index": repeat_index,
            "timestamp": time.time(),
            "loss": float(loss),
            "genotype_summary": str(genotype)[:512],
            **config,
        }
        self._records.append(row)
        self._flush_records()

    # ---------------- evaluación centralizada ----------------
    def _evaluate_configuration(
        self,
        trial,
        trial_number: int,
        config: dict,
        operators: dict,
        repeats: int,
    ):
        """
        Evalúa una configuración concreta de hiperparámetros del
        algoritmo genético dentro de un trial de Optuna.

        Para una configuración fija de hiperparámetros y operadores ya
        instanciados, este método ejecuta múltiples repeticiones independientes
        del algoritmo genético con diferentes semillas aleatorias. En cada
        repetición:
        - se ejecuta el algoritmo genético completo,
        - se registra el valor de la función objetivo (loss),
        - se almacena el genotipo asociado a la mejor solución encontrada.

        Durante la evaluación, el método reporta métricas parciales a Optuna,
        permitiendo la aplicación de poda temprana (pruning) si el rendimiento
        promedio intermedio no es competitivo. En caso de poda, los resultados
        acumulados hasta ese punto se persisten antes de interrumpir la
        ejecución.

        Al finalizar todas las repeticiones, se calcula y registra el valor
        promedio de la función objetivo, junto con el mejor genotipo observado
        en el conjunto de repeticiones.

        Parameters
        ----------
        trial : optuna.trial.Trial
            Trial activo de Optuna utilizado para reportar métricas intermedias
            y decidir la poda temprana de la evaluación.
        trial_number : int
            Identificador numérico del trial dentro del estudio de Optuna.
        config : dict
            Diccionario que describe la configuración de hiperparámetros del
            algoritmo genético evaluada en este trial (por ejemplo,
            population_size y nombres de operadores).
        operators : dict
            Diccionario con las instancias ya construidas de los operadores del
            algoritmo genético (initialization, selection, crossover, mutation
            y replacement).
        repeats : int
            Número de ejecuciones independientes del algoritmo genético
            realizadas para esta configuración, utilizado para estimar una
            métrica promedio robusta frente a la estocasticidad del método.

        Returns
        -------
        avg_loss : float
            Valor promedio de la función objetivo obtenido tras evaluar todas
            las repeticiones (o las ejecutadas antes de la poda).
        best_genotype : object
            Genotipo correspondiente a la mejor solución individual observada
            en cualquiera de las repeticiones.
        losses : list of float
            Lista con los valores de la función objetivo obtenidos en cada
            repetición individual.
        """
        self.logger.info(
            "Evaluando configuración trial=%s | %s",
            trial_number,
            config,
        )

        losses = []
        best_loss = float("inf")
        best_genotype = None

        for i in range(repeats):
            # Para reproducibilidad
            seed = self.seed + trial_number + i
            np.random.seed(seed)
            random.seed(seed)

            # Es necesario hacer una copia porque durante el desarrollo
            # del ag, el problema se modifica
            problem_copy = deepcopy(self.problem)

            ga = GeneticAlgorithmSO(
                problem=problem_copy,
                population_size=config["population_size"],
                initialization=operators["initialization"],
                selection=operators["selection"],
                crossover=operators["crossover"],
                mutation=operators["mutation"],
                replacement=operators["replacement"],
            )

            ga.initialize_random_state()
            result = self._run_genetic_algorithm(ga)

            loss = float(result.best_fitness)  # type: ignore
            genotype = result.best_solution

            self.logger.info(
                "Trial %s | iter %d/%d | loss=%.4f",
                trial_number,
                i + 1,
                repeats,
                loss,
            )
            print("\n")
            self._record_single_evaluation(
                trial_number=trial_number,
                config=config,
                repeat_index=i,
                loss=loss,
                genotype=genotype,
            )

            losses.append(loss)
            if loss < best_loss:
                best_loss = loss
                best_genotype = deepcopy(genotype)

            avg_partial = float(np.mean(losses))
            trial.report(avg_partial, i + 1)
            if trial.should_prune():
                self.logger.info(
                    "Trial %s podado en iteración %d (avg_loss=%.4f)",
                    trial_number,
                    i + 1,
                    avg_partial,
                )
                self._flush_records(force=True)
                raise optuna.exceptions.TrialPruned()

        avg_loss = float(np.mean(losses))
        self._record_single_evaluation(
            trial_number=trial_number,
            config=config,
            repeat_index="avg",
            loss=avg_loss,
            genotype=best_genotype,
        )

        self.logger.info(
            "Trial %s completado | avg_loss=%.4f",
            trial_number,
            avg_loss,
        )

        return avg_loss, best_genotype, losses

    # ---------------- objective ----------------
    def _objective(self, trial):
        """
        Función objetivo de Optuna para la optimización de hiperparámetros del
        algoritmo genético.

        Este método define el espacio de búsqueda de hiperparámetros, muestrea
        una configuración concreta mediante el objeto `trial` de Optuna y
        construye los operadores correspondientes del algoritmo genético.
        A continuación, delega la evaluación de dicha configuración a
        `_evaluate_configuration`, que ejecuta múltiples repeticiones
        independientes del algoritmo genético y devuelve una métrica promedio
        robusta.

        El valor que devuelve (pérdida media) es utilizado por Optuna para comparar
        y ordenar los trials. De forma adicional, este método mantiene un
        seguimiento global de:
        - la mejor solución individual observada en cualquier repetición,
        - la mejor configuración según el valor promedio de la función objetivo.

        Parameters
        ----------
        trial : optuna.trial.Trial
            Trial activo de Optuna utilizado para muestrear hiperparámetros y
            reportar métricas parciales durante la evaluación.

        Returns
        -------
        float
            Valor promedio de la función objetivo obtenido tras evaluar la
            configuración de hiperparámetros muestreada en este trial. Este
            valor es el que Optuna minimiza.
        """
        sp = self.search_space
        tn = trial.number

        initialization_info = trial.suggest_categorical(
            "initialization_name", sp["initialization_name"]
        )
        initialization = initialization_factory(initialization_info)

        tournament_size_info = sp["tournament_size"]
        tournament_size = trial.suggest_int(
            "tournament_size",
            tournament_size_info["min"],
            tournament_size_info["max"],
            step=tournament_size_info["step"],
        )
        selection = TournamentSelection(tournament_size)

        crossover_info = trial.suggest_categorical(
            "crossover_name", sp["crossover_name"]
        )
        crossover = crossover_factory(crossover_info)

        mr_info = sp["mutation_rate"]
        mutation_rate = trial.suggest_float(
            "mutation_rate", mr_info["min"], mr_info["max"], step=mr_info["step"]
        )
        mutation_info = trial.suggest_categorical("mutation_name", sp["mutation_name"])
        mutation = mutation_factory(mutation_info, mutation_rate=mutation_rate)

        replacement_name = trial.suggest_categorical(
            "replacement_name", sp["replacement_name"]
        )
        params = {}
        if replacement_name == "elitist":
            elite_conf = sp["elite_size"]
            params["elite_size"] = trial.suggest_int(
                "elite_size",
                elite_conf["min"],
                elite_conf["max"],
                step=elite_conf["step"],
            )
        replacement = replacement_factory(replacement_name, **params)

        population_size_info = sp["population_size"]
        population_size = trial.suggest_int(
            "population_size",
            population_size_info["min"],
            population_size_info["max"],
            step=population_size_info["step"],
        )

        config = {
            "population_size": population_size,
            "initialization": initialization_info,
            "tournament_size": tournament_size,
            "crossover": crossover_info,
            "mutation": mutation_info,
            "replacement": replacement_name,
            "mutation_rate": mutation_rate,
            **params,
        }

        operators = {
            "initialization": initialization,
            "selection": selection,
            "crossover": crossover,
            "mutation": mutation,
            "replacement": replacement,
        }

        avg_loss, best_genotype, losses = self._evaluate_configuration(
            trial=trial,
            trial_number=tn,
            config=config,
            operators=operators,
            repeats=self.conf_repetitions,
        )
        if min(losses) < self.best_ever_loss:
            self.best_ever_loss = min(losses)
            self.best_ever_solution = deepcopy(best_genotype)

        if avg_loss < self.best_ever_avg_loss:
            self.best_ever_avg_loss = avg_loss
            self.best_ever_conf = deepcopy(config)

        print("**************************************")
        return avg_loss

    # ---------------- ejecución ----------------
    def run(self, callbacks=None):
        """
        Ejecuta el proceso completo de optimización de hiperparámetros mediante Optuna.

        Este método gestiona la ejecución iterativa de trials de Optuna en rondas,
        permitiendo controlar el número total de evaluaciones y aplicar mecanismos
        de poda temprana definidos en la función objetivo. En cada ronda se ejecuta
        un número limitado de trials, tras lo cual se actualiza el estado interno
        del estudio y se registran los resultados intermedios.

        Al finalizar la optimización:
        - se fuerzan los volcados pendientes de la traza de evaluaciones,
        - se registra información sobre la mejor configuración encontrada,
        - y se devuelve el objeto `Study` de Optuna para análisis posterior.

        Parameters
        ----------
        callbacks : list of callable, optional
            Lista de callbacks de Optuna que se ejecutan tras la finalización de
            cada trial. Se utiliza, por ejemplo, para logging avanzado, early
            stopping externo o integración con herramientas de monitorización.
            Si no se proporciona, se utiliza una lista vacía.

        Returns
        -------
        optuna.study.Study
            Objeto `Study` de Optuna que contiene el histórico completo de trials,
            las configuraciones evaluadas y las métricas asociadas.
        """
        callbacks = callbacks or []
        executed = 0

        self.logger.info("Inicio optimización Optuna (%d trials)", self.total_trials)

        while executed < self.total_trials:
            remaining = self.total_trials - executed
            n_trials = min(self.prune_frequency, remaining)

            self.logger.info(
                "Ejecutando ronda: %d trials (restantes %d)",
                n_trials,
                remaining,
            )

            self.study.optimize(
                self._objective,
                n_trials=n_trials,
                callbacks=callbacks,
            )

            executed = len(self.study.trials)

        self._flush_records(force=True)

        self.logger.info("Optimización finalizada")
        if self.study.best_trial:
            self.logger.info(
                "Mejor avg_loss=%.4f | params=%s",
                self.study.best_value,
                self.study.best_params,
            )

        return self.study


class OptunaAnalyzer:
    """
    Clase para analizar exhaustivamente los resultados de un estudio de Optuna
    aplicado a un algoritmo genético. Permite:
      - Resumir métricas por trial (mean, std, cv, min, max)
      - Identificar top-K configuraciones
      - Evaluar estabilidad y robustez
      - Test estadístico de diferencias
      - Estimación de importancia de hiperparámetros
      - Análisis de genotipos y similitud estructural
      - Visualizaciones de distribución y superficies de respuesta
      - Análisis de interacciones entre hiperparámetros categóricos y continuos
    """

    def __init__(
        self,
        study: optuna.study.Study,
        parquet_path: str = "./data/optuna_ga_results.parquet",
    ):
        self.study = study
        self.parquet_path = parquet_path
        self.df = pd.read_parquet(parquet_path)
        self.df_avg = self.df[self.df["repeat_index"] == "avg"].copy()
        self.summary = None  # resumen agregado por trial_number

    # ------------------------- Estadísticas por trial -------------------------
    def summarize_trials(self):
        group = self.df[self.df["repeat_index"] != "avg"].groupby("trial_number")
        summary = (
            group["loss"]
            .agg(["mean", "median", "std", "min", "max", "count"])
            .reset_index()
        )
        params = self.df[self.df["repeat_index"] == 0].set_index("trial_number")[
            [
                "population_size",
                "initialization",
                "tournament_size",
                "crossover",
                "mutation",
                "replacement",
                "elite_size",
                "mutation_rate",
            ]
        ]
        summary = summary.join(params, on="trial_number")
        summary["cv"] = summary["std"] / (summary["mean"] + 1e-12)
        self.summary = summary
        return summary

    # ------------------------- Top-K y estabilidad -------------------------
    def top_k_trials(self, k=5, sort_by_cv=False):
        if self.summary is None:
            self.summarize_trials()
        if sort_by_cv:
            topk = self.summary.sort_values(["mean", "cv"]).head(k)
        else:
            topk = self.summary.sort_values("mean").head(k)
        return topk

    # ------------------------- Visualizaciones -------------------------
    def plot_box_by_category(self, cat_col="crossover", value_col="loss"):
        plt.figure(figsize=(10, 6))
        sns.boxplot(
            data=self.df[self.df["repeat_index"] != "avg"], x=cat_col, y=value_col
        )
        plt.title(f"Distribución de {value_col} por {cat_col}")
        plt.xticks(rotation=45)
        plt.show()

    def plot_heatmap_numeric(
        self,
        x_col="population_size",
        y_col="tournament_size",
        value_col="loss",
        aggfunc=np.mean,
    ):
        df_heat = self.df_avg.copy()
        df_heat["x_bin"] = pd.qcut(df_heat[x_col], q=6)
        pivot = df_heat.pivot_table(
            index="x_bin", columns=y_col, values=value_col, aggfunc=aggfunc
        )
        plt.figure(figsize=(10, 6))
        sns.heatmap(pivot, annot=True, fmt=".2f")
        plt.title(f"{value_col} promedio por {x_col} bin / {y_col}")
        plt.show()

    # ------------------------- Test estadístico -------------------------
    def compare_top_trials(self, top_n=2, test="mannwhitney"):
        topk = self.top_k_trials(k=top_n)
        t1 = self.df[
            (self.df.trial_number == topk.iloc[0].trial_number)
            & (self.df.repeat_index != "avg")
        ]
        t2 = self.df[
            (self.df.trial_number == topk.iloc[1].trial_number)
            & (self.df.repeat_index != "avg")
        ]

        if test == "mannwhitney":
            stat, p_value = stats.mannwhitneyu(
                t1.loss, t2.loss, alternative="two-sided"
            )
        elif test == "wilcoxon":
            stat, p_value = stats.wilcoxon(t1.loss.values, t2.loss.values)
        else:
            raise ValueError("test must be 'mannwhitney' or 'wilcoxon'")
        return stat, p_value

    # ------------------------- Feature importances -------------------------
    def estimate_param_importance(self):
        df_rep = self.df[self.df.repeat_index != "avg"].copy()
        cat_cols = ["initialization", "crossover", "mutation", "replacement"]
        num_cols = ["population_size", "tournament_size", "elite_size", "mutation_rate"]
        X = df_rep[cat_cols + num_cols]
        y = df_rep["loss"]

        preprocessor = ColumnTransformer(
            [
                ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ],
            remainder="passthrough",
        )

        pipe = Pipeline(
            [
                ("pre", preprocessor),
                ("rf", RandomForestRegressor(n_estimators=200, random_state=0)),
            ]
        )
        pipe.fit(X, y)
        feats = pipe.named_steps["pre"].get_feature_names_out()
        importances = pipe.named_steps["rf"].feature_importances_
        imp_df = pd.Series(importances, index=feats).sort_values(ascending=False)
        return imp_df

    # ------------------------- Genotype analysis -------------------------
    @staticmethod
    def genotype_to_edges(s):
        perm = np.fromstring(s.strip("[]"), sep=" ", dtype=int)
        edges = set()
        for a, b in zip(perm, np.roll(perm, -1)):
            edges.add((min(a, b), max(a, b)))
        return edges

    def genotype_similarity(self, trial_number):
        rows = self.df[
            (self.df.trial_number == trial_number) & (self.df.repeat_index != "avg")
        ]
        edges_list = [self.genotype_to_edges(s) for s in rows["genotype_summary"]]
        n = len(edges_list)
        if n < 2:
            return np.nan
        sims = []
        for i in range(n):
            for j in range(i + 1, n):
                e1, e2 = edges_list[i], edges_list[j]
                sims.append(len(e1 & e2) / len(e1 | e2))
        return np.mean(sims)

    # ------------------------- Análisis hiperparámetros -------------------------
    def analyze_hyperparams(self):
        """
        Analiza la influencia de los hiperparámetros sobre el loss:
          - Ignora repeat_index='avg'
          - Categóricos: boxplots y mean loss
          - Continuos: scatter + línea de tendencia, correlación
          - Interacciones pares: heatmaps, scatter coloreados o boxplots mixtos
        """
        df = self.df[self.df["repeat_index"] != "avg"].copy()
        cat_cols = ["initialization", "crossover", "mutation", "replacement"]
        num_cols = ["population_size", "tournament_size", "mutation_rate", "elite_size"]

        # ----------- Categóricos -----------
        print("### Efecto de parámetros categóricos sobre loss ###")
        for col in cat_cols:
            plt.figure(figsize=(8, 4))
            sns.boxplot(data=df, x=col, y="loss")
            plt.title(f"Distribución de loss según {col}")
            plt.show()
            mean_loss = df.groupby(col)["loss"].mean().sort_values()
            print(f"Promedio de loss por {col}:\n{mean_loss}\n")

        # ----------- Continuos -----------
        print("### Efecto de parámetros continuos sobre loss ###")
        for col in num_cols:
            plt.figure(figsize=(6, 4))
            sns.scatterplot(data=df, x=col, y="loss", alpha=0.6)
            sns.regplot(data=df, x=col, y="loss", scatter=False, color="red")
            plt.title(f"Tendencia de loss según {col}")
            plt.show()
            corr = df[col].corr(df["loss"])
            print(f"Correlación {col} vs loss: {corr:.3f}")

        # ----------- Interacciones de pares -----------
        print("### Interacciones entre pares de hiperparámetros ###")
        for i, col1 in enumerate(cat_cols + num_cols):
            for col2 in (cat_cols + num_cols)[i + 1 :]:
                if col1 in cat_cols and col2 in cat_cols:
                    pivot = df.pivot_table(
                        index=col1, columns=col2, values="loss", aggfunc=np.mean
                    )
                    plt.figure(figsize=(8, 5))
                    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="coolwarm")
                    plt.title(f"Mean loss: {col1} x {col2}")
                    plt.show()
                elif col1 in num_cols and col2 in num_cols:
                    plt.figure(figsize=(6, 5))
                    sns.scatterplot(
                        data=df, x=col1, y=col2, hue="loss", palette="viridis"
                    )
                    plt.title(f"Interacción: {col1} vs {col2} (color=loss)")
                    plt.show()
                else:
                    cont, cat = (col1, col2) if col1 in num_cols else (col2, col1)
                    plt.figure(figsize=(8, 4))
                    sns.boxplot(data=df, x=cat, y="loss")
                    sns.scatterplot(
                        data=df, x=cat, y=cont, hue="loss", palette="viridis", alpha=0.6
                    )
                    plt.title(f"Interacción: {cat} (categórico) vs {cont} (continuo)")
                    plt.show()
