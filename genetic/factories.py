"""
Factories para generar objetos mas facilmente en optuna y ahorrarnos codigo
"""

from typing import Any, Dict, Type

from .crossover import *
from .initialization import *
from .mutation import *
from .replacement import *


# --- Crossover Factory ---
def crossover_factory(name: str, **kwargs) -> Crossover:
    operators: Dict[str, Type[Crossover]] = {
        "order": OrderCrossover,
        "pmx": PMXCrossover,
        "cycle": CycleCrossover,
        "edge_recombination": EdgeRecombinationCrossover,
    }

    name = name.lower().replace(" ", "_")
    if name not in operators:
        raise ValueError(
            f"Crossover '{name}' no reconocido. Opciones: {list(operators.keys())}"
        )

    return operators[name](**kwargs)


# --- Initialization Factory ---
def initialization_factory(name: str, **kwargs) -> Initialization:
    operators: Dict[str, Type[Initialization]] = {
        "permutation": PermutationInitialization,
        "neighbor": NeighborInitialization,
        "diverse_nn": DiverseNNInitialization,
    }

    name = name.lower().replace(" ", "_")
    if name not in operators:
        raise ValueError(
            f"Initialization '{name}' no reconocido. Opciones: {list(operators.keys())}"
        )

    return operators[name](**kwargs)


# --- Mutation Factory ---
def mutation_factory(name: str, **kwargs) -> Mutation:
    operators: Dict[str, Type[Mutation]] = {
        "swap": SwapMutation,
        "inversion": InversionMutation,
        "scramble": ScrambleMutation,
    }

    name = name.lower().replace(" ", "_")
    if name not in operators:
        raise ValueError(
            f"Mutation '{name}' no reconocido. Opciones: {list(operators.keys())}"
        )

    return operators[name](**kwargs)


# --- Replacement Factory ---
def replacement_factory(name: str, **kwargs) -> Replacement:
    operators: Dict[str, Type[Replacement]] = {
        "generational": GenerationalReplacement,
        "elitist": ElitistReplacement,
        "mu+lambda": MuPlusLambdaReplacement,
    }

    name = name.lower().replace(" ", "_")
    if name not in operators:
        raise ValueError(
            f"Replacement '{name}' no reconocido. Opciones: {list(operators.keys())}"
        )

    return operators[name](**kwargs)
