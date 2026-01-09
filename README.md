<a id="readme-top"></a>
<div align="center">
  <h1 align="center">🧬 Práctica de Computación Evolutiva y Bioinspirada</h1>

  [![Python Version](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
  [![Dependency Manager](https://img.shields.io/badge/uv-astral-purple?logo=python&logoColor=white)](https://docs.astral.sh/uv/)
  [![Code Style Black](https://img.shields.io/badge/Code%20Style-Black-black)](https://github.com/psf/black)
  [![Imports isort](https://img.shields.io/badge/Imports-isort-blue)](https://pycqa.github.io/isort/)
  [![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
  [![Status](https://img.shields.io/badge/Status-En_Desarrollo-orange)]()
  <p align="center">
    Implementación de Algoritmos Genéticos y Estrategias Evolutivas 
    <br />
    <strong>Máster en Aprendizaje Automático y Datos Masivos</strong>
    <br />
    <br />
    <a href="#requisitos-previos">Requisitos</a> •
    <a href="#instalación">Instalación</a> •
    <a href="#metodologia-y-experimentos">Experimentación</a> •
    <a href="#estructura-del-proyecto">Estructura</a> •
    <a href="#ejecucion-del-codigo">Uso</a> •
    <a href="#roadmap">Roadmap</a> •
    <a href="#autores">Autores</a> •
    <a href="#licencia">Licencia</a>
  </p>
</div>

---

<a id="sobre-el-proyecto"></a>
## ℹ️ Sobre el Proyecto

Este repositorio contiene la entrega práctica de la asignatura **Computación Evolutiva y Bioinspirada** del **Máster en Aprendizaje Automático y Datos Masivos**. El objetivo principal es el desarrollo, análisis y comparativa de diferentes algoritmos bioinspirados aplicados a problemas de optimización mono-objetivo y multi-objetivo.

### 📋 Alineación con el enunciado

- Implementación de GA **mono-objetivo** desde cero para Himmelblau y TSP.
- Implementación de GA **multi-objetivo** desde cero con soporte de dominancia de Pareto.
- **Comparativas** con PSO, Scipy, ACO y NSGA-II de Pymoo.
- Uso de **Pymoo** para problemas benchmark (ZDT3, MW7, MW14).
- **Múltiples ejecuciones** independientes con semillas distintas para reducir ruido estocástico.
- **Generación de CSV** con formato especificado (historial de convergencia, frentes de Pareto).

### ✨ Características principales

- **Framework modular de Algoritmos Genéticos** implementado desde cero con arquitectura orientada a objetos extensible y operadores genéticos intercambiables.
- **Problemas de optimización**: Himmelblau y TSP (mono-objetivo); ZDT1, ZDT3, MW7 y TSP MO (multi-objetivo). Integración con benchmarks de Pymoo.
- **Sistema de experimentación**: Tracking de evaluaciones, exportación a CSV, notebooks reproducibles y comparación con PSO, Scipy, ACO y NSGA-II.
- **Multi-objetivo**: Dominancia de Pareto, NSGA-II, métricas de calidad y visualización de frentes de Pareto.

<p align="right">(<a href="#readme-top">Volver arriba</a>)</p>

<a id="empezando"></a>
## 🔥 Empezando

Sigue estos pasos para levantar el entorno de desarrollo localmente.

### Requisitos previos

Este proyecto ha sido desarrollado usando **Python (>3.11)**.

> [!IMPORTANT]
> Para la gestión de dependencias y entornos virtuales se utiliza **[uv](https://docs.astral.sh/uv/)**, un gestor de paquetes extremadamente rápido escrito en Rust.
> 
> Si no dispones de `uv`, instálalo ejecutando:
> ```bash
> # En macOS/Linux
> curl -LsSf https://astral.sh/uv/install.sh | sh
>
> # En Windows
> powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
> ```

### Dependencias principales

El proyecto utiliza las siguientes librerías:

| Librería | Versión | Propósito |
|----------|---------|-----------|
| `numpy` | ≥2.3.5 | Operaciones numéricas y matriciales |
| `scipy` | ≥1.16.3 | Métodos de optimización tradicionales |
| `matplotlib` | ≥3.10.8 | Visualización de resultados |
| `pandas` | ≥2.3.3 | Manipulación de datos y exportación CSV |
| `pymoo` | ≥0.6.1.6 | Benchmark problems multi-objetivo (ZDT, MW) |
| `pyswarms` | ≥1.3.0 | Particle Swarm Optimization (PSO) |
| `optuna` | ≥4.6.0 | Optimización de hiperparámetros |

> [!TIP]
> Todas las dependencias se instalan automáticamente con el comando `uv sync`.

### Instalación

Una vez instalado `uv`, la configuración es automática. Desde la raíz del repositorio:

1. Clona el repositorio:
   ```bash
   git clone https://github.com/yabol02/practica_biocomp.git
   cd practica_biocomp
   ```

2. Sincroniza el entorno:
    ```bash
    uv sync
    ```
    Este comando creará el virtual environment (`.venv`) e instalará todas las librerías exactas definidas en el `uv.lock`.

3. Verifica la instalación:
    ```bash
    source .venv/bin/activate  # Linux/macOS
    # o .venv\Scripts\activate en Windows
    
    python -c "from genetic.algorithms import GeneticAlgorithmSO; print('✓ Instalación correcta')"
    ```

### ⚡ Quick Start

Prueba rápidamente el framework con un ejemplo mínimo:

```python
from genetic.algorithms import GeneticAlgorithmSO
from genetic.configurations import ProblemConfig
from genetic.problems import HimmelblauProblem
from genetic.initialization import RandomInitialization
from genetic.selection import TournamentSelection
from genetic.crossover import BlendCrossover
from genetic.mutation import UniformMutation
from genetic.replacement import GenerationalReplacement

# Configurar y ejecutar
config = ProblemConfig(max_evaluations=3500, seed=42, output_dir="results/himmelblau")
problem = HimmelblauProblem(config)

ga = GeneticAlgorithmSO(
    problem=problem,
    population_size=30,
    initialization=RandomInitialization(),
    selection=TournamentSelection(tournament_size=3),
    crossover=BlendCrossover(alpha=0.5),
    mutation=UniformMutation(mutation_rate=0.3, bounds=problem.get_bounds()),
    replacement=GenerationalReplacement(elite_size=2)
)

result = ga.run()
print(f"✓ Mejor fitness: {result.best_fitness:.6f}")
print(f"✓ Solución: {result.best_solution}")
```

Para explorar los experimentos completos, consulta los notebooks en el directorio [`notebooks/`](./notebooks/).


<p align="right">(<a href="#readme-top">Volver arriba</a>)</p>

<a id="metodologia-y-experimentos"></a>
## 🧪 Metodología y experimentos

Todos los resultados se obtienen a partir de múltiples ejecuciones independientes con distintas semillas aleatorias, asegurando reproducibilidad y reducción del ruido estocástico inherente a los algoritmos genéticos. Se utilizan presupuestos de evaluaciones fijados por el enunciado para cada problema. Los resultados se analizan mediante estadísticas básicas (media, desviación, mínimos) y gráficos de convergencia y frentes de Pareto. En el módulo [`experiments/`](./experiments/) se encuentra el código para lanzar múltiples experimentos realizando un grid search básico y almacenando resultados, y con [`genetic/optuna.py`](./genetic/optuna.py) se pueden ejecutar experimentos de optimización de hiperparámetros con Optuna para encontrar las mejores combinaciones de operadores y parámetros.

### Algoritmos implementados

Implementación completa de algoritmos genéticos desde cero en el módulo [`genetic/`](./genetic/) con:

- **Inicialización**:
  - `RandomInitialization`: Aleatoria uniforme para problemas continuos.
  - `PermutationInitialization`: Permutaciones aleatorias para TSP.
  - `NeighborInitialization`: Vecino más cercano estocástico, elegido por proporcionar soluciones iniciales de mayor calidad.
  - `DiverseNNInitialization`: Vecino más cercano diversificado, elegido para maximizar diversidad inicial.

- **Selección**:
  - `TournamentSelection`: Elegido por su balance entre presión selectiva y diversidad.
  - `WeightedSelection`: Para escalarización de objetivos.
  - `ParetoSelection`: Para multi-objetivo, selecciona frentes no-dominados.

- **Operadores de cruce**:
  - `BlendCrossover` (BLX-α): Para problemas continuos, elegido por explorar el espacio entre padres.
  - `OrderCrossover` (OX): Para permutaciones, preserva el orden relativo de ciudades.
  - `PMXCrossover`: Elegido por su capacidad de preservar segmentos contiguos.
  - `CycleCrossover` (CX): Preserva posiciones absolutas de genes.
  - `EdgeRecombinationCrossover` (ERX): Maximiza aristas de los padres, elegido por su efectividad en TSP.

- **Mutaciones**:
  - `UniformMutation`: Para variables reales, genera diversidad uniforme.
  - `SwapMutation`: Intercambio de dos genes, mínima disrupción para permutaciones.
  - `InversionMutation`: Equivalente a 2-opt, elegida por eliminar cruces y mejorar geometría.
  - `ScrambleMutation`: Baraja un segmento, útil como operador exploratorio.
  - `CombinedMutation`: Aplica múltiples mutaciones secuencialmente.

- **Reemplazo**:
  - `GenerationalReplacement`: La descendencia reemplaza completamente a los padres.
  - `ElitistReplacement`: Elegido para preservar las mejores soluciones encontradas.
  - `MuPlusLambdaReplacement` (μ+λ): Elegido por combinar explotación y exploración.

### Problemas de optimización

#### Optimización mono-objetivo

**1. Función de Himmelblau** (`genetic/problems.py::HimmelblauProblem`)
- Función analítica: f(x,y) = (x²+y-11)² + (x+y²-7)²
- Dominio: x,y ∈ [-5, 5]
- 4 mínimos globales conocidos
- **Presupuesto de evaluaciones: 3.500** (según enunciado)
- Métodos de comparación: PySwarms (PSO), Scipy (L-BFGS-B, SLSQP, Nelder-Mead)

**2. Problema del Viajante (TSP)** (`genetic/problems.py::TSProblem`)
- Minimización de distancia total del recorrido
- Representación mediante permutaciones
- Distancia euclidiana entre ciudades
- **Presupuesto de evaluaciones: 1.000.000** (según enunciado)
- Métodos de comparación: Algoritmo de la colonia de hormigas (implementación propia en [`aco.py`](./aco.py))


#### Optimización multi-objetivo

**1. Problemas benchmark de Pymoo** (`genetic/problems.py::PymooProblem`)
- **ZDT3**: Frente de Pareto discontinuo (2 objetivos, 30 variables) — **Presupuesto: 10.000 evaluaciones**
- **MW7**: Problema con geometría compleja — **Presupuesto: 10.000 evaluaciones**
- **MW14**: Problema altamente no lineal con frente irregular y 3 objetivos — **Presupuesto: 10.000 evaluaciones**
- Disponibilidad de frente de Pareto verdadero para métricas

**2. TSP Multi-Objetivo** (`genetic/problems.py::MOTSProblem`)
- **Objetivo 1**: Distancia total (Euclidiana)
- **Objetivo 2**: Tiempo total de viaje (basado en elevación del terreno con ruido Perlin)
- Trade-off entre ruta corta vs. ruta rápida
- **Presupuesto de evaluaciones: 100.000** (según enunciado)

### Notebooks de experimentación

Los experimentos se encuentran en el directorio `notebooks/`:
- `himmelblau.ipynb`: Optimización de la función Himmelblau con GA y comparativa con PSO/Scipy
- `tsp.ipynb`: Optimización de un problema TSP dadas N ciudades con GA y comparativa con ACO
- `zdt1.ipynb`, `zdt3.ipynb`: Problemas benchmark ZDT
- `mw7.ipynb`, `mw14.ipynb`: Problemas benchmark MW
- `mo_tsp.ipynb`: TSP multi-objetivo con análisis de Pareto

### Formatos de resultados

Los resultados de optimización se exportan en formato CSV según el tipo de problema:

**Mono-objetivo (TSP, Himmelblau)** — `SingleObjectiveResult.save_csv()`:
| Columna | Descripción |
|---------|-------------|
| `generation` | Número de generación |
| `min_fitness` | Mejor fitness encontrado hasta esa generación |

**Multi-objetivo (TSP MO, ZDT, MW)** — `MultiObjectiveResult.save_csv()`:
| Columna | Descripción |
|---------|-------------|
| `objective1`, `objective2`, ... | Valores de cada objetivo en el frente de Pareto |

<a id="estructura-del-proyecto"></a>
## 🪴 Estructura del proyecto

```
practica_biocomp/
├── genetic/                    # Módulo principal de Algoritmos Genéticos
│   ├── __init__.py             # Inicializa el paquete y facilita imports básicos
│   ├── algorithms.py           # Implementación de GA mono y multi-objetivo
│   ├── configurations.py       # Configuración de problemas
│   ├── crossover.py            # Operadores de cruce
│   ├── individual.py           # Representación de individuos
│   ├── initialization.py       # Estrategias de inicialización de población
│   ├── mutation.py             # Operadores de mutación
│   ├── population.py           # Clase Population y operaciones
│   ├── problems.py             # Problemas de optimización
│   ├── replacement.py          # Estrategias de reemplazo
│   ├── results.py              # Almacenamiento de resultados
│   └── selection.py            # Métodos de selección
│
├── notebooks/                  # Jupyter Notebooks con experimentos
│   ├── himmelblau.ipynb        # Optimización de Himmelblau (GA vs PSO vs Scipy)
│   ├── tsp.ipynb               # TSP con Algoritmo Genético
│   ├── zdt1.ipynb              # Problema benchmark ZDT1
│   ├── zdt3.ipynb              # Problema benchmark ZDT3
│   ├── mw7.ipynb               # Problema benchmark MW7
│   ├── mw14.ipynb              # Problema benchmark MW14 (si implementado)
│   └── mo_tsp.ipynb            # TSP Multi-Objetivo (distancia vs tiempo)
│
├── diagrams/                   # Diagramas y visualizaciones
│   └── classes.excalidraw      # Diagrama de arquitectura de clases
│
├── aco.py                      # Implementación de ACO para TSP
├── main.py                     # Script de ejemplo de uso
├── pyproject.toml              # Configuración del proyecto y dependencias
├── uv.lock                     # Lock file de dependencias (uv)
├── .python-version             # Versión de Python requerida
├── CHANGELOG.md                # Historial de cambios
├── LICENSE                     # Licencia MIT
└── README.md                   # Este archivo
```

### Componentes principales

El módulo [`genetic/`](./genetic/) implementa un framework modular de algoritmos genéticos:

- **`Individual`** / **`Population`**: Representación de soluciones (`RealIndividual` para continuos, `PermutationIndividual` para TSP) y colecciones con evaluación de fitness.
- **`Problem`**: Clase base para problemas (`SingleObjectiveProblem`, `MultiObjectiveProblem`) con tracking de evaluaciones.
- **`ProblemConfig`**: Centraliza semillas, budget de evaluaciones y directorio de salida.
- **Operadores**: Intercambiables y configurables (ver sección Metodología para lista completa).
- **`GeneticAlgorithmSO`** / **`GeneticAlgorithmMO`**: Lógica principal del algoritmo evolutivo mono y multi-objetivo.
- **`SingleObjectiveResult`** / **`MultiObjectiveResult`**: Exportación a CSV y visualización.

<p align="right">(<a href="#readme-top">Volver arriba</a>)</p>

<a id="ejecucion-del-codigo"></a>
## 🐍 Ejecución del código

Los notebooks en [`notebooks/`](./notebooks/) contienen experimentos completos y reproducibles (ver lista en sección Metodología).

### Comparación con métodos tradicionales

Para problemas de referencia, puedes usar los métodos integrados:

```python
from genetic.configurations import ProblemConfig
from genetic.problems import HimmelblauProblem

config = ProblemConfig(max_evaluations=3500, seed=42, output_dir="results/himmelblau")

problem = HimmelblauProblem(config)

# Himmelblau con PySwarms (PSO)
pso_result = problem.get_pyswarms_result(c1=0.5, c2=0.3, w=0.9)

# Himmelblau con Scipy (L-BFGS-B)
scipy_result = problem.get_scipy_result(method='L-BFGS-B', tol=1e-8)

# Comparar resultados
print(f"GA: {ga_result.best_fitness}")
print(f"PSO: {pso_result.best_fitness}")
print(f"Scipy: {scipy_result.best_fitness}")
```

### 🔧 Añadir nuevos problemas

El framework está diseñado para ser fácilmente extensible. Para añadir un nuevo problema:

#### Problema mono-objetivo

```python
from genetic.problems import SingleObjectiveProblem
from typing import List, Tuple

class MiProblema(SingleObjectiveProblem):
    def __init__(self, config):
        super().__init__(config, minimize=True)  # o False para maximizar
        self.bounds = [(-10, 10), (-10, 10)]  # límites de variables
    
    def _fitness_function(self, solution: List) -> float:
        """
        Definición de la función objetivo: 

        f(x, y) = 0.26·(x² + y²) - 0.48·x·y
        """
        x, y = solution
        return 0.26 * (x**2 + y**2) - 0.48 * x * y  # Ejemplo: Función Matyas
    
    def get_bounds(self) -> List[Tuple[float, float]]:
        return self.bounds
```

#### Problema multi-objetivo

```python
from genetic.problems import MultiObjectiveProblem
import numpy as np

class MiProblemaMO(MultiObjectiveProblem):
    def __init__(self, config):
        super().__init__(config, n_objectives=2)
        self.bounds = [(0, 1)] * 5  # 5 variables en [0,1]
    
    def _fitness_function(self, solution: List) -> np.ndarray:
        """Devuelve un array con los valores de cada objetivo"""
        x = np.array(solution)
        f1 = np.sum(x**2)
        f2 = np.sum((x - 1)**2)
        return np.array([f1, f2])
    
    def get_bounds(self) -> List[Tuple[float, float]]:
        return self.bounds
```

Luego úsalo como cualquier otro problema del framework. 

Cada uno de los módulos de [`genetic/`](./genetic/) cuenta con la interfaz básica de cada uno de los operadores para poder implementar todos los necesarios.

<p align="right">(<a href="#readme-top">Volver arriba</a>)</p>

<a id="roadmap"></a>
## 🪙 Roadmap

- [x] Creación de la población inicial
- [x] Ordenación
- [x] Selección
- [x] Mutación
- [x] Cruce
- [x] Evolución
- [x] TSP
- [x] Tres tipos de mutaciones
- [x] Optimización final de Himmelblau
- [ ] Optimización final de Himmelblau con onlyone=True
- [x] MO: Optimización de ZDT3
- [x] MO: Optimización de MW7
- [ ] MO: Optimización de MW14
- [x] MO: Optimización de TSP MO
- [ ] MO: Comparativas con NSGA
- [ ] MO: Implementación y cálculo de métricas
- [ ] Final: presentación

<p align="right">(<a href="#readme-top">Volver arriba</a>)</p>

<a id="autores"></a>
## 🫂 Autores

<a href="https://github.com/yabol02/practica_biocomp/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=yabol02/practica_biocomp" />
</a>

###### Made with [contrib.rocks](https://contrib.rocks).

- [Aguirregabiria Herrero, Rodrigo](https://github.com/raguirregabiria)
- [Boleas Francisco, Yago](https://github.com/yabol02)
- [Estoquera Núñez, Adrian](https://github.com/aestoquera)

<p align="right">(<a href="#readme-top">Volver arriba</a>)</p>

<a id="licencia"></a>
## 🗝️ Licencia

Distribuido bajo la licencia MIT. Ve a [`LICENSE`](LICENSE) para mayor información.

<p align="right">(<a href="#readme-top">Volver arriba</a>)</p>

## 📖 Referencias

### Librerías y frameworks utilizados
- **NumPy**: https://numpy.org/ — Computación numérica en Python
- **Pymoo**: https://pymoo.org/ — Framework de optimización multiobjetivo en Python
- **PySwarms**: https://pyswarms.readthedocs.io/ — Particle Swarm Optimization en Python
- **SciPy**: https://scipy.org/ — Biblioteca de computación científica en Python

### Algoritmos implementados
- **Algoritmos Genéticos**: Goldberg, D.E. (1989). *Genetic Algorithms in Search, Optimization and Machine Learning*
- **NSGA-II**: Deb, K., Pratap, A., Agarwal, S. & Meyarivan, T. (2002). *A fast and elitist multiobjective genetic algorithm: NSGA-II*. IEEE Transactions on Evolutionary Computation, 6(2): 182-197. DOI: 10.1109/4235.996017

### Problemas benchmark
- **ZDT Test Suite**: Zitzler, E., Deb, K., & Thiele, L. (2000). *Comparison of Multiobjective Evolutionary Algorithms: Empirical Results*. Evolutionary Computation, 8(2), 173–195
- **MW Test Suite**: Ma, Z., & Wang, Y. (2019). *Evolutionary Constrained Multiobjective Optimization: Test Suite Construction and Performance Comparisons*. IEEE Transactions on Evolutionary Computation, 23(6), 972–986


<p align="right">(<a href="#readme-top">Volver arriba</a>)</p>

<a href="https://www.etsisi.upm.es/">
  <img src="https://www.upm.es/gsfs/SFS11386"></img>
</a>