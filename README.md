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

### ✨ Características principales

- 🧬 **Framework modular de Algoritmos Genéticos** implementado desde cero
  - Arquitectura orientada a objetos altamente extensible
  - Soporte completo para problemas mono y multi-objetivo
  - Operadores genéticos intercambiables (selección, cruce, mutación, reemplazo)

- 🎯 **Problemas de optimización implementados**
  - **Mono-objetivo**: Himmelblau, TSP clásico
  - **Multi-objetivo**: ZDT1, ZDT3, MW7, TSP multi-objetivo (distancia + tiempo)
  - Integración con benchmark problems de Pymoo

- 📊 **Sistema completo de experimentación**
  - Tracking automático de evaluaciones
  - Exportación de resultados (CSV, gráficas)
  - Notebooks Jupyter con análisis reproducibles
  - Comparación con métodos tradicionales (PSO, Scipy)

- 🔬 **Algoritmos multi-objetivo**
  - Dominancia de Pareto y fronts no-dominados
  - NSGA-II (Non-dominated Sorting Genetic Algorithm II)
  - Métricas de calidad (spread, ratio de no-dominancia)
  - Visualización de frentes de Pareto

### 📚 Contenido del proyecto

El proyecto abarca implementaciones desde cero (from scratch) de:
* **Algoritmos Genéticos (GA):** Selección, cruce, mutación y reemplazo
* **Optimización Mono-objetivo:** Función de Himmelblau y Problema del Viajante (TSP)
* **Optimización Multi-objetivo (MOEA):** Implementación de frentes de Pareto, métricas de diversidad y convergencia (ZDT1, ZDT3, MW7)

* **Comparativas:** Benchmarking contra PySwarms (PSO), Scipy (métodos clásicos) y Pymoo (MOEA)

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
> powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
> ```

### Dependencias principales

El proyecto utiliza las siguientes librerías:

| Librería | Versión | Propósito |
|----------|---------|-----------|
| `numpy` | ≥2.3.5 | Operaciones numéricas y matriciales |
| `scipy` | ≥1.16.3 | Métodos de optimización tradicionales |
| `matplotlib` | ≥3.10.8 | Visualización de resultados |
| `pandas` | ≥2.3.3 | Manipulación de datos y exportación |
| `pymoo` | ≥0.6.1.6 | Benchmark problems multi-objetivo |
| `pyswarms` | ≥1.3.0 | Particle Swarm Optimization (PSO) |
| `noise` | ≥1.2.2 | Generación de ruido Perlin (elevación en MO-TSP) |
| `tqdm` | ≥4.67.1 | Barras de progreso |
| `ipykernel` | ≥7.1.0 | Soporte para Jupyter notebooks |

> **Nota**: Todas las dependencias se instalan automáticamente con `uv sync`.

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
    
    python -c "from genetic import GeneticAlgorithmSO; print('✓ Instalación correcta')"
    ```

### ⚡ Quick Start

Prueba rápidamente el framework con un ejemplo mínimo:

```python
from genetic import GeneticAlgorithmSO
from genetic.configurations import ProblemConfig
from genetic.problems import HimmelblauProblem
from genetic.initialization import UniformInitialization
from genetic.selection import TournamentSelection
from genetic.crossover import BlendCrossover
from genetic.mutation import UniformMutation
from genetic.replacement import ElitismReplacement

# Configurar y ejecutar
config = ProblemConfig(max_evaluations=1000, seed=42)
problem = HimmelblauProblem(config)

ga = GeneticAlgorithmSO(
    problem=problem,
    population_size=30,
    initialization=UniformInitialization(),
    selection=TournamentSelection(tournament_size=3),
    crossover=BlendCrossover(alpha=0.5),
    mutation=UniformMutation(mutation_rate=0.1, bounds=problem.get_bounds()),
    replacement=ElitismReplacement(elite_size=2)
)

result = ga.run()
print(f"✓ Mejor fitness: {result.best_fitness:.6f}")
print(f"✓ Solución: {result.best_solution}")
```

Para explorar los experimentos completos, consulta los notebooks en el directorio `notebooks/`.


<p align="right">(<a href="#readme-top">Volver arriba</a>)</p>

<a id="metodologia-y-experimentos"></a>
## 🧪 Metodología y experimentos

Este proyecto implementa y compara diferentes enfoques de optimización bioinspirada para problemas mono-objetivo y multi-objetivo.

### Algoritmos implementados

#### 🧬 Algoritmo Genético (GA)
Implementación completa desde cero (`genetic/`) con:
- **Inicialización**: Aleatoria uniforme para individuos reales y permutaciones
- **Selección**: Torneo, selección ponderada, selección multi-objetivo (NSGA-II)
- **Operadores de cruce**:
  - `BlendCrossover`: Para problemas de variable continua (BLX-α)
  - `OrderCrossover`: Para problemas de permutación (OX)
  - Soporte estocástico de padres (selección aleatoria vs. secuencial)
- **Mutaciones**:
  - `UniformMutation`: Mutación uniforme para variables reales
  - `SwapMutation`: Intercambio de genes para permutaciones
  - `InversionMutation`: Inversión de subsecuencias
- **Reemplazo**: Elitismo, reemplazo generacional, reemplazo μ+λ (mu+lambda)

### Problemas de optimización

#### Optimización mono-objetivo

**1. Función de Himmelblau** (`genetic/problems.py::HimmelblauProblem`)
- Función analítica: f(x,y) = (x²+y-11)² + (x+y²-7)²
- Dominio: x,y ∈ [-5, 5]
- 4 mínimos globales conocidos
- Métodos de comparación:
  - PySwarms (Particle Swarm Optimization)
  - Scipy (L-BFGS-B, SLSQP, Nelder-Mead)

**2. Problema del Viajante (TSP)** (`genetic/problems.py::TSProblem`)
- Minimización de distancia total del recorrido
- Representación mediante permutaciones
- Distancia euclidiana entre ciudades


#### Optimización multi-objetivo

**1. Problemas benchmark de Pymoo** (`genetic/problems.py::PymooProblem`)
- **ZDT1**: Problema convexo clásico (2 objetivos, 30 variables)
- **ZDT3**: Frente de Pareto discontinuo
- **MW7**: Problema con geometría compleja
- Disponibilidad de frente de Pareto verdadero para métricas

**2. TSP Multi-Objetivo** (`genetic/problems.py::MOTSProblem`)
- **Objetivo 1**: Distancia total (Euclidiana)
- **Objetivo 2**: Tiempo total de viaje
  - Basado en elevación del terreno (ruido Perlin)
  - Penalización por subidas (coef. 2.0)
  - Beneficio por bajadas (coef. 0.5)
- Trade-off entre ruta corta vs. ruta rápida

### Algoritmos multi-objetivo

**Selección basada en dominancia de Pareto**:
- `ParetoSelection`: Torneo con dominancia
- `ParetoSelection`: Ordenación por fronts (NSGA-II)
- Actualización incremental del frente de Pareto

**Métricas de evaluación**:
- Número de soluciones no-dominadas
- Spread del frente (varianza de objetivos)
- Ratio de no-dominancia en población

### Notebooks de experimentación

Los experimentos se encuentran en el directorio `notebooks/`:
- `himmelblau.ipynb`: Optimización y comparativa con PSO/Scipy
- `tsp.ipynb`: TSP con GA
- `zdt1.ipynb`, `zdt3.ipynb`: Problemas benchmark ZDT
- `mw7.ipynb`, `mw14.ipynb`: Problemas benchmark MW
- `mo_tsp.ipynb`: TSP multi-objetivo con análisis de Pareto
- `Pymoo alumno.ipynb`: Comparativa con algoritmos de Pymoo (NSGA-II, MOEAD)

<a id="estructura-del-proyecto"></a>
## 🪴 Estructura del proyecto

```
practica_biocomp/
├── genetic/                    # Módulo principal de Algoritmos Genéticos
│   ├── __init__.py            # Inicializa el paquete y facilita imports básicos
│   ├── algorithms.py          # Implementación de GA mono y multi-objetivo
│   ├── configurations.py      # Configuración de problemas
│   ├── crossover.py           # Operadores de cruce
│   ├── individual.py          # Representación de individuos
│   ├── initialization.py      # Estrategias de inicialización de población
│   ├── mutation.py            # Operadores de mutación
│   ├── population.py          # Clase Population y operaciones
│   ├── problems.py            # Problemas de optimización
│   ├── replacement.py         # Estrategias de reemplazo
│   ├── results.py             # Almacenamiento de resultados
│   └── selection.py           # Métodos de selección
│
├── notebooks/                  # Jupyter Notebooks con experimentos
│   ├── himmelblau.ipynb       # Optimización de Himmelblau (GA vs PSO vs Scipy)
│   ├── tsp.ipynb              # TSP con Algoritmo Genético
│   ├── zdt1.ipynb             # Problema benchmark ZDT1
│   ├── zdt3.ipynb             # Problema benchmark ZDT3
│   ├── mw7.ipynb              # Problema benchmark MW7
│   ├── mw14.ipynb             # Problema benchmark MW14 (si implementado)
│   ├── mo_tsp.ipynb           # TSP Multi-Objetivo (distancia vs tiempo)
│   ├── Pymoo alumno.ipynb     # Comparativas con algoritmos de Pymoo
│   ├── AG alumno.ipynb        # Desarrollo y pruebas de AG
│   └── TSP_development.ipynb  # Desarrollo y depuración de TSP
│
├── diagrams/                   # Diagramas y visualizaciones
│   └── classes.excalidraw     # Diagrama de arquitectura de clases
│
├── aco.py                      # Implementación de ACO para TSP
├── main.py                     # Script de ejemplo de uso
├── pyproject.toml             # Configuración del proyecto y dependencias
├── uv.lock                    # Lock file de dependencias (uv)
├── .python-version            # Versión de Python requerida
├── CHANGELOG.md               # Historial de cambios
├── LICENSE                    # Licencia MIT
└── README.md                  # Este archivo
```

### Componentes principales

#### Módulo `genetic`
Framework completo y modular para algoritmos genéticos:

- **`Individual`**: Representación abstracta de soluciones
  - `RealIndividual`: Individuos con genes de valor real
  - `PermutationIndividual`: Individuos con genes de permutación (para TSP)

- **`Population`**: Colección de individuos con operaciones de:
  - Evaluación en paralelo
  - Ordenación por fitness
  - Soporte multi-objetivo (frentes de Pareto)

- **`Problem`**: Clase base para definir problemas de optimización
  - `SingleObjectiveProblem`: Problemas con un solo objetivo
  - `MultiObjectiveProblem`: Problemas con múltiples objetivos
  - Sistema de tracking de evaluaciones y historial

- **Operadores genéticos**:
  - Totalmente intercambiables y configurables
  - Separación clara de responsabilidades
  - Fácil extensión para nuevos operadores

#### Sistema de configuración
- **`ProblemConfig`**: Centraliza parámetros de experimentación
  - Semillas aleatorias para reproducibilidad
  - Budget de evaluaciones
  - Directorio de salida de resultados

#### Sistema de resultados
- **`SingleObjectiveResult`**: Para problemas mono-objetivo
- **`MultiObjectiveResult`**: Para problemas multi-objetivo
- Exportación a CSV y visualización de convergencia/Pareto

### 💡 Características avanzadas

- **Reproducibilidad**: Todas las ejecuciones son reproducibles mediante semillas
- **Extensibilidad**: Arquitectura modular que facilita añadir nuevos problemas y operadores
- **Performance**: Uso de NumPy para operaciones vectorizadas eficientes
- **Tracking completo**: Historial de evaluaciones, convergencia y métricas
- **Visualización**: Gráficas automáticas de convergencia y frentes de Pareto

<p align="right">(<a href="#readme-top">Volver arriba</a>)</p>

<a id="ejecucion-del-codigo"></a>
## 🐍 Ejecución del código

### Ejemplo básico: Optimización de Himmelblau

```python
from genetic.algorithms import GeneticAlgorithmSO
from genetic.configurations import ProblemConfig
from genetic.crossover import BlendCrossover
from genetic.initialization import UniformInitialization
from genetic.mutation import UniformMutation
from genetic.problems import HimmelblauProblem
from genetic.replacement import ElitismReplacement
from genetic.selection import TournamentSelection

# Configurar el problema
config = ProblemConfig(
    max_evaluations=5000,
    seed=42,
    output_dir="results/himmelblau"
)
problem = HimmelblauProblem(config)

# Configurar el algoritmo genético
ga = GeneticAlgorithmSO(
    problem=problem,
    population_size=50,
    initialization=UniformInitialization(),
    selection=TournamentSelection(tournament_size=3),
    crossover=BlendCrossover(alpha=0.5),
    mutation=UniformMutation(mutation_rate=0.1, bounds=problem.get_bounds()),
    replacement=ElitismReplacement(elite_size=2)
)

# Ejecutar
result = ga.run()
print(f"Mejor solución: {result.best_solution}")
print(f"Fitness: {result.best_fitness}")

# Guardar resultados
result.save_csv("results/himmelblau/history.csv")
result.plot_convergence().savefig("results/himmelblau/convergence.png")
```

### Ejemplo: TSP con Algoritmo Genético

```python
from genetic.algorithms import GeneticAlgorithmSO
from genetic.configurations import ProblemConfig
from genetic.crossover import OrderCrossover
from genetic.initialization import PermutationInitialization
from genetic.mutation import SwapMutation
from genetic.problems import TSProblem
from genetic.replacement import GenerationalReplacement
from genetic.selection import TournamentSelection

# Definir ciudades
cities = [(0, 0), (1, 5), (5, 3), (8, 1), (4, 7)]

# Configurar problema
config = ProblemConfig(max_evaluations=10000, seed=42)
problem = TSProblem(config, cities)

# Configurar GA
ga = GeneticAlgorithmSO(
    problem=problem,
    population_size=100,
    initialization=PermutationInitialization(),
    selection=TournamentSelection(tournament_size=5),
    crossover=OrderCrossover(),
    mutation=SwapMutation(mutation_rate=0.2),
    replacement=GenerationalReplacement()
)

# Ejecutar
result = ga.run()
print(f"Mejor ruta: {result.best_solution}")
print(f"Distancia total: {result.best_fitness}")
```

### Ejemplo: TSP con ACO

```python
from aco import AntColony


```

### Ejemplo: Optimización Multi-Objetivo (ZDT3)

```python
from genetic.algorithms import GeneticAlgorithmMO
from genetic.configurations import ProblemConfig
from genetic.crossover import BlendCrossover
from genetic.initialization import UniformInitialization
from genetic.mutation import UniformMutation
from genetic.problems import PymooProblem
from genetic.replacement import ElitismReplacement
from genetic.selection import NonDominatedSortingSelection

# Configurar problema
config = ProblemConfig(max_evaluations=25000, seed=42)
problem = PymooProblem(config, problem_name="zdt3", n_var=30)

# Configurar MOEA
ga = GeneticAlgorithmMO(
    problem=problem,
    population_size=100,
    initialization=UniformInitialization(),
    selection=NonDominatedSortingSelection(),
    crossover=BlendCrossover(alpha=0.5),
    mutation=UniformMutation(mutation_rate=0.05, bounds=problem.get_bounds()),
    replacement=ElitismReplacement(elite_size=10)
)

# Ejecutar
result = ga.run()

# Visualizar frente de Pareto
problem.plot_pareto_front(show_true_front=True, problem_name="ZDT3")

# Acceder a soluciones
print(f"Número de soluciones no-dominadas: {len(result.pareto_front)}")
```

### Uso de notebooks

Los notebooks en `notebooks/` contienen experimentos completos y reproducibles:

1. **Inicia Jupyter**:
   ```bash
   uv run jupyter notebook
   ```

2. **Notebooks disponibles**:
   - `himmelblau.ipynb`: Comparación GA vs PSO vs métodos tradicionales
   - `tsp.ipynb`: Resolución del TSP con GA en diferentes instancias
   - `zdt1.ipynb`, `zdt3.ipynb`: Optimización multi-objetivo ZDT
   - `mw7.ipynb`: Problema MW7 con análisis de Pareto
   - `mo_tsp.ipynb`: TSP bi-objetivo (distancia vs tiempo)

3. **Ejecuta las celdas** para reproducir los experimentos

### Comparación con métodos tradicionales

Para problemas de referencia, puedes usar los métodos integrados:

```python
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
        """Define tu función objetivo aquí"""
        x, y = solution
        return x**2 + y**2  # ejemplo: esfera
    
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
        """Retorna un array con los valores de cada objetivo"""
        x = np.array(solution)
        f1 = np.sum(x**2)
        f2 = np.sum((x - 1)**2)
        return np.array([f1, f2])
    
    def get_bounds(self) -> List[Tuple[float, float]]:
        return self.bounds
```

Luego úsalo como cualquier otro problema del framework.

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
- [ ] Optimización final de Himmelblau con onlyone::True
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

---

## 📖 Referencias

### Librerías y frameworks utilizados
- **Pymoo**: [pymoo.org](https://pymoo.org/) - Framework multi-objetivo para Python
- **PySwarms**: [pyswarms.readthedocs.io](https://pyswarms.readthedocs.io/) - Particle Swarm Optimization
- **Scipy**: [scipy.org](https://scipy.org/) - Scientific computing library
- **NumPy**: [numpy.org](https://numpy.org/) - Numerical computing

### Algoritmos implementados
- **Algoritmos Genéticos**: Goldberg, D.E. (1989). "Genetic Algorithms in Search, Optimization, and Machine Learning"
- **NSGA-II**: Deb, K., et al. (2002). "A fast and elitist multiobjective genetic algorithm: NSGA-II"
- **Ant Colony Optimization**: Dorigo, M., & Stützle, T. (2004). "Ant Colony Optimization"
- **2-opt**: Croes, G.A. (1958). "A Method for Solving Traveling-Salesman Problems"

### Problemas benchmark
- **ZDT Test Suite**: Zitzler, E., Deb, K., & Thiele, L. (2000). "Comparison of Multiobjective Evolutionary Algorithms"
- **MW Test Suite**: Ma, Z., & Wang, Y. (2019). "Evolutionary Constrained Multiobjective Optimization"

<p align="right">(<a href="#readme-top">Volver arriba</a>)</p>

<a href="https://www.etsisi.upm.es/">
  <img src="https://www.upm.es/gsfs/SFS11386"></img>
</a>