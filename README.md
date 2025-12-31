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

Este repositorio contiene la entrega práctica de la asignatura **Computación Evolutiva y Bioinspirada**. El objetivo principal es el desarrollo, análisis y comparativa de diferentes algoritmos bioinspirados aplicados a problemas de optimización mono-objetivo y multi-objetivo.

El proyecto abarca implementaciones desde cero (from scratch) de:
* **Algoritmos Genéticos (GA):** Selección, cruce y mutación.
* **Optimización Mono-objetivo:** Función de Himmelblau y Problema del Viajante (TSP).
* **Optimización Multi-objetivo (MOEA):** Implementación de frentes de Pareto, métricas de diversidad y convergencia (ZDT3, MW7, MW14).

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

### Instalación

Una vez instalado `uv`, la configuración es automática. Desde la raíz del repositorio:

1. Clona el repositorio:
   ```bash
   git clone https://github.com/yabol02/practica_biocomp.git
   ```

2. Sincroniza el entorno:
    ```bash
    uv sync
    ```
    Este comando creará el virtual environment (`.venv`) e instalará todas las librerías exactas definidas en el `uv.lock`.


<p align="right">(<a href="#readme-top">Volver arriba</a>)</p>

<a id="metodologia-y-experimentos"></a>
## 🧪 Metodología y experimentos

> [!WARNING]
> Sección en desarrollo

<a id="estructura-del-proyecto"></a>
## 🪴 Estructura del proyecto

> [!WARNING]
> Sección en desarrollo

<a id="ejecucion-del-codigo"></a>
## 🐍 Ejecución del código

> [!WARNING]
> Sección en desarrollo

<a id="roadmap"></a>
## 🪙 Roadmap

- [X] Creación de la población inicial
- [ ] Ordenación
- [ ] Selección
- [ ] Mutación
- [ ] Cruce
- [ ] Evolución
- [ ] TSP
- [ ] Tres tipos de mutaciones
- [ ] Optimización final de Himmelblau
- [ ] Optimización final de Himmelblau con onlyone::True
- [ ] MO: Optimización de ZDT3
- [ ] MO: Optimización de MW7
- [ ] MO: Optimización de MW14
- [ ] MO: Optimización de TSP MO
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

<a href="https://www.etsisi.upm.es/">
  <img src="https://www.upm.es/gsfs/SFS11386"></img>
</a>