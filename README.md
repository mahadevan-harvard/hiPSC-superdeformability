# hiPSC-superdeformability

This repository contains simulation and analysis scripts for the **hiPSC superdeformability project**.
It complements the data repository [doi:10.5281/zenodo.17153949](https://doi.org/10.5281/zenodo.17153949), which provides the simulation output files.

---

## Getting Started

### Installation
Clone the repository and install in editable mode to use the plotting utilities:

```bash
pip install -e .
```

This will install `plot_utils.PlotLibrary` for figure generation and analysis.

### Requirements
The simulations rely on:
- [JAX](https://docs.jax.dev/en/latest/installation.html)
- [JAX-MD](https://github.com/jax-md/jax-md)

Make sure both are installed and working on your system.

---

## Repository Usage

### Data
- Users should download the simulation data from [Zenodo](https://doi.org/10.5281/zenodo.17153949).
- Place the data inside a folder called **`Data/`** at the top level of this repository.

### Plots
- Before running analysis scripts, create an empty folder called **`Plots/`**.
- Generated figures will be saved here.

## Folder Structure

- **`2DMechanics/`** – comparison of 2D theory and 2D simulations for mechanical response
- **`2Dsimulation/`** – the simulation code
- **`3DMechanics/`** – 3D cellular model and comparisons with experimental data
- **`OnsetPlots/`** – analysis of onsets of soft modes in simulation and comparison with theory
- **`simulationVisualization/`** – generate snapshots and movies from simulations
- **`src/`** – contains `plot_utils.PlotLibrary`
- **`TiltBendDistribution/`** – analyze distribution of soft modes

---
