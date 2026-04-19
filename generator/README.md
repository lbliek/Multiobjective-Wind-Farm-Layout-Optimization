# Wind Farm Problem Generator

This repository provides a simple and modular way to generate synthetic wind farm layout problems for optimization experiments.

The generator creates:

- a **feasible area polygon**, which represents the area available for wind turbine placement after excluding shipping and fishery constraints,
- an **oil & gas field polygon**, which represents an additional forbidden region,
- some **turbine** and **hub** points, which represents the location of candidate wind turbines and hub,
- a **bird** group, which represents birds that may approach the turbines too closely,
- deterministic problem instances, so the same configuration returns the same problems every time.

The code is split into small modules so that it is easy to use inside optimization algorithms.

## Repository structure

- `generator/config.py`  
  Configuration dataclass with all user-facing settings.

- `generator/geometry.py`  
  Low-level geometry utilities, including validity repair, coverage calculation, and scaling.

- `generator/generator.py`  
  Deterministic generation of problem instances.

- `generator/instance.py`  
  `ProblemInstance` class and the indicator functions used during optimization.

- `generator/visualization.py`  
  Plotting utilities to inspect a generated layout.

- `generator/example.py`  
  Small example showing how to generate and use problem instances.

- `generator/example_NSGA2.py`  
  Small example showing how to use NSGA2 to optimise the problem.

- `generator/optimisation`  
  Optimisation methods.

- `generator/analysis.py`  
  Small example showing how to analysis the result of NSGA2.

- `generator/Ensemble.pkl`  
  A surrogate model used to simulate objective 1.

- `generator/results`  
  Results of optimisation.

## Installation

Install the required packages:

```bash
pip install -r requirements.txt
```

On Linux, the following is also needed for visualization:

```python
pip install pyQT6

sudo apt-get install -y libxcb-cursor-dev
```


## Typical use case

```python
from config import GeneratorConfig
from generator import generate_problem_instances
from visualization import plot_problem

from evaluation import WindFarmEvaluator

config = GeneratorConfig(
    n_designs=3,
    seed=2026,
    n_reservoirs=4,
    context_side=3,
    target_feasible_coverage_percent=95.0,
    target_reservoir_coverage_percent=20.0,
)

problems = generate_problem_instances(config)
problem_2 = problems[2]

evaluator = WindFarmEvaluator(problem_2, ensemble_file="Ensemble.pkl", n_turbines=5)

# candidate solution(locations of turbines and the hub)
hub = [1.15, 1.35]
x = [0.87601546, 0.30708387, 0.95377465, 0.57068488, 0.51457379, 
     0.53644864, 0.76385136, 0.91770092, 0.67833953, 0.06152623]

print(evaluator.evaluate(x, hub))

# problem plot
plot_problem(problem_2, len_plot=1.6, title="Toy problem_2")

# problem and solution plot
plot_problem(problem_2, x=x, hub=hub, len_plot=1.6, evaluator=evaluator, title="Problem 2", path=f"results/problem_2.png")
```

## Reproducibility

The generator uses a fixed random seed from the configuration.  
As long as the configuration and the seed remain unchanged, the same problem instances will be generated.

## Notes on geometry

### Available area
The available area is generated as a convex or non-convex polygon and then tuned using **uniform scaling** so that its coverage inside the unit square matches a target percentage.

### Oil & gas field
The oil & gas field is generated inside a larger context square centered on the unit square. It is also tuned using **uniform scaling** so that its intersection with the unit square matches the requested target percentage.

### Visualization
Plots only show the `1 x 1` solution space, even though the oil & gas polygon may extend well outside it.

## example

Run:

```bash
python generator/example.py
```

This will:

1. generate a few deterministic problem instances,
2. evaluate a sample point,
3. plot the first instance.
