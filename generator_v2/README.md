# Wind Farm Problem Generator

This repository provides a simple and modular way to generate synthetic wind farm layout problems for optimization experiments.

The generator creates:

- a **feasible area polygon**, which represents the area available for wind turbine placement after excluding shipping and fishery constraints,
- an **oil & gas field polygon**, which represents an additional forbidden region,
- deterministic problem instances, so the same configuration returns the same problems every time.

The code is split into small modules so that it is easy to use inside optimization algorithms.

## Repository structure

- `windfarm_problem/config.py`  
  Configuration dataclass with all user-facing settings.

- `windfarm_problem/geometry.py`  
  Low-level geometry utilities, including validity repair, coverage calculation, and scaling.

- `windfarm_problem/generator.py`  
  Deterministic generation of problem instances.

- `windfarm_problem/instance.py`  
  `ProblemInstance` class and the indicator functions used during optimization.

- `windfarm_problem/visualization.py`  
  Plotting utilities to inspect a generated layout.

- `examples/toy_example.py`  
  Small example showing how to generate and use problem instances.

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
from windfarm_problem import GeneratorConfig, generate_problem_instances
from windfarm_problem.visualization import plot_problem

config = GeneratorConfig(
    n_designs=3,
    seed=7,
    target_feasible_coverage_percent=80.0,
    target_reservoir_coverage_percent=15.0,
)

problems = generate_problem_instances(config)

problem_1 = problems[1]

print(problem_1.available_area_indicator(0.30, 0.60))
print(problem_1.oil_gas_indicator(0.30, 0.60))
print(problem_1.feasibility_indicator(0.30, 0.60))

plot_problem(problem_1, title="Problem 1")
```

## Indicator functions

Each problem instance provides three indicator functions:

- `available_area_indicator(x, y)`  
  Returns `1` if the point is inside the available area polygon, otherwise `0`.

- `oil_gas_indicator(x, y)`  
  Returns `1` if the point is inside the oil & gas polygon, otherwise `0`.

- `feasibility_indicator(x, y)`  
  Returns `1` if the point is inside the unit-square solution space, inside the available area, and outside the oil & gas polygon. Otherwise it returns `0`.

These functions are intentionally simple so they can be called directly inside optimization routines.

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

## Toy example

Run:

```bash
python examples/toy_example.py
```

This will:

1. generate a few deterministic problem instances,
2. evaluate a sample point,
3. plot the first instance.
