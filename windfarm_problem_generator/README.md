Wind Farm Problem Generator
This repository provides a simple and modular way to generate synthetic wind farm layout problems for optimization experiments.
The generator creates:
a feasible area polygon, which represents the area available for wind turbine placement after excluding shipping and fishery constraints,
an oil & gas field polygon, which represents an additional forbidden region,
deterministic problem instances, so the same configuration returns the same problems every time.
The code is split into small modules so that it is easy to use inside optimization algorithms.
Repository structure
`windfarm\_problem/config.py`  
Configuration dataclass with all user-facing settings.
`windfarm\_problem/geometry.py`  
Low-level geometry utilities, including validity repair, coverage calculation, and scaling.
`windfarm\_problem/generator.py`  
Deterministic generation of problem instances.
`windfarm\_problem/instance.py`  
`ProblemInstance` class and the indicator functions used during optimization.
`windfarm\_problem/visualization.py`  
Plotting utilities to inspect a generated layout.
`examples/toy\_example.py`  
Small example showing how to generate and use problem instances.
Installation
Install the required packages:
```bash
pip install -r requirements.txt
```
Typical use case
```python
from windfarm\_problem import GeneratorConfig, generate\_problem\_instances
from windfarm\_problem.visualization import plot\_problem

config = GeneratorConfig(
    n\_designs=3,
    seed=7,
    target\_feasible\_coverage\_percent=80.0,
    target\_reservoir\_coverage\_percent=15.0,
)

problems = generate\_problem\_instances(config)

problem\_1 = problems\[1]

print(problem\_1.available\_area\_indicator(0.30, 0.60))
print(problem\_1.oil\_gas\_indicator(0.30, 0.60))
print(problem\_1.feasibility\_indicator(0.30, 0.60))

plot\_problem(problem\_1, title="Problem 1")
```
Indicator functions
Each problem instance provides three indicator functions:
`available\_area\_indicator(x, y)`  
Returns `1` if the point is inside the available area polygon, otherwise `0`.
`oil\_gas\_indicator(x, y)`  
Returns `1` if the point is inside the oil & gas polygon, otherwise `0`.
`feasibility\_indicator(x, y)`  
Returns `1` if the point is inside the unit-square solution space, inside the available area, and outside the oil & gas polygon. Otherwise it returns `0`.
These functions are intentionally simple so they can be called directly inside optimization routines.
Reproducibility
The generator uses a fixed random seed from the configuration.  
As long as the configuration and the seed remain unchanged, the same problem instances will be generated.
Notes on geometry
Available area
The available area is generated as a convex or non-convex polygon and then tuned using uniform scaling so that its coverage inside the unit square matches a target percentage.
Oil & gas field
The oil & gas field is generated inside a larger context square centered on the unit square. It is also tuned using uniform scaling so that its intersection with the unit square matches the requested target percentage.
Visualization
Plots only show the `1 x 1` solution space, even though the oil & gas polygon may extend well outside it.
Toy example
Run:
```bash
python examples/toy\_example.py
```
This will:
generate a few deterministic problem instances,
evaluate a sample point,
plot the first instance.
