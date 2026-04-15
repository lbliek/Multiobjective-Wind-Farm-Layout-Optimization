import ast

import numpy as np
import pandas as pd
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from config import GeneratorConfig
from generator import generate_problem_instances
from evaluation import WindFarmEvaluator
from visualization import plot_problem


CSV_PATH = "random_search_results.csv"


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


# 3. Load random search results
df = pd.read_csv(CSV_PATH)

cur_feas = df[df["feasible"] == 1].copy()

if len(cur_feas) == 0:
    raise RuntimeError("No feasible solutions found in random_search_results.csv")


# 4. Non-dominated sorting on feasible solutions
F = cur_feas[["f1", "f2", "f3"]].to_numpy(dtype=float)
fronts = NonDominatedSorting().do(F)
first_front_idx = fronts[0]

print("Number of feasible solutions:", len(cur_feas))
print("Number of non-dominated solutions:", len(first_front_idx))


# 5. Pick one non-dominated solution
pick_id = 0   # you can change this to 1, 2, 3, ...
if pick_id >= len(first_front_idx):
    raise IndexError(f"pick_id={pick_id} is out of range for first front of size {len(first_front_idx)}")

row = cur_feas.iloc[first_front_idx[pick_id]]

print("Chosen eval_id:", row["eval_id"])
print("f values:", row[["f1", "f2", "f3"]].to_dict())
print("g values:", row[["g1", "g2"]].to_dict())


# 6. Recover x and hub
x = ast.literal_eval(row["x"])
hub = ast.literal_eval(row["hub"])

x = np.array(x, dtype=float)
hub = np.array(hub, dtype=float)

print("x =", x)
print("hub =", hub)


# 7. Plot the selected layout
plot_problem(
    problem_2,
    x=x,
    hub=hub,
    evaluator=evaluator,
    title=f"Chosen non-dominated solution (eval_id={row['eval_id']})",
)