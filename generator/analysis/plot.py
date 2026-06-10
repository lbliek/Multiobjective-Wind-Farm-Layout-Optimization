import ast

import numpy as np
import pandas as pd
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # 指向 generator 文件夹
sys.path.insert(0, str(PROJECT_ROOT))

from config import GeneratorConfig
from generator import generate_problem_instances
from evaluation import WindFarmEvaluator
from visualization import plot_problem



problem_seed = 1
algorithm_seed = 2026

config = GeneratorConfig(
    n_designs=3,
    seed=1,
    n_reservoirs=3,
    context_side=3,
    target_feasible_coverage_percent=95.0,
    reservoir_coverage_percent=10.0,   # assign 5% coverage percentage to all reservoirs
    # reservoir_coverage_percent=[10.0, 15.0, 7.0, 3.5, 5.5], # assign different coverage percentages to reservoirs
    max_reservoir_attempts=2000,
)

# CSV_PATH = f"results/random_search_{problem_seed}_{algorithm_seed}.csv"

CSV_PATH = f"results/nsga2_{problem_seed}_{algorithm_seed}.csv"

# CSV_PATH = f"results/qlognehvi_{problem_seed}_{algorithm_seed}.csv"

# CSV_PATH = f"results/qlognparego_{problem_seed}_{algorithm_seed}.csv"

problems = generate_problem_instances(config)

problem_1 = problems[1]


evaluator = WindFarmEvaluator(problem_1, ensemble_file="Ensemble.pkl", n_turbines=5)



# Load optimisation results
df = pd.read_csv(CSV_PATH)

cur_feas = df[df["feasible"] == 1].copy()

if len(cur_feas) == 0:
    raise RuntimeError("No feasible solutions found in random_search_results.csv")


# Non-dominated sorting on feasible solutions
F = cur_feas[["f1", "f2", "f3"]].to_numpy(dtype=float)
fronts = NonDominatedSorting().do(F)
first_front_idx = fronts[0]

print("Number of feasible solutions:", len(cur_feas))
print("Number of non-dominated solutions:", len(first_front_idx))


# Pick one non-dominated solution
pick_id = 1   
if pick_id >= len(first_front_idx):
    raise IndexError(f"pick_id={pick_id} is out of range for first front of size {len(first_front_idx)}")

row = cur_feas.iloc[first_front_idx[pick_id]]

print("Chosen eval_id:", row["eval_id"])
print("f values:", row[["f1", "f2", "f3"]].to_dict())
print("g values:", row[["g1", "g2"]].to_dict())


# Recover x and hub
x = ast.literal_eval(row["x"])
hub = ast.literal_eval(row["hub"])

x = np.array(x, dtype=float)
hub = np.array(hub, dtype=float)

print("x =", x)
print("hub =", hub)


plot_problem(
    problem_1,
    
    x=x,
    hub=hub,
    len_plot=2,
    evaluator=evaluator,
    # title=f"random_search_{problem_seed}_{algorithm_seed}",
    title=f"nsga2_{problem_seed}_{algorithm_seed}",
    # title=f"qlognehvi_{problem_seed}_{algorithm_seed}",
    # title=f"qlognparego_{problem_seed}_{algorithm_seed}",


    # path=f"results/random_search_{problem_seed}_{algorithm_seed}.png"
    path=f"results/nsga2_{problem_seed}_{algorithm_seed}.png"
    # path=f"results/qlognehvi_{problem_seed}_{algorithm_seed}.png"
    # path=f"results/qlognparego_{problem_seed}_{algorithm_seed}.png"


)



