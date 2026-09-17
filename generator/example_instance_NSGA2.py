import pickle

from evaluation import WindFarmEvaluator
from optimisation.NSGA2 import run_nsga2


algorithm_seed = 2026
problem_nr = 0  


# Load one fixed benchmark set
with open("results/generated/classC/problems.pkl", "rb") as f:
    problems = pickle.load(f)

with open("results/generated/classC/hub_list.pkl", "rb") as f:
    hub_list = pickle.load(f)

with open("results/generated/classC/evaluatorparams.pkl", "rb") as f:
    evaluatorparamlist = pickle.load(f)


# Get the corresponding problem, hub, and evaluator parameters
problem = problems[problem_nr + 1]
hub = hub_list[problem_nr]
params = evaluatorparamlist[problem_nr]


# Build the evaluator for this specific problem
evaluator = WindFarmEvaluator(
    problem=problem,
    ensemble_file="Ensemble.pkl",
    n_turbines=params["n_turbines"],
    bird_angle=params["bird_angle"],
    nr_birds=params["nr_birds"],
    bird_mean=params["bird_mean"],
    x_sigma=params["x_sigma"],
    rotor_diameter=params["rotor_diameter"],
    farm_length=params["farm_length"],
    seed=params["seed"],
)


# Run NSGA2: it optimises turbines only and always uses this fixed hub
df, feas, res = run_nsga2(
    evaluator=evaluator,
    hub=hub,
    n_eval=500,
    pop_size=50,
    seed=algorithm_seed,
    save_csv=True,
    csv_path=f"results/nsga2_classC_problem_{problem_nr + 1}_seed_{algorithm_seed}.csv",
)


print("Problem number:", problem_nr + 1)
print("Fixed hub:", hub)
print("\nTop 5 feasible solutions (by f1):")
print(feas.sort_values("f1", ascending=True).head())