import pickle

from evaluation import WindFarmEvaluator
from optimisation.randomsearch import run_random_search


algorithm_seed = 2026
problem_nr = 19  # 0 = classC Problem 1


with open("results/generated/classC/problems.pkl", "rb") as f:
    problems = pickle.load(f)

with open("results/generated/classC/hub_list.pkl", "rb") as f:
    hub_list = pickle.load(f)

with open("results/generated/classC/evaluatorparams.pkl", "rb") as f:
    evaluatorparamlist = pickle.load(f)


# Matching data for one fixed problem instance
problem = problems[problem_nr + 1]
hub = hub_list[problem_nr]
params = evaluatorparamlist[problem_nr]


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


df, feas = run_random_search(
    evaluator=evaluator,
    hub=hub,
    n_eval=500,
    seed=algorithm_seed,
    save_csv=True,
    csv_path=(
        f"results/random_classC_problem_{problem_nr + 1}"
        f"_seed_{algorithm_seed}.csv"
    ),
)


print("Problem number:", problem_nr + 1)
print("Fixed hub:", hub)
print("\nTop 5 feasible solutions (by f13):")
print(feas.sort_values("f13", ascending=True).head())