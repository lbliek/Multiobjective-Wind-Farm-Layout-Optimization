import pickle

from evaluation import WindFarmEvaluator
from optimisation.qLogNParEGO import run_qlognparego  # use your actual filename


algorithm_seed = 2026

nr_of_problems = 50 # reduce this if 50 problems is too much to solve





# Load one fixed benchmark set
with open("results/generated/classB/problems.pkl", "rb") as f:
    problems = pickle.load(f)

with open("results/generated/classB/hub_list.pkl", "rb") as f:
    hub_list = pickle.load(f)

with open("results/generated/classB/evaluatorparams.pkl", "rb") as f:
    evaluatorparamlist = pickle.load(f)



for problem_nr in range(0,nr_of_problems):

    print('Loading problem number ', problem_nr + 1)


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

    # qLogNParEGO optimises turbines only and uses this fixed hub
    df, feas, model = run_qlognparego(
        evaluator=evaluator,
        hub=hub,
        n_eval=500,
        n_initial=50,
        seed=algorithm_seed,
        save_csv=True,
        csv_path=(
            f"results/qlognparego_classB_problem_{problem_nr + 1}"
            f"_seed_{algorithm_seed}.csv"
        ),
    )


    print("Solved problem number ", problem_nr + 1)
    #print("Fixed hub:", hub)
    #print("\nTop 5 feasible solutions (by f13):")
    #print(feas.sort_values("f13", ascending=True).head())