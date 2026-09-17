import pickle

from evaluation import WindFarmEvaluator
from visualization import plot_problem


# Choose one of the generated problems:
problem_nr = 0


# Load the fixed benchmark data
with open("results/generated/classC/problems.pkl", "rb") as f:
    problems = pickle.load(f)

with open("results/generated/classC/hub_list.pkl", "rb") as f:
    hub_list = pickle.load(f)

with open("results/generated/classC/evaluatorparams.pkl", "rb") as f:
    evaluatorparamlist = pickle.load(f)


# Get the matching components of one benchmark problem
problem = problems[problem_nr + 1]
hub = hub_list[problem_nr]
params = evaluatorparamlist[problem_nr]


# Create the evaluator using this problem's fixed parameters
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


# Candidate layout: turbine coordinates only
x = [
    0.87601546, 0.30708387, 0.95377465, 0.57068488, 0.51457379,
    0.53644864, 0.76385136, 0.91770092, 0.67833953, 0.06152623,
]


# Evaluate turbines using this problem's fixed hub
results = evaluator.evaluate(x, hub)

print("Problem number:", problem_nr + 1)
print("Fixed hub:", hub)
print("Evaluation:", results)


# Plot the layout
plot_problem(
    problem,
    x=x,
    hub=hub,
    len_plot=1.6,
    evaluator=evaluator,
    title=f"Solution test — classC Problem {problem_nr + 1}",
    path=f"results/Solution_classC_problem_{problem_nr + 1}.png",
)