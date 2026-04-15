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

hub = [1.15, 1.35]

# x = [0.1, 0.2, 0.3, 0.4, 0.5,
#      0.6, 0.7, 0.8, 0.2, 0.1]

x = [0.87601546, 0.30708387, 0.95377465, 0.57068488, 0.51457379, 
     0.53644864, 0.76385136, 0.91770092, 0.67833953, 0.06152623]

print(evaluator.evaluate(x, hub))

# # problem plot
# plot_problem(problem_2, len_plot=1.6, title="Toy problem_2")

# problem and solution plot
plot_problem(problem_2, x=x, hub=hub, len_plot=1.6, evaluator=evaluator, title="Problem 2")