from config import GeneratorConfig
from generator import generate_problem_instances
from visualization import plot_problem

from evaluation import WindFarmEvaluator


problem_seed = 1


config = GeneratorConfig(
    n_designs=3,
    seed=problem_seed,
    n_reservoirs=3,
    context_side=3,
    target_feasible_coverage_percent=95.0,
    reservoir_coverage_percent=10.0,   # assign 10% coverage percentage to all reservoirs
    # reservoir_coverage_percent=[10.0, 15.0, 5.5], # assign different 10%, 15%, 5.5% coverage percentages to reservoirs
    max_reservoir_attempts=2000,
)

problems = generate_problem_instances(config)
problem_1 = problems[1]

evaluator = WindFarmEvaluator(problem_1, ensemble_file="Ensemble.pkl", n_turbines=5)

# candidate solution(locations of turbines and the hub)
hub = [1.15, 1.35]
x = [0.87601546, 0.30708387, 0.95377465, 0.57068488, 0.51457379, 
     0.53644864, 0.76385136, 0.91770092, 0.67833953, 0.06152623]

print(evaluator.evaluate(x, hub))

# problem plot
plot_problem(problem_1, len_plot=2, title="Toy problem_1", path=f"results/problem_1.png")

# problem and solution plot
plot_problem(problem_1, x=x, hub=hub, len_plot=2, evaluator=evaluator, title="Solution_test", path=f"results/Solution_test.png")