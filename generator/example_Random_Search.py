from config import GeneratorConfig
from generator import generate_problem_instances
from evaluation import WindFarmEvaluator
from optimisation.randomsearch import run_random_search


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
problems = generate_problem_instances(config)
problem_1 = problems[1]


evaluator = WindFarmEvaluator(problem_1, ensemble_file="Ensemble.pkl", n_turbines=5)



df, feas = run_random_search(
    evaluator,
    n_eval=500,  
    seed=algorithm_seed,
    save_csv=True,
    csv_path=f"results/random_search_{problem_seed}_{algorithm_seed}.csv",
)



print("\nTop 5 feasible solutions (by f1):")
print(feas.sort_values("f1", ascending=True).head())