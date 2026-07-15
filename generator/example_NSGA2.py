from config import GeneratorConfig
from generator import generate_problem_instances
from evaluation import WindFarmEvaluator
from optimisation.NSGA2 import run_nsga2


problem_seed = 1
algorithm_seed = 2026

config = GeneratorConfig(
    n_designs=3,
    seed=problem_seed,
    n_reservoirs=3,
    context_side=3,
    target_feasible_coverage_percent=95.0,
    reservoir_coverage_percent=10.0,
    # reservoir_coverage_percent=[10.0, 15.0, 7.0],
    max_reservoir_attempts=2000,
)

problems = generate_problem_instances(config)
problem = problems[3]

evaluator = WindFarmEvaluator(
    problem,
    ensemble_file="Ensemble.pkl",
    n_turbines=5,
)

df, feas, res = run_nsga2(
    evaluator,
    n_eval=500,
    pop_size=50,
    seed=algorithm_seed,
    save_csv=True,
    csv_path=f"results/nsga2_{problem_seed}_{algorithm_seed}.csv",
)

print("\nTop 5 feasible solutions (by f1):")
print(feas.sort_values("f1", ascending=True).head())