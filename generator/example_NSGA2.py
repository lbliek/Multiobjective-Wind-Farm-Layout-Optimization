from config import GeneratorConfig
from generator import generate_problem_instances
from evaluation import WindFarmEvaluator
from optimisation.NSGA2 import run_nsga2


config = GeneratorConfig(
    n_designs=3,
    seed=2026,
    n_reservoirs=4,
    context_side=3,
    target_feasible_coverage_percent=95.0,
    target_reservoir_coverage_percent=20.0,
    hub_outer_bound=1.5,
)
seed = config.seed
problems = generate_problem_instances(config)
problem_2 = problems[2]

evaluator = WindFarmEvaluator(
    problem_2,
    ensemble_file="Ensemble.pkl",
    n_turbines=5,
)

df, feas, res = run_nsga2(
    evaluator,
    n_eval=5000,
    pop_size=50,
    seed=2026,
    save_csv=True,
    csv_path=f"results/nsga2_seed{seed}.csv",
)

print("\nTop 5 feasible solutions (by f1):")
print(feas.sort_values("f1", ascending=True).head())