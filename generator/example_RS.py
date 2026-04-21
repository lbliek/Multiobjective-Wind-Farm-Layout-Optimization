from config import GeneratorConfig
from generator import generate_problem_instances
from evaluation import WindFarmEvaluator
from optimisation.randomsearch import run_random_search

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
seed = config.seed


df, feas = run_random_search(
    evaluator,
    n_eval=5000,
    hub_bounds=(1.0, 1.5),  
    seed=2026,
    save_csv=True,
    csv_path=f"results/random_search_seed{seed}.csv",
)


# 4. 打印一些结果
print("\nTop 5 feasible solutions (by f1):")
print(feas.sort_values("f1", ascending=False).head())