from windfarm_problem import GeneratorConfig, generate_problem_instances
from windfarm_problem.visualization import plot_problem

config = GeneratorConfig(
    n_designs=3,
    seed=7,
    target_feasible_coverage_percent=80.0,
    target_reservoir_coverage_percent=15.0,
)

problems = generate_problem_instances(config)

problem_1 = problems[1]

print("Available area indicator at (0.30, 0.60):", problem_1.available_area_indicator(0.30, 0.60))
print("Oil & gas indicator at (0.30, 0.60):", problem_1.oil_gas_indicator(0.30, 0.60))
print("Overall feasibility at (0.30, 0.60):", problem_1.feasibility_indicator(0.30, 0.60))
print("Detailed check at (0.30, 0.60):", problem_1.check_point(0.30, 0.60))

plot_problem(problem_1, title="Toy problem")
