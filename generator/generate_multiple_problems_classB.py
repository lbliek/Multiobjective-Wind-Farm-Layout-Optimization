from config import GeneratorConfig
from generator import generate_problem_instances
from evaluation import WindFarmEvaluator
from visualization import plot_problem
import numpy as np
import pickle



problem_seed = 444444444
algorithm_seed = 444444444

nr_of_problems = 50

config = GeneratorConfig(
    n_designs=nr_of_problems,
    seed=problem_seed,
    n_reservoirs=1,
    context_side=3,
    target_feasible_coverage_percent=92.0,
    reservoir_coverage_percent=3.0,
    # reservoir_coverage_percent=[10.0, 15.0, 7.0],
    max_reservoir_attempts=5000,
    external_reservoir_center=(5.8,5.8)
)


def generate_hub(hub_outer_bound,rng):
    # sample hub uniformly from [0, hub_outer_bound]^2 \ [0,1]^2

    if hub_outer_bound <= 1.0:
        raise ValueError("hub_outer_bound must be larger than 1.0.")

    area_top = 1.0 * (hub_outer_bound - 1.0)
    area_right = (hub_outer_bound - 1.0) * hub_outer_bound

    prob_top = area_top / (area_top + area_right)

    if rng.random() < prob_top:
        # top region: [0,1] x [1,hub_outer_bound]
        hx = rng.uniform(0.0, 1.0)
        hy = rng.uniform(1.0, hub_outer_bound)
    else:
        # right region: [1,hub_outer_bound] x [0,hub_outer_bound]
        hx = rng.uniform(1.0, hub_outer_bound)
        hy = rng.uniform(0.0, hub_outer_bound)

    hub = [float(hx), float(hy)]
    return hub

hub_outer_bound = 1.5
hub_list = np.zeros((nr_of_problems, 2))
rng = np.random.default_rng(problem_seed)

bird_corridor_angles = 360*np.random.rand(nr_of_problems)

problems = generate_problem_instances(config)

evaluatorparamlist = []

for problem_nr in range(nr_of_problems):
    problem = problems[problem_nr+1]

    hub = generate_hub(hub_outer_bound,rng)
    hub_list[problem_nr,:] = hub
    x = [0.25,0.5,0.75,0.33,0.67,0.75,0.75,0.75,0.25,0.25] # some default solution

    n_turbines = 5
    n_birds = 100
    bird_mean = -25000
    rotor_diameter = 126
    farm_length = 333.33 * 5
    seed = problem_seed

    bird_sigma = 8+8*np.random.rand() # random std for bird group between 8 and 16

    evaluatorparams =  {'n_turbines': n_turbines, 'bird_angle': bird_corridor_angles[problem_nr],
                        'nr_birds': n_birds, 'bird_mean': bird_mean, 'x_sigma': bird_sigma,
                        'rotor_diameter': rotor_diameter, 'farm_length': farm_length, 'seed': seed}
    evaluatorparamlist.append(evaluatorparams)


    # problem and solution plot
    #plot_problem(problem, x=x, hub=hub, len_plot=2, evaluator=evaluator, title="Problem", path=f"results/generated/problem{problem_nr+1}.png")


# Save the problems

with open("results/generated/problems.pkl", "wb") as f:
    pickle.dump(problems, f)

with open("results/generated/hub_list.pkl", "wb") as f:
    pickle.dump(hub_list, f)

with open("results/generated/evaluatorparams.pkl", "wb") as f:
    pickle.dump(evaluatorparamlist, f)


# To load the problems:

with open("results/generated/problems.pkl", "rb") as f:
    problems2 = pickle.load(f)
with open("results/generated/hub_list.pkl", "rb") as f:
    hub_list2 = pickle.load(f)
with open("results/generated/evaluatorparams.pkl", "rb") as f:
    evaluatorparamlist2 = pickle.load(f)


for problem_nr in range(len(problems2)):
    x = [0.25, 0.5, 0.75, 0.33, 0.67, 0.75, 0.75, 0.75, 0.25, 0.25]

    epl = evaluatorparamlist2[problem_nr]

    evaluator2 = WindFarmEvaluator(
        problem=problems[problem_nr+1],
        ensemble_file="Ensemble.pkl",
        n_turbines=epl['n_turbines'],
        bird_angle=epl['bird_angle'],
        nr_birds=epl['nr_birds'],
        bird_mean=epl['bird_mean'],
        x_sigma=epl['x_sigma'],
        rotor_diameter=epl['rotor_diameter'],
        farm_length=epl['farm_length'],
        seed=epl['seed']
    )
    plot_problem(problems[problem_nr+1], x=x, hub=hub_list2[problem_nr], len_plot=2, evaluator=evaluator2, title=f"Problem {problem_nr+1}",
                 path=f"results/generated/problem{problem_nr + 1}.png")
