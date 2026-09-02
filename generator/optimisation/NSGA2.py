import numpy as np
import pandas as pd

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.callback import Callback


def decode_solution(candidate, n_turbines):
    """
    Decode candidate vector into:
    - x: turbine decision vector [x1,...,xn,y1,...,yn]
    - hub: [hub_x, hub_y]
    """
    candidate = np.asarray(candidate, dtype=float)

    x = candidate[: 2 * n_turbines]
    hub = candidate[2 * n_turbines: 2 * n_turbines + 2].tolist()

    return x, hub


def sample_solution_nsga2(problem, n_turbines: int, hub_outer_bound=1.5, rng=None):
    """
    Generate one initial solution for NSGA2.

    Turbines:
        sampled in [0,1] x [0,1]

    Hub:
        sampled uniformly from [0, hub_outer_bound]^2 \ [0,1]^2
    """
    if rng is None:
        rng = np.random.default_rng()

    turbine_coords = []

    # sample turbines only in [0,1] x [0,1]
    for _ in range(n_turbines):
        xi = rng.uniform(0.0, 1.0)
        yi = rng.uniform(0.0, 1.0)
        turbine_coords.append((xi, yi))

    xs = [p[0] for p in turbine_coords]
    ys = [p[1] for p in turbine_coords]
    x = np.array(xs + ys, dtype=float)

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

    return x, hub


class NSGA2Problem(ElementwiseProblem):
    def __init__(self, evaluator):
        self.evaluator = evaluator
        n = evaluator.n_turbines
        hub_outer_bound = evaluator.problem.hub_outer_bound

        xl = np.concatenate([
            np.zeros(2 * n),
            np.array([0.0, 0.0])
        ])

        xu = np.concatenate([
            np.ones(2 * n),
            np.array([hub_outer_bound, hub_outer_bound])
        ])

        super().__init__(
            n_var=2 * n + 2,
            n_obj=3,
            n_constr=3,
            xl=xl,
            xu=xu
        )

    def _evaluate(self, candidate, out, *args, **kwargs):
        x, hub = decode_solution(candidate, self.evaluator.n_turbines)
        res = self.evaluator.evaluate(x, hub)

        out["F"] = [res["f1"], res["f2"], res["f3"]]
        out["G"] = [res["g1"], res["g2"], res["g3"]]


class MyCallback(Callback):
    def __init__(self):
        super().__init__()
        self.X = []
        self.F = []
        self.G = []
        self._added_initial_pop = False

    def notify(self, algorithm):
        if not self._added_initial_pop:
            pop = algorithm.pop
            self.X.append(pop.get("X"))
            self.F.append(pop.get("F"))
            self.G.append(pop.get("G"))
            self._added_initial_pop = True

        off = getattr(algorithm, "off", None)
        if off is not None and len(off) > 0:
            self.X.append(off.get("X"))
            self.F.append(off.get("F"))
            self.G.append(off.get("G"))


def make_initial_population(evaluator, pop_size: int, seed: int = 2026):
    """
    Generate an initial feasible population using NSGA2's own sampling logic.
    """
    rng = np.random.default_rng(seed)
    X_init = []

    for _ in range(pop_size):
        x, hub = sample_solution_nsga2(
            problem=evaluator.problem,
            n_turbines=evaluator.n_turbines,
            hub_outer_bound=evaluator.problem.hub_outer_bound,
            rng=rng,
        )
        candidate = np.concatenate([x, np.asarray(hub, dtype=float)])
        X_init.append(candidate)

    return np.array(X_init, dtype=float)


def run_nsga2(
    evaluator,
    n_eval: int = 500,
    pop_size: int = 50,
    seed: int = 2026,
    save_csv: bool = True,
    csv_path: str = "nsga2_results.csv",
):
    problem = NSGA2Problem(evaluator)
    callback = MyCallback()

    X_init = make_initial_population(
        evaluator=evaluator,
        pop_size=pop_size,
        seed=seed,
    )

    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=X_init,
    )

    res = minimize(
        problem,
        algorithm,
        termination=("n_eval", n_eval),
        seed=seed,
        callback=callback,
        verbose=False
    )

    n_var = 2 * evaluator.n_turbines + 2

    X_all = np.vstack(callback.X) if len(callback.X) else np.empty((0, n_var))
    F_all = np.vstack(callback.F) if len(callback.F) else np.empty((0, 3))
    G_all = np.vstack(callback.G) if len(callback.G) else np.empty((0, 3))

    n = min(n_eval, len(X_all))
    X_all, F_all, G_all = X_all[:n], F_all[:n], G_all[:n]

    g1 = G_all[:, 0]
    g2 = G_all[:, 1]
    g3 = G_all[:, 2]
    feasible = ((g1 <= 0) & (g2 <= 0) & (g3 <= 0)).astype(int)

    rows = []
    for i in range(n):
        candidate = X_all[i]
        x, hub = decode_solution(candidate, evaluator.n_turbines)

        row = {
            "eval_id": i,
            "x": list(x),
            "hub": hub,
            "f1": float(F_all[i, 0]),
            "f2": float(F_all[i, 1]),
            "f3": float(F_all[i, 2]),
            "g1": float(g1[i]),
            "g2": float(g2[i]),
            "g3": float(g3[i]),
            "feasible": int(feasible[i]),
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    if save_csv:
        df.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")

    feasible_rate = df["feasible"].mean()
    n_feasible = int(df["feasible"].sum())
    print(f"feasible_rate = {feasible_rate:.4f} ({n_feasible}/{len(df)})")

    feas = df[df["feasible"] == 1].copy()

    return df, feas, res