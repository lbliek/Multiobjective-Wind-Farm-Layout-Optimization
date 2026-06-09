import numpy as np
import pandas as pd

def sample_solution(problem, n_turbines: int, rng=None):
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
    hub_outer_bound = problem.hub_outer_bound

    if hub_outer_bound <= 1.0:
        raise ValueError("hub_outer_bound must be larger than 1.0.")

    # Region A: top region [0,1] x [1,hub_outer_bound]
    area_top = 1.0 * (hub_outer_bound - 1.0)

    # Region B: right region [1,hub_outer_bound] x [0,hub_outer_bound]
    area_right = (hub_outer_bound - 1.0) * hub_outer_bound

    prob_top = area_top / (area_top + area_right)

    if rng.random() < prob_top:
        hx = rng.uniform(0.0, 1.0)
        hy = rng.uniform(1.0, hub_outer_bound)
    else:
        hx = rng.uniform(1.0, hub_outer_bound)
        hy = rng.uniform(0.0, hub_outer_bound)

    hub = [float(hx), float(hy)]

    return x, hub


def run_random_search(
    evaluator,
    n_eval: int = 500,
    # hub_bounds=(1.0, 1.5),
    seed: int = 2026,
    save_csv: bool = True,
    csv_path: str = "random_search_results.csv",
):
    rng = np.random.default_rng(seed)
    rows = []

    for i in range(n_eval):
        x, hub = sample_solution(
            problem=evaluator.problem,
            n_turbines=evaluator.n_turbines,
            # hub_bounds=hub_bounds,
            rng=rng,
        )

        res = evaluator.evaluate(x, hub)

        f1 = res["f1"]
        f2 = res["f2"]
        f3 = res["f3"]
        g1 = res["g1"]
        g2 = res["g2"]
        g3 = res["g3"]

        feasible = int((g1 <= 0) and (g2 <= 0) and (g3 <= 0))   

        row = {
            "eval_id": i,
            "x": list(np.asarray(x, dtype=float)),
            "hub": list(np.asarray(hub, dtype=float)),
            "f1": float(f1),
            "f2": float(f2),
            "f3": float(f3),
            "g1": float(g1),
            "g2": float(g2),
            "g3": float(g3),
            "feasible": feasible,
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    if save_csv:
        df.to_csv(csv_path, index=False)
        print(f"Saved results to {csv_path}")

    feasible_rate = df["feasible"].mean()
    n_feasible = int(df["feasible"].sum())
    print(f"feasible_rate = {feasible_rate:.4f} ({n_feasible}/{len(df)})")

    feas = df[df["feasible"] == 1].copy()

    return df, feas