import numpy as np
import pandas as pd


def sample_solution(problem, n_turbines: int, hub_bounds=(1.0, 1.5), rng=None, max_tries: int = 10000):
    """
    Randomly sample one candidate solution.

    Turbines:
        sampled in [0,1] x [0,1], but must satisfy problem.feasibility_indicator(x, y) == 1

    Hub:
        sampled in [hub_bounds[0], hub_bounds[1]]^2, and must also satisfy
        problem.feasibility_indicator(hx, hy) == 1

    Returns
    -------
    x : np.ndarray
        Turbine decision vector in the form [x1, x2, ..., y1, y2, ...].
    hub : list[float]
        Hub coordinate [hub_x, hub_y].
    """
    if rng is None:
        rng = np.random.default_rng()

    turbine_coords = []

    # sample turbines
    for _ in range(n_turbines):
        accepted = False
        for _ in range(max_tries):
            xi = rng.uniform(0.0, 1.0)
            yi = rng.uniform(0.0, 1.0)

            if problem.feasibility_turbine(xi, yi) == 1:
                turbine_coords.append((xi, yi))
                accepted = True
                break

        if not accepted:
            raise RuntimeError("Failed to sample a feasible turbine location within max_tries.")

    xs = [p[0] for p in turbine_coords]
    ys = [p[1] for p in turbine_coords]
    x = np.array(xs + ys, dtype=float)

    # sample hub
    # sample hub in [0, hub_bounds[1]] x [0, hub_bounds[1]], but outside [0,1] x [0,1]
    accepted = False
    for _ in range(max_tries):
        hx = rng.uniform(0.0, hub_bounds[1])
        hy = rng.uniform(0.0, hub_bounds[1])

        outside_unit_square = (hx > 1.0) or (hy > 1.0)

        if outside_unit_square and problem.feasibility_hub(hx, hy) == 1:
            hub = [float(hx), float(hy)]
            accepted = True
            break

    if not accepted:
        raise RuntimeError("Failed to sample a feasible hub location within max_tries.")

    return x, hub


def run_random_search(
    evaluator,
    n_eval: int = 500,
    hub_bounds=(1.0, 1.5),
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
            hub_bounds=hub_bounds,
            rng=rng,
        )

        res = evaluator.evaluate(x, hub)

        f1 = res["f1"]
        f2 = res["f2"]
        f3 = res["f3"]
        g1 = res["g1"]
        g2 = res["g2"]

        feasible = int((g1 <= 0) and (g2 <= 0))

        row = {
            "eval_id": i,
            "x": list(np.asarray(x, dtype=float)),
            "hub": list(np.asarray(hub, dtype=float)),
            "f1": float(f1),
            "f2": float(f2),
            "f3": float(f3),
            "g1": float(g1),
            "g2": float(g2),
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