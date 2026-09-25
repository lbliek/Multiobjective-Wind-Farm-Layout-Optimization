import numpy as np
import pandas as pd


def sample_solution(n_turbines: int, rng=None):
    """Sample turbine coordinates in [0, 1] x [0, 1]."""
    if rng is None:
        rng = np.random.default_rng()

    turbine_coords = []

    for _ in range(n_turbines):
        xi = rng.uniform(0.0, 1.0)
        yi = rng.uniform(0.0, 1.0)
        turbine_coords.append((xi, yi))

    xs = [p[0] for p in turbine_coords]
    ys = [p[1] for p in turbine_coords]

    return np.array(xs + ys, dtype=float)


def run_random_search(
    evaluator,
    hub,
    n_eval: int = 500,
    seed: int = 2026,
    save_csv: bool = True,
    csv_path: str = "random_search_results.csv",
):
    """Randomly sample turbine layouts using one fixed hub."""
    hub = np.asarray(hub, dtype=float)

    rng = np.random.default_rng(seed)
    rows = []

    for i in range(n_eval):
        # Candidate contains turbine coordinates only
        x = sample_solution(
            n_turbines=evaluator.n_turbines,
            rng=rng,
        )

        # The hub is fixed for all evaluations
        res = evaluator.evaluate(x, hub)

        f13 = res["f13"] #combine objectives 1 and 3 into one
        f2 = res["f2"]
        #f3 = res["f3"]
        g1 = res["g1"]
        g2 = res["g2"]
        g3 = res["g3"]

        feasible = int((g1 <= 0) and (g2 <= 0) and (g3 <= 0))

        rows.append({
            "eval_id": i,
            "x": list(x),
            "hub": list(hub),
            "f13": float(f13),
            "f2": float(f2),
            #"f3": float(f3),
            "g1": float(g1),
            "g2": float(g2),
            "g3": float(g3),
            "feasible": feasible,
        })

    df = pd.DataFrame(rows)

    if save_csv:
        df.to_csv(csv_path, index=False)
        print(f"Saved results to {csv_path}")

    feasible_rate = df["feasible"].mean()
    n_feasible = int(df["feasible"].sum())
    print(f"feasible_rate = {feasible_rate:.4f} ({n_feasible}/{len(df)})")

    feas = df[df["feasible"] == 1].copy()

    return df, feas