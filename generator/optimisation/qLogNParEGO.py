import time
import warnings
import numpy as np
import pandas as pd
import torch

from botorch.models import SingleTaskGP, ModelListGP
from botorch.models.transforms.outcome import Standardize
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.multi_objective.parego import qLogNParEGO
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective


def decode_solution(candidate, n_turbines):
    candidate = np.asarray(candidate, dtype=float)
    x = candidate[: 2 * n_turbines]
    hub = candidate[2 * n_turbines: 2 * n_turbines + 2].tolist()
    return x, hub

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


def run_qlognparego(
    evaluator,
    n_eval: int = 500,
    n_initial: int = 50,
    seed: int = 2026,
    save_csv: bool = True,
    csv_path: str = "qlogparego_results.csv",
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cpu")
    dtype = torch.double

    n_turbines = evaluator.n_turbines
    dim_x = 2 * n_turbines
    dim_hub = 2
    # dim = dim_x + dim_hub

    def evaluate_blackbox(candidate_np: np.ndarray):
        x, hub = decode_solution(candidate_np, evaluator.n_turbines)
        res = evaluator.evaluate(x, hub)

        f1 = float(res["f1"])
        f2 = float(res["f2"])
        f3 = float(res["f3"])
        g1 = float(res["g1"])
        g2 = float(res["g2"])
        g3 = float(res["g3"])

        return x, hub, f1, f2, f3, g1, g2, g3

    def pack_Y(f1, f2, f3, g1, g2, g3):
        return torch.tensor(
            [[-f1, -f2, -f3, g1, g2, g3]],
            device=device,
            dtype=dtype,
        )

    def fit_model(X_train, Y_train):
        models = []

        for j in range(Y_train.shape[1]):
            y = Y_train[:, j:j + 1]

            if torch.std(y) < 1e-8:
                y = y + 1e-6 * torch.randn_like(y)

            gp = SingleTaskGP(
                X_train,
                y,
                outcome_transform=Standardize(m=1),
            )
            models.append(gp)

        model = ModelListGP(*models).to(device=device, dtype=dtype)
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        return model

    rng = np.random.default_rng(seed)

    X_list = []
    Y_list = []
    records = []

    n_initial = min(int(n_initial), int(n_eval))

    for i in range(n_initial):
        x, hub = sample_solution(
            problem=evaluator.problem,
            n_turbines=evaluator.n_turbines,
            rng=rng,
        )

        candidate_np = np.concatenate([x, np.asarray(hub, dtype=float)])

        candidate_tensor = torch.tensor(
            candidate_np,
            device=device,
            dtype=dtype,
        )

        x, hub, f1, f2, f3, g1, g2, g3 = evaluate_blackbox(candidate_np)

        Y_list.append(pack_Y(f1, f2, f3, g1, g2, g3))
        X_list.append(candidate_tensor.view(1, -1))

        feasible = int((g1 <= 0.0) and (g2 <= 0.0) and (g3 <= 0.0))

        records.append({
            "eval_id": i,
            "x": list(x),
            "hub": hub,
            "f1": f1,
            "f2": f2,
            "f3": f3,
            "g1": g1,
            "g2": g2,
            "g3": g3,
            "feasible": feasible,
        })

    X = torch.cat(X_list, dim=0)
    Y = torch.cat(Y_list, dim=0)

    lb = torch.cat([
        torch.zeros(dim_x, device=device, dtype=dtype),
        torch.tensor([0.0, 0.0], device=device, dtype=dtype),
    ])

    ub = torch.cat([
        torch.ones(dim_x, device=device, dtype=dtype),
        torch.tensor(
            [evaluator.problem.hub_outer_bound, evaluator.problem.hub_outer_bound],
            device=device,
            dtype=dtype,
        ),
    ])

    bounds = torch.stack([lb, ub])

    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([16]))

    t_start_total = time.perf_counter()
    model = None

    for t in range(n_initial, int(n_eval)):
        t_start_iter = time.perf_counter()

        model = fit_model(X, Y)

        weights = torch.rand(
            3,
            device=device,
            dtype=dtype,
        )
        weights = weights / weights.sum()

        acq = qLogNParEGO(
            model=model,
            X_baseline=X,
            scalarization_weights=weights,
            objective=IdentityMCMultiOutputObjective(outcomes=[0, 1, 2]),
            constraints=[
                lambda samples: samples[..., 3],
                lambda samples: samples[..., 4],
                lambda samples: samples[..., 5],
            ],
            sampler=sampler,
        )

        try:
            candidate, _ = optimize_acqf(
                acq_function=acq,
                bounds=bounds,
                q=1,
                num_restarts=1,
                raw_samples=20,
                options={"batch_limit": 5, "maxiter": 50},
            )

            x_next_internal = candidate.detach().squeeze(0)
            candidate_np = x_next_internal.cpu().numpy()

        except Exception as err:
            warnings.warn(
                f"qLogParEGO candidate generation failed: {err}. "
                "Using feasible random fallback."
            )

            x_fb, hub_fb = sample_solution(
                problem=evaluator.problem,
                n_turbines=evaluator.n_turbines,
                rng=rng,
            )

            candidate_np = np.concatenate([x_fb, np.asarray(hub_fb, dtype=float)])

            x_next_internal = torch.tensor(
                candidate_np,
                device=device,
                dtype=dtype,
            )

        x, hub, f1, f2, f3, g1, g2, g3 = evaluate_blackbox(candidate_np)
        y_next = pack_Y(f1, f2, f3, g1, g2, g3)

        X = torch.cat([X, x_next_internal.view(1, -1)], dim=0)
        Y = torch.cat([Y, y_next], dim=0)

        feasible = int((g1 <= 0.0) and (g2 <= 0.0) and (g3 <= 0.0))

        records.append({
            "eval_id": t,
            "x": list(x),
            "hub": hub,
            "f1": f1,
            "f2": f2,
            "f3": f3,
            "g1": g1,
            "g2": g2,
            "g3": g3,
            "feasible": feasible,
        })

        iter_time = time.perf_counter() - t_start_iter
        print(f"qLogParEGO progress: {t + 1}/{n_eval}, iter_time={iter_time:.3f}s")

    total_time = time.perf_counter() - t_start_total
    print(f"\nTotal qLogParEGO loop time: {total_time:.3f} seconds")

    df = pd.DataFrame(records)

    if save_csv:
        df.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")

    feasible_rate = df["feasible"].mean()
    n_feasible = int(df["feasible"].sum())
    print(f"feasible_rate = {feasible_rate:.4f} ({n_feasible}/{len(df)})")

    feas = df[df["feasible"] == 1].copy()

    return df, feas, model