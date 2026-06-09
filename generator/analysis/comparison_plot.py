import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.indicators.hv import HV


BATCH_SIZE = 1
REF_POINT = np.array([1.1, 1.1, 1.1], dtype=float)

results_dir = Path("results")

CSV_PATHS = {
    "Random": results_dir / "random_search_1_2026.csv",
    "NSGA2": results_dir / "nsga2_1_2026.csv",
    "EHVI": results_dir / "qlognehvi_1_2026.csv",
    "ParEGO": results_dir / "qlognparego_1_2026.csv",
}



f_min = np.array([-5.87787781e+01,  1.00000000e-03,  1.46898078e+03], dtype=float)
f_max = np.array([-1.25478811e+01,  1.00000000e+00,  4.59072239e+03], dtype=float)



denom = f_max - f_min
denom[denom == 0] = 1.0

hv_indicator = HV(ref_point=REF_POINT)


def compute_hv_curve(csv_path):
    df = pd.read_csv(csv_path)

    xs, hvs = [], []
    n = len(df)

    for end in range(BATCH_SIZE, n + 1, BATCH_SIZE):
        cur = df.iloc[:end]
        cur_feas = cur[cur["feasible"] == 1]

        xs.append(end)

        if len(cur_feas) == 0:
            hvs.append(0.0)
            continue

        F = cur_feas[["f1", "f2", "f3"]].to_numpy(dtype=float)

        fronts = NonDominatedSorting().do(F)
        F_best = F[fronts[0]]

        F_best_norm = (F_best - f_min) / denom
        hv_value = float(hv_indicator(F_best_norm))
        hvs.append(hv_value)

    return xs, hvs


plt.figure()

color_map = {
    "Random": "C0",
    "NSGA2": "C1",
    "EHVI": "C2",
    "ParEGO": "C3",
}

for label, path in CSV_PATHS.items():
    xs, hvs = compute_hv_curve(path)
    plt.plot(xs, hvs, label=label, color=color_map.get(label, None))

plt.xlabel("Evaluation")
plt.ylabel("HV")
plt.title(f"Cumulative HV Curve (batch={BATCH_SIZE})")
plt.legend()
plt.grid(True)

output_path = results_dir / "hv_curve_all_algorithms.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved figure to: {output_path}")