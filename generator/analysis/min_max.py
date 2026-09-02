import pandas as pd
from pathlib import Path

results_dir = Path("results")

files = [
    "random_search_1_2026.csv",
    "nsga2_1_2026.csv",
    "qlognehvi_1_2026.csv",
    "qlognparego_1_2026.csv",
]

all_feas = []

for f in files:
    path = results_dir / f
    df = pd.read_csv(path)

    feas = df[df["feasible"] == 1][["f1", "f2", "f3"]]
    all_feas.append(feas)

all_feas = pd.concat(all_feas, ignore_index=True)

F = all_feas.to_numpy(dtype=float)

f_min = F.min(axis=0)
f_max = F.max(axis=0)

print("f_min =", f_min)
print("f_max =", f_max)