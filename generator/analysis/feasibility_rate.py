import pandas as pd
from pathlib import Path

results_dir = Path("results")

files = {
    "Random Search": "random_search_1_2026.csv",
    "NSGA2": "nsga2_1_2026.csv",
    "EHVI": "qlognehvi_1_2026.csv",
    "ParEGO": "qlognparego_1_2026.csv",
}

for name, filename in files.items():
    path = results_dir / filename
    df = pd.read_csv(path)

    feasible_count = df["feasible"].sum()
    total_count = len(df)
    feasible_rate = feasible_count / total_count

    print(
        f"{name} feasible_rate = {feasible_rate:.4f} "
        f"({feasible_count}/{total_count})"
    )