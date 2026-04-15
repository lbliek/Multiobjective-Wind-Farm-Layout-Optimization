
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import truncnorm
from shapely.geometry import box
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import minimum_spanning_tree
from geometry import UNIT_SQUARE


def plot_problem(problem, x=None, hub=None, evaluator=None, len_plot=1.6, title: str = "Wind farm problem") -> None:
    """
    Plot the 1x1 solution space, feasible region, reservoir regions,
    optionally turbine locations, and optionally bird group.

    Parameters
    ----------
    problem : ProblemInstance
        Generated problem instance.
    x : array-like or None
        Candidate solution in the form [x1, x2, ..., y1, y2, ...].
    evaluator : WindFarmEvaluator or None
        If provided, bird group will be plotted using evaluator parameters.
    len_plot : float
        Plot range for both x and y axes.
    title : str
        Plot title.
    """
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect("equal", "box")
    ax.set_xlim(0, len_plot)
    ax.set_ylim(0, len_plot)

    # 1x1 solution space
    ux, uy = UNIT_SQUARE.exterior.xy
    ax.plot(ux, uy, linewidth=2, color="black")

    # Feasible region
    fx, fy = problem.feasible.exterior.xy
    ax.fill(fx, fy, alpha=0.18, label="Feasible region")
    ax.plot(fx, fy, linewidth=2)

    # Reservoir regions
    view_box = box(0.0, 0.0, len_plot, len_plot)
    for i, reservoir in enumerate(problem.reservoirs, start=1):
        rx, ry = reservoir.exterior.xy
        if i == 1:
            ax.fill(rx, ry, alpha=0.30, color="orange", label="Reservoir") #lightcoral
        else:
            ax.fill(rx, ry, alpha=0.30, color="orange")
        ax.plot(rx, ry, linewidth=1.8, color="orange")

        visible_part = reservoir.intersection(view_box)
        if not visible_part.is_empty:
            c = visible_part.centroid
            ax.text(c.x, c.y, f"R{i}", ha="center", va="center", fontsize=10)

    # Turbine locations
    if x is not None:
        x = np.asarray(x, dtype=float).reshape(-1)

        if len(x) % 2 != 0:
            raise ValueError("x must have even length: [x1, x2, ..., y1, y2, ...]")

        n_turbines = len(x) // 2
        xs = x[:n_turbines]
        ys = x[n_turbines:]
        coords = np.column_stack((xs, ys))

        ax.scatter(coords[:, 0], coords[:, 1], s=80, marker="o", color="blue", label="Turbines")
        for i, (tx, ty) in enumerate(coords, start=1):
            ax.text(tx + 0.015, ty + 0.015, f"T{i}", fontsize=10)

        # Hub
        if hub is not None:
            hub = np.asarray(hub, dtype=float).reshape(-1)
            if hub.shape[0] != 2:
                raise ValueError("hub must be a 2D coordinate like [hub_x, hub_y].")

            ax.scatter(hub[0], hub[1], s=120, marker="s", color="red", label="Hub")
            ax.text(hub[0] + 0.015, hub[1] + 0.015, "Hub", fontsize=10)

            # MST cables
            points = np.vstack([hub.reshape(1, 2), coords])   # point 0 = hub
            dist_matrix = cdist(points, points)
            mst = minimum_spanning_tree(dist_matrix).toarray()

            for i in range(mst.shape[0]):
                for j in range(mst.shape[1]):
                    if mst[i, j] > 0:
                        ax.plot(
                            [points[i, 0], points[j, 0]],
                            [points[i, 1], points[j, 1]],
                            "-k",
                            linewidth=1.5,
                        )


    # Bird group
    if evaluator is not None:
        bird_std = 25000 / evaluator.x_sigma
        birds_m = truncnorm.rvs(
            evaluator.x_sigma,
            evaluator.x_sigma + evaluator.farm_length / bird_std,
            loc=evaluator.bird_mean,
            scale=bird_std,
            size=evaluator.nr_birds,
            random_state=2026
        )

        birds_x = birds_m / evaluator.farm_length
        rng_plot = np.random.default_rng(evaluator.seed + 1)
        birds_y = rng_plot.uniform(0, 1, size=len(birds_x))

        ax.scatter(birds_x, birds_y, s=10, alpha=0.3, color="green", label="Birds")

    ax.set_title(title)
    ax.legend()
    plt.show()