
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import truncnorm
from shapely.geometry import box
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import minimum_spanning_tree
from geometry import UNIT_SQUARE


def plot_problem(problem, x=None, hub=None, evaluator=None, len_plot=1.6, title: str = "Wind farm problem", path="results/chosen_layout.png") -> None:
    """
    Plot the 1x1 solution space, feasible region, reservoir regions, Reservoir centres
    optionally turbine locations, hub locations and  bird group.

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
            ax.fill(rx, ry, alpha=0.30, color="orange", label="Reservoir")
        else:
            ax.fill(rx, ry, alpha=0.30, color="orange")

        ax.plot(rx, ry, linewidth=1.8, color="orange")

        visible_part = reservoir.intersection(view_box)
        if not visible_part.is_empty:
            c = visible_part.centroid
            ax.text(c.x, c.y, f"R{i}", ha="center", va="center", fontsize=10)

        # Reservoir centres / platforms
        if hasattr(problem, "reservoir_centres") and problem.reservoir_centres:
            if i - 1 < len(problem.reservoir_centres):
                centres = problem.reservoir_centres[i - 1]

                for j, (cx, cy) in enumerate(centres, start=1):

                    if 0.0 <= cx <= len_plot and 0.0 <= cy <= len_plot:
                        label = "Platform" if (i == 1 and j == 1) else None

                        ax.scatter(
                            cx,
                            cy,
                            s=90,
                            marker="^",
                            color="purple",
                            label=label,
                            zorder=5,
                        )

                        ax.text(
                            cx + 0.015,
                            cy + 0.015,
                            f"P{i}.{j}",
                            fontsize=9,
                            color="purple",
                        )

                        # Optional: draw radius circle
                        if hasattr(problem, "reservoir_centre_radius"):
                            circle = plt.Circle(
                                (cx, cy),
                                problem.reservoir_centre_radius,
                                fill=False,
                                color="purple",
                                linestyle="--",
                                linewidth=1.0,
                                alpha=0.7,
                            )
                            ax.add_patch(circle)


    # External reservoir
    external_reservoir = getattr(problem, "external_reservoir", None)

    if external_reservoir is not None and not external_reservoir.is_empty:
        ex, ey = external_reservoir.exterior.xy

        ax.fill(ex, ey, alpha=0.22, color="brown", label="External reservoir")
        ax.plot(ex, ey, linewidth=1.8, color="brown")

        visible_part = external_reservoir.intersection(view_box)
        if not visible_part.is_empty:
            c = visible_part.centroid
            ax.text(c.x, c.y, "ER", ha="center", va="center", fontsize=10)

        external_centres = getattr(problem, "external_reservoir_centres", [])

        for j, (cx, cy) in enumerate(external_centres, start=1):
            if True:
            # if 0.0 <= cx <= len_plot and 0.0 <= cy <= len_plot:

                if 0.0 <= cx <= len_plot and 0.0 <= cy <= len_plot:
                    label = "External platform" if j == 1 else None

                    ax.scatter(
                        cx,
                        cy,
                        s=90,
                        marker="^",
                        color="purple",
                        label=label,
                        zorder=5,
                    )

                    ax.text(
                        cx + 0.015,
                        cy + 0.015,
                        f"EP{j}",
                        fontsize=9,
                        color="purple",
                    )

                if hasattr(problem, "reservoir_centre_radius"):
                    circle = plt.Circle(
                        (cx, cy),
                        problem.reservoir_centre_radius,
                        fill=False,
                        color="purple",
                        linestyle="--",
                        linewidth=1.0,
                        alpha=0.7,
                    )
                    ax.add_patch(circle)


    # Turbine locations
    if x is not None:
        x = np.asarray(x, dtype=float).reshape(-1)

        if len(x) % 2 != 0:
            raise ValueError("x must have even length: [x1, x2, ..., y1, y2, ...]")

        n_turbines = len(x) // 2
        xs = x[:n_turbines]
        ys = x[n_turbines:]
        coords = np.column_stack((xs, ys))

  
        ax.scatter(coords[:, 0], coords[:, 1], s=250, marker="1", color="blue", label="Turbines")
        for i, (tx, ty) in enumerate(coords, start=1):
            ax.text(tx + 0.015, ty + 0.015, f"T{i}", fontsize=10)

        # Hub
        if hub is not None:
            hub = np.asarray(hub, dtype=float).reshape(-1)
            if hub.shape[0] != 2:
                raise ValueError("substation must be a 2D coordinate like [sub_x, sub_y].")

            ax.scatter(hub[0], hub[1], s=120, marker="s", color="red", label="Substation")
            ax.text(hub[0] + 0.015, hub[1] + 0.015, "S", fontsize=10)

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
        bird_std = np.abs(evaluator.bird_mean / evaluator.x_sigma)
        birds_m = truncnorm.rvs(
            evaluator.x_sigma,
            evaluator.x_sigma + evaluator.farm_length / bird_std,
            loc=evaluator.bird_mean,
            scale=bird_std,
            size=evaluator.nr_birds,
            random_state=2026
        )

        # Randomly generate birds where the bird corridor is aligned with the horizontal axis y=0
        # (this corresponds to an angle of 0)
        birds_y = birds_m / evaluator.farm_length # distance from bird corridor and wind farm corner point
        rng_plot = np.random.default_rng(evaluator.seed + 1)
        birds_x = rng_plot.uniform(0, 1, size=len(birds_y)) # overwrite this later


        # Rotate birds for an angle other than 0
        angle = evaluator.bird_angle
        rad = angle * np.pi / 180  # convert angle to radians

        if 0 <= angle <= 90:
            Px = 1
            Py = 0 #x and y of point where bird corridor touches wind farm (corner point)

            for i in range(len(birds_y)):
                minx = Px-birds_y[i]/(np.sin(rad)+0.000000001)
                miny = Py
                if minx<0:
                    print('test', minx)
                    miny = -1*np.tan(rad+0.000000001)*minx
                    minx = 0
                if minx>1:
                    minx=1
                    print('This should not happen?')
                maxx = Px
                maxy = Py+birds_y[i]/(np.cos(rad)+0.000000001)
                if maxy>1:
                    maxy=1
                if maxy<0:
                    maxy=0
                p_min = np.asarray([minx, miny])
                p_max = np.asarray([maxx, maxy])
                maxdist = np.sqrt(np.dot(p_max-p_min,p_max-p_min)) #distance from one edge to the other
                z = rng_plot.uniform(0, maxdist, size=1)[0] # distance along line
                birds_x[i] = minx + z*np.cos(rad)
                birds_y[i] = miny + z*np.sin(rad)
        elif 90 < angle <= 180:
            Px = 1
            Py = 1

            for i in range(len(birds_y)):
                minx = Px
                miny = Py+birds_y[i]/(np.cos(rad)+0.000000001)
                if miny < 0:
                    minx -= -1 * np.tan(rad - np.pi/2 + 0.000000001) * miny
                    miny = 0
                if miny > 1:
                    miny = 1
                    print('This should not happen?')
                maxx = Px-birds_y[i]/(np.sin(rad)+0.000000001)
                if maxx > 1:
                    maxx=1
                if maxx < 0:
                    maxx = 0
                maxy = Py
                print(minx,maxx,miny,maxy)
                p_min = np.asarray([minx, miny])
                p_max = np.asarray([maxx, maxy])
                maxdist = np.sqrt(np.dot(p_max-p_min,p_max-p_min)) #distance from one edge to the other
                z = rng_plot.uniform(0, maxdist, size=1)[0] # distance along line
                birds_x[i] = minx + z*np.cos(rad)
                birds_y[i] = miny + z*np.sin(rad)

        elif 180 < angle <= 270:
            Px = 0
            Py = 1

            rad2 = rad-np.pi


            for i in range(len(birds_y)):
                minx = Px + birds_y[i] / (np.sin(rad2) + 0.000000001)
                miny = Py
                if minx > 1:
                    miny -= np.tan(rad - np.pi + 0.000000001) * minx
                    minx = 1
                if minx < 0:
                    minx = 0
                    print('This should not happen?')

                maxx = Px
                maxy = Py - birds_y[i] / (np.cos(rad2) + 0.000000001)
                if maxy > 1:
                    maxy = 1
                if maxy < 0:
                    maxy = 0
                print(minx, maxx, miny, maxy)
                p_min = np.asarray([minx, miny])
                p_max = np.asarray([maxx, maxy])
                maxdist = np.sqrt(np.dot(p_max - p_min, p_max - p_min))  # distance from one edge to the other
                z = rng_plot.uniform(0, maxdist, size=1)[0]  # distance along line
                birds_x[i] = minx - z * np.cos(rad2)
                birds_y[i] = miny - z * np.sin(rad2)
        elif 270 < angle <= 360:
            Px = 0
            Py = 0
            rad2 = rad - 1.5*np.pi

            for i in range(len(birds_y)):
                minx = Px
                miny = Py + birds_y[i] / (np.sin(rad2) + 0.000000001)
                if miny > 1:
                    minx +=  np.tan(rad - 1.5*np.pi + 0.000000001) * miny
                    miny = 1
                if miny < 0:
                    miny = 0
                    print('This should not happen?')
                maxx = Px + birds_y[i] / (np.cos(rad2) + 0.000000001)
                if maxx > 1:
                    maxx = 1
                if maxx < 0:
                    maxx = 0
                maxy = Py

                print(minx, maxx, miny, maxy)
                p_min = np.asarray([minx, miny])
                p_max = np.asarray([maxx, maxy])
                maxdist = np.sqrt(np.dot(p_max - p_min, p_max - p_min))  # distance from one edge to the other
                z = rng_plot.uniform(0, maxdist, size=1)[0]  # distance along line
                birds_x[i] = minx + z * np.cos(rad)
                birds_y[i] = miny + z * np.sin(rad)
        else:
            raise ValueError(f"Angle must be between 0 and 360 degrees, got {angle}")




        ax.scatter(birds_x, birds_y, s=10, marker='$v$', alpha=0.3, color="green", label="Birds")

    ax.set_title(title)
    ax.legend()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()
    