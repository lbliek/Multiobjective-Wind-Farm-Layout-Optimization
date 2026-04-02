import matplotlib.pyplot as plt

from geometry import UNIT_SQUARE


def plot_problem(problem, title: str = "Wind farm problem") -> None:
    """
    Plot the 1x1 solution space, the available area polygon, and the oil & gas polygon.
    Only the 1x1 area is visualized.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal", "box")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ux, uy = UNIT_SQUARE.exterior.xy
    ax.plot(ux, uy, linewidth=2, color="black")

    fx, fy = problem.feasible.exterior.xy
    ax.fill(fx, fy, alpha=0.18)
    ax.plot(fx, fy, linewidth=2)

    rx, ry = problem.reservoir.exterior.xy
    ax.fill(rx, ry, alpha=0.30, color="red")
    ax.plot(rx, ry, linewidth=1.8, color="red")

    ax.set_title(title)
    plt.show()
