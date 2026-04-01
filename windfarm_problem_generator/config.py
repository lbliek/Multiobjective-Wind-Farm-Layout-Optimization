from dataclasses import dataclass


@dataclass(frozen=True)
class GeneratorConfig:
    """
    User-facing configuration for the problem generator.

    Notes
    -----
    - The generator is deterministic for a fixed seed.
    - Coverage percentages refer to the fraction of the 1x1 unit square covered
      by the corresponding polygon.
    """

    # Number of different problem instances to generate.
    n_designs: int = 5

    # Fixed seed for deterministic generation.
    seed: int = 7

    # Maximum number of generation attempts.
    max_attempts: int = 60000

    # ---------- Available area polygon ----------
    # Polygon type: "nonconvex" or "convex".
    feasible_mode: str = "nonconvex"

    # Minimum and maximum number of vertices for the available area polygon.
    feasible_min_vertices: int = 4
    feasible_max_vertices: int = 10

    # Margin used when sampling the initial center of the available area polygon.
    feasible_center_margin: float = 0.15

    # Radius range used when generating the initial available area polygon.
    feasible_r_min: float = 0.05
    feasible_r_max: float = 0.50

    # Target fraction of the 1x1 solution space covered by the available area polygon.
    target_feasible_coverage_percent: float = 80.0

    # Allowed deviation from the target available-area coverage.
    feasible_tolerance_percent: float = 2.0

    # ---------- Oil & gas polygon ----------
    # Side length of the larger context square centered on the solution space.
    context_side: float = 25.0

    # Number of vertices for the oil & gas polygon.
    reservoir_min_vertices: int = 8
    reservoir_max_vertices: int = 12

    # Radius range used when generating the initial oil & gas polygon.
    reservoir_r_min: float = 0.6
    reservoir_r_max: float = 5.0

    # Target fraction of the 1x1 solution space covered by the oil & gas polygon.
    target_reservoir_coverage_percent: float = 15.0

    # Allowed deviation from the target oil & gas coverage.
    reservoir_tolerance_percent: float = 1.0

    # Whether points on polygon boundaries are treated as inside.
    allow_boundary: bool = True
