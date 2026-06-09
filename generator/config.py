from dataclasses import dataclass
from typing import List, Union


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
    seed: int = 1

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

    # ---------- hub ----------
    # outer_bound of the hub 
    hub_outer_bound: float = 1.5

    # ---------- Oil & gas polygon ----------
    # Number of reservoirs. Must be between 0 and 5.
    n_reservoirs: int = 3

    # Side length of the larger context square centered on the solution space.
    context_side: float = 25.0

    # Number of vertices for the oil & gas polygon.
    reservoir_min_vertices: int = 8
    reservoir_max_vertices: int = 12

    # Radius range used when generating the initial oil & gas polygon.
    reservoir_r_min: float = 0.6
    reservoir_r_max: float = 5.0

    # Target coverage percentage for each reservoir.
    # Can be:
    # - a single float, e.g. 5.0 means every reservoir covers about 5%
    # - a list, e.g. [5.0, 8.0, 3.0] means each reservoir has its own target coverage
    reservoir_coverage_percent: Union[float, List[float]] = 5.0

    # Maximum attempts for placing each reservoir without overlap.
    max_reservoir_attempts: int = 2000

    # Allowed deviation from the target oil & gas coverage.
    reservoir_tolerance_percent: float = 1.0

    # Whether points on polygon boundaries are treated as inside.
    allow_boundary: bool = True


    # ---------- Reservoir centre / platform ----------
    # Decide number of centres by full reservoir polygon area.
    # If reservoir.area <= first threshold: 1 centre
    # If reservoir.area <= second threshold: 2 centres
    # Otherwise: 3 centres
    reservoir_centre_area_thresholds: tuple[float, float] = (0.2, 0.5)

    # Radius around each reservoir centre/platform.
    # This will be used later as a constraint distance.
    reservoir_centre_radius: float = 0.1

    # Number of valid interior sample points used for KMeans.
    reservoir_centre_n_samples: int = 1000

    # Maximum random attempts when sampling points inside one reservoir.
    reservoir_centre_max_sampling_attempts: int = 20000
