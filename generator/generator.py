from typing import Dict, Tuple, List
import numpy as np
from config import GeneratorConfig
from geometry import (
    UNIT_SQUARE,
    ensure_valid_polygon,
    make_context_square,
    tune_polygon_uniform_scale_to_coverage,
)
from instance import ProblemInstance
from shapely.geometry import Polygon, Point
from sklearn.cluster import KMeans


def random_star_polygon(
    n_vertices: int,
    rng: np.random.Generator,
    center: Tuple[float, float],
    r_min: float,
    r_max: float,
) -> Polygon:
    """
    Generate a simple star-shaped polygon from random polar coordinates.
    """
    cx, cy = center
    angles = np.sort(rng.uniform(0.0, 2.0 * np.pi, size=n_vertices))
    radii = rng.uniform(r_min, r_max, size=n_vertices)

    xs = cx + radii * np.cos(angles)
    ys = cy + radii * np.sin(angles)

    return ensure_valid_polygon(Polygon(np.column_stack([xs, ys])))


def random_feasible_polygon_inside_unit(
    config: GeneratorConfig,
    rng: np.random.Generator,
) -> Polygon:
    """
    Generate an initial available-area polygon inside the unit square.
    It is later tuned by uniform scaling to meet the requested coverage target.
    """
    n_vertices = int(rng.integers(config.feasible_min_vertices, config.feasible_max_vertices + 1))
    margin = float(config.feasible_center_margin)
    mode = str(config.feasible_mode).lower()

    cx = rng.uniform(margin, 1.0 - margin)
    cy = rng.uniform(margin, 1.0 - margin)

    if mode == "nonconvex":
        poly = random_star_polygon(
            n_vertices=n_vertices,
            rng=rng,
            center=(cx, cy),
            r_min=float(config.feasible_r_min),
            r_max=float(config.feasible_r_max),
        )
    elif mode == "convex":
        pts = rng.uniform(margin, 1.0 - margin, size=(n_vertices * 4, 2))
        poly = ensure_valid_polygon(Polygon(pts).convex_hull)
    else:
        raise ValueError("feasible_mode must be 'nonconvex' or 'convex'.")

    if poly.is_empty:
        return Polygon()

    if not UNIT_SQUARE.contains(poly):
        return Polygon()

    return poly


def random_reservoir_polygon_in_context(
    config: GeneratorConfig,
    rng: np.random.Generator,
    context: Polygon,
) -> Polygon:
    """
    Generate an initial oil & gas polygon in the larger context square.
    """
    n_vertices = int(
        rng.integers(
            config.reservoir_min_vertices,
            config.reservoir_max_vertices + 1,
        )
    )

    minx, miny, maxx, maxy = context.bounds
    cx = rng.uniform(minx, maxx)
    cy = rng.uniform(miny, maxy)

    return random_star_polygon(
        n_vertices=n_vertices,
        rng=rng,
        center=(cx, cy),
        r_min=float(config.reservoir_r_min),
        r_max=float(config.reservoir_r_max),
    )


def get_reservoir_coverage_targets(config: GeneratorConfig) -> List[float]:
    """
    Convert reservoir coverage configuration to a list of fractions.

    Examples
    --------
    n_reservoirs = 3
    reservoir_coverage_percent = 5.0
    -> [0.05, 0.05, 0.05]

    n_reservoirs = 3
    reservoir_coverage_percent = [5.0, 8.0, 3.0]
    -> [0.05, 0.08, 0.03]
    """
    n_reservoirs = int(config.n_reservoirs)

    if not (0 <= n_reservoirs <= 5):
        raise ValueError("n_reservoirs must be between 0 and 5.")

    cov = config.reservoir_coverage_percent

    if isinstance(cov, (int, float)):
        targets = [float(cov) / 100.0] * n_reservoirs
    else:
        if len(cov) != n_reservoirs:
            raise ValueError(
                "If reservoir_coverage_percent is a list, "
                "its length must equal n_reservoirs."
            )

        targets = [float(x) / 100.0 for x in cov]

    for target in targets:
        if not (0.0 < target < 1.0):
            raise ValueError(
                "Each reservoir coverage percent must be between 0 and 100."
            )

    return targets


def overlaps_existing_reservoirs(
    reservoir: Polygon,
    reservoirs: List[Polygon],
    eps: float = 1e-12,
) -> bool:
    """
    Check whether the new reservoir overlaps existing reservoirs.

    Positive-area overlap is forbidden.
    Boundary touching is allowed.
    """
    return any(
        reservoir.intersection(old).area > eps
        for old in reservoirs
    )

def decide_n_reservoir_centres(
    reservoir: Polygon,
    config: GeneratorConfig,
) -> int:
    """
    Decide number of centres based on full reservoir polygon area.
    """
    area = float(reservoir.area)
    small_threshold, medium_threshold = config.reservoir_centre_area_thresholds

    if area <= small_threshold:
        return 1
    elif area <= medium_threshold:
        return 2
    else:
        return 3


def sample_points_inside_polygon(
    polygon: Polygon,
    rng: np.random.Generator,
    n_samples: int,
    max_attempts: int,
) -> np.ndarray:
    """
    Uniformly sample valid points inside a polygon using rejection sampling.
    """
    if polygon.is_empty:
        raise ValueError("Cannot sample from an empty polygon.")

    minx, miny, maxx, maxy = polygon.bounds

    points = []
    attempts = 0

    while len(points) < n_samples and attempts < max_attempts:
        attempts += 1

        x = rng.uniform(minx, maxx)
        y = rng.uniform(miny, maxy)

        if polygon.contains(Point(x, y)):
            points.append([x, y])

    if len(points) == 0:
        rp = polygon.representative_point()
        return np.array([[float(rp.x), float(rp.y)]], dtype=float)

    return np.array(points, dtype=float)


def generate_reservoir_centres(
    reservoir: Polygon,
    config: GeneratorConfig,
    rng: np.random.Generator,
) -> List[Tuple[float, float]]:
    """
    Generate reservoir centres/platforms.

    Small reservoir:
        1 centre, using centroid.

    Medium / large reservoir:
        sample points inside reservoir, then use sklearn KMeans.
    """
    n_centres = decide_n_reservoir_centres(reservoir, config)

    if n_centres == 1:
        c = reservoir.centroid

        if not reservoir.contains(c):
            c = reservoir.representative_point()

        return [(float(c.x), float(c.y))]

    points = sample_points_inside_polygon(
        polygon=reservoir,
        rng=rng,
        n_samples=int(config.reservoir_centre_n_samples),
        max_attempts=int(config.reservoir_centre_max_sampling_attempts),
    )

    if len(points) < n_centres:
        return [(float(x), float(y)) for x, y in points]

    kmeans = KMeans(
        n_clusters=n_centres,
        random_state=int(rng.integers(0, 2**32 - 1)),
        n_init="auto",
    )

    kmeans.fit(points)
    kmeans_centres = kmeans.cluster_centers_

    centres = []

    for x, y in kmeans_centres:
        p = Point(float(x), float(y))

        if reservoir.contains(p):
            centres.append((float(x), float(y)))
        else:
            # KMeans centre can fall outside a non-convex polygon.
            # Replace it with the nearest sampled point, which is guaranteed inside.
            distances = np.linalg.norm(points - np.array([x, y]), axis=1)
            nearest = points[int(np.argmin(distances))]
            centres.append((float(nearest[0]), float(nearest[1])))

    return centres


def generate_problem_instances(config: GeneratorConfig) -> Dict[int, ProblemInstance]:
    """
    Generate deterministic problem instances.

    Parameters
    ----------
    config : GeneratorConfig
        User-facing generator settings.

    Returns
    -------
    dict[int, ProblemInstance]
        Dictionary mapping 1..n_designs to generated problem instances.
    """
    rng = np.random.default_rng(config.seed)
    context = make_context_square(float(config.context_side))

    target_feasible = float(config.target_feasible_coverage_percent) / 100.0
    tol_feasible = float(config.feasible_tolerance_percent) / 100.0

    reservoir_targets = get_reservoir_coverage_targets(config)
    tol_reservoir = float(config.reservoir_tolerance_percent) / 100.0

    problems: Dict[int, ProblemInstance] = {}
    attempts = 0

    while len(problems) < int(config.n_designs) and attempts < int(config.max_attempts):
        attempts += 1

        feasible_raw = random_feasible_polygon_inside_unit(config, rng)

        if feasible_raw.is_empty:
            continue

        try:
            feasible, feasible_cov, _ = tune_polygon_uniform_scale_to_coverage(
                feasible_raw,
                target_cov=target_feasible,
                tol=tol_feasible,
            )
        except Exception:
            continue


        reservoirs: List[Polygon] = []
        reservoir_covs: List[float] = []
        reservoir_centres: List[List[Tuple[float, float]]] = []

        ok = True

        for target_cov in reservoir_targets:
            placed = False

            for _ in range(int(config.max_reservoir_attempts)):
                reservoir_raw = random_reservoir_polygon_in_context(
                    config=config,
                    rng=rng,
                    context=context,
                )

                if reservoir_raw.is_empty:
                    continue

                try:
                    reservoir, reservoir_cov, _ = tune_polygon_uniform_scale_to_coverage(
                        reservoir_raw,
                        target_cov=target_cov,
                        tol=tol_reservoir,
                    )
                except Exception:
                    continue

                if overlaps_existing_reservoirs(reservoir, reservoirs):
                    continue

                centres = generate_reservoir_centres(
                    reservoir=reservoir,
                    config=config,
                    rng=rng,
                )

                reservoirs.append(reservoir)
                reservoir_covs.append(reservoir_cov)
                reservoir_centres.append(centres)

                placed = True
                break



            if not placed:
                ok = False
                break

        if not ok:
            continue

        idx = len(problems) + 1


        problems[idx] = ProblemInstance(
            feasible=feasible,
            reservoirs=reservoirs,
            feasible_cov=feasible_cov,
            reservoir_covs=reservoir_covs,
            reservoir_centres=reservoir_centres,
            reservoir_centre_radius=float(config.reservoir_centre_radius),
            allow_boundary=bool(config.allow_boundary),
            hub_outer_bound=float(config.hub_outer_bound),
        )



    if len(problems) < int(config.n_designs):
        raise RuntimeError(
            f"Only generated {len(problems)}/{config.n_designs} problems after {attempts} attempts. "
            "Try increasing max_attempts, increasing max_reservoir_attempts, "
            "reducing n_reservoirs, reducing reservoir coverage, or loosening tolerances."
        )

    return problems