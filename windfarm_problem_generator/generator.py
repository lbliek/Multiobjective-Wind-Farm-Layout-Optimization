from typing import Dict, Tuple

import numpy as np
from shapely.geometry import Polygon

from .config import GeneratorConfig
from .geometry import (
    UNIT_SQUARE,
    ensure_valid_polygon,
    make_context_square,
    tune_polygon_uniform_scale_to_coverage,
)
from .instance import ProblemInstance


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


def random_feasible_polygon_inside_unit(config: GeneratorConfig, rng: np.random.Generator) -> Polygon:
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


def random_reservoir_polygon_in_context(config: GeneratorConfig, rng: np.random.Generator, context: Polygon) -> Polygon:
    """
    Generate an initial oil & gas polygon in the larger context square.
    """
    n_vertices = int(rng.integers(config.reservoir_min_vertices, config.reservoir_max_vertices + 1))

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

    target_reservoir = float(config.target_reservoir_coverage_percent) / 100.0
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

        reservoir_raw = random_reservoir_polygon_in_context(config, rng, context)
        if reservoir_raw.is_empty:
            continue

        try:
            reservoir, reservoir_cov, _ = tune_polygon_uniform_scale_to_coverage(
                reservoir_raw,
                target_cov=target_reservoir,
                tol=tol_reservoir,
            )
        except Exception:
            continue

        idx = len(problems) + 1
        problems[idx] = ProblemInstance(
            feasible=feasible,
            reservoir=reservoir,
            feasible_cov=feasible_cov,
            reservoir_cov=reservoir_cov,
            allow_boundary=bool(config.allow_boundary),
        )

    if len(problems) < int(config.n_designs):
        raise RuntimeError(
            f"Only generated {len(problems)}/{config.n_designs} problems after {attempts} attempts. "
            "Try increasing max_attempts or loosening tolerances."
        )

    return problems
