from typing import Tuple

from shapely.affinity import scale as shp_scale
from shapely.geometry import Polygon, box


UNIT_SQUARE = box(0.0, 0.0, 1.0, 1.0)

def make_context_square(side: float) -> Polygon:
    """
    Return a square of side length `side` centered on the unit-square center (0.5, 0.5).
    """
    half = side / 2.0
    cx, cy = 0.5, 0.5
    return box(cx - half, cy - half, cx + half, cy + half)

def ensure_valid_polygon(poly: Polygon) -> Polygon:
    """
    Repair minor invalidity and ensure that a single Polygon is returned.
    """
    if poly.is_empty:
        return poly

    if not poly.is_valid:
        poly = poly.buffer(0)

    if poly.geom_type == "MultiPolygon":
        poly = max(poly.geoms, key=lambda g: g.area)

    return poly

def coverage_in_unit_square(poly: Polygon) -> float:
    """
    Coverage = area(poly ∩ unit_square). The unit-square area equals 1.
    """
    if poly.is_empty:
        return 0.0
    return float(poly.intersection(UNIT_SQUARE).area)

def scale_about_centroid(poly: Polygon, scale_factor: float) -> Polygon:
    """
    Uniformly scale a polygon about its centroid.
    """
    c = poly.centroid
    return shp_scale(poly, xfact=scale_factor, yfact=scale_factor, origin=(c.x, c.y))

def tune_polygon_uniform_scale_to_coverage(
    poly: Polygon,
    target_cov: float,
    tol: float,
    interval_tol: float = 1e-12,
    max_scale: float = 1e6,
) -> Tuple[Polygon, float, float]:
    """
    Tune a polygon by uniform scaling so that its coverage inside the unit square
    matches `target_cov` within tolerance.

    Returns
    -------
    tuned_polygon, achieved_coverage, scale_factor
    """
    poly = ensure_valid_polygon(poly)

    if poly.is_empty or poly.area <= 1e-15:
        raise ValueError("Degenerate polygon cannot be tuned.")

    if not (0.0 < target_cov < 1.0):
        raise ValueError("target_cov must be in (0, 1).")

    s_low = 0.0
    s_high = 1.0

    tuned_high = ensure_valid_polygon(scale_about_centroid(poly, s_high))
    cov_high = coverage_in_unit_square(tuned_high)

    if abs(cov_high - target_cov) <= tol:
        return tuned_high, cov_high, s_high

    while cov_high < target_cov:
        s_high *= 2.0
        if s_high > max_scale:
            raise RuntimeError("Failed to bracket target coverage. Adjust generation parameters.")
        tuned_high = ensure_valid_polygon(scale_about_centroid(poly, s_high))
        cov_high = coverage_in_unit_square(tuned_high)

    while True:
        s_mid = 0.5 * (s_low + s_high)
        tuned_mid = ensure_valid_polygon(scale_about_centroid(poly, s_mid))
        cov_mid = coverage_in_unit_square(tuned_mid)

        if abs(cov_mid - target_cov) <= tol:
            return tuned_mid, cov_mid, s_mid

        if (s_high - s_low) <= interval_tol:
            return tuned_mid, cov_mid, s_mid

        if cov_mid < target_cov:
            s_low = s_mid
        else:
            s_high = s_mid
