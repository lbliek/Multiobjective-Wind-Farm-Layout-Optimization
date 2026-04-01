from dataclasses import dataclass

from shapely.geometry import Point

from .geometry import UNIT_SQUARE, ensure_valid_polygon


@dataclass
class ProblemInstance:
    """
    A single deterministic wind-farm layout problem.
    """
    feasible: object
    reservoir: object
    feasible_cov: float
    reservoir_cov: float
    allow_boundary: bool = True

    def __post_init__(self) -> None:
        self.feasible = ensure_valid_polygon(self.feasible)
        self.reservoir = ensure_valid_polygon(self.reservoir)

    def available_area_indicator(self, x: float, y: float) -> int:
        """
        Return 1 if the point is inside the available area polygon, otherwise 0.
        """
        p = Point(float(x), float(y))
        inside = self.feasible.covers(p) if self.allow_boundary else self.feasible.contains(p)
        return int(inside)

    def oil_gas_indicator(self, x: float, y: float) -> int:
        """
        Return 1 if the point is inside the oil & gas polygon, otherwise 0.
        """
        p = Point(float(x), float(y))
        inside = self.reservoir.covers(p) if self.allow_boundary else self.reservoir.contains(p)
        return int(inside)

    def feasibility_indicator(self, x: float, y: float) -> int:
        """
        Return 1 if the point is:
        - inside the 1x1 solution space,
        - inside the available area,
        - outside the oil & gas polygon.
        Otherwise return 0.
        """
        p = Point(float(x), float(y))

        in_solution = UNIT_SQUARE.covers(p) if self.allow_boundary else UNIT_SQUARE.contains(p)
        in_feasible = self.feasible.covers(p) if self.allow_boundary else self.feasible.contains(p)
        in_reservoir = self.reservoir.covers(p) if self.allow_boundary else self.reservoir.contains(p)

        return int(in_solution and in_feasible and (not in_reservoir))

    def check_point(self, x: float, y: float) -> dict:
        """
        Rich diagnostic version of the feasibility check.
        """
        p = Point(float(x), float(y))

        in_solution = UNIT_SQUARE.covers(p) if self.allow_boundary else UNIT_SQUARE.contains(p)
        in_feasible = self.feasible.covers(p) if self.allow_boundary else self.feasible.contains(p)
        in_reservoir = self.reservoir.covers(p) if self.allow_boundary else self.reservoir.contains(p)

        ok = bool(in_solution and in_feasible and (not in_reservoir))

        if not in_solution:
            reason = "outside_solution_space"
        elif not in_feasible:
            reason = "not_in_feasible_region"
        elif in_reservoir:
            reason = "inside_oil_gas_reservoir"
        else:
            reason = "ok"

        return {
            "ok": ok,
            "reason": reason,
            "in_solution_space": bool(in_solution),
            "in_feasible_region": bool(in_feasible),
            "in_reservoir": bool(in_reservoir),
        }
