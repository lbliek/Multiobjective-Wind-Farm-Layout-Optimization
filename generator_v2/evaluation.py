

import pickle
import sys
import numpy as np
import xgboost as xgb
from scipy.stats import truncnorm
from instance import ProblemInstance
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import minimum_spanning_tree


class medClassifier:
    def __init__(self, classifiers=None):
        self.classifiers = classifiers or []

    def predict(self, X):
        predictions = []
        for classifier in self.classifiers:
            try:
                predictions.append(classifier.predict(X))
            except Exception:
                X_dmatrix = xgb.DMatrix(X)
                predictions.append(classifier.predict(X_dmatrix))

        return np.median(predictions, axis=0)


class WindFarmEvaluator:
    def __init__(
        self,
        problem: ProblemInstance,
        ensemble_file: str = "Ensemble.pkl",
        n_turbines: int = 5,
        nr_birds: int = 1000,
        bird_mean: float = -25000,
        x_sigma: float = 12,
        rotor_diameter: float = 126,
        farm_length: float = 333.33 * 5,
        seed: int = 2026,
    ):
        self.problem = problem
        self.n_turbines = n_turbines

        self.nr_birds = nr_birds
        self.bird_mean = bird_mean
        self.x_sigma = x_sigma
        self.rotor_diameter = rotor_diameter
        self.farm_length = farm_length
        self.seed = seed

        # 让 pickle 能找到当初保存 Ensemble.pkl 时使用的 __main__.medClassifier
        sys.modules["__main__"].medClassifier = medClassifier

        with open(ensemble_file, "rb") as f:
            self.ensemble = pickle.load(f)

    def _validate_x(self, x):
        x = np.asarray(x, dtype=float).reshape(-1)
        expected_dim = 2 * self.n_turbines
        if x.shape[0] != expected_dim:
            raise ValueError(f"x must have length {expected_dim}, got {x.shape[0]}")
        return x

    def _to_coords(self, x):
        x = self._validate_x(x)
        xs = x[:self.n_turbines]
        ys = x[self.n_turbines:]
        coords = np.column_stack((xs, ys))
        return coords
    
    def _validate_hub(self, hub):
        hub = np.asarray(hub, dtype=float).reshape(-1)
        if hub.shape[0] != 2:
            raise ValueError("hub must be a 2D coordinate like [hub_x, hub_y].")
        return hub
    
    def objective3(self, x, hub) -> float:
        coords = self._to_coords(x)
        hub = self._validate_hub(hub).reshape(1, 2)

        points = np.vstack([hub, coords])
        dist_matrix = cdist(points, points)
        mst = minimum_spanning_tree(dist_matrix)
        total_length = mst.sum()

        return float(total_length) * self.farm_length


    def objective1(self, x) -> float:
        x = self._validate_x(x)
        X = np.array([x], dtype=float)
        pred = self.ensemble.predict(X)
        return float(pred[0])

    def objective2(self, x) -> float:
        x = np.asarray(x, dtype=float)
        coords = self._to_coords(x)

        bird_std = 25000 / self.x_sigma
        birds = truncnorm.rvs(
            self.x_sigma,
            self.x_sigma + self.farm_length / bird_std,
            loc=self.bird_mean,
            scale=bird_std,
            size=self.nr_birds,
            random_state=self.seed
        )

        leftmost = np.min(coords[:, 0]) * self.farm_length
        threshold = leftmost - self.rotor_diameter
        close_birds = np.sum(birds >= threshold) / self.nr_birds
        return float(close_birds)

    def constraint1(self, x) -> float:
        coords = self._to_coords(x)

        min_dist = np.inf
        for turb in range(self.n_turbines - 1):
            dists = cdist([coords[turb]], coords[turb + 1:])
            next_min = np.min(dists)
            if next_min < min_dist:
                min_dist = next_min

        cv = 2 * self.rotor_diameter - min_dist * self.farm_length
        return float(cv)
    
    # check the feasibility of turbines and the hub
    def constraint2(self, x, hub) -> int:
        coords = self._to_coords(x)
        hub = self._validate_hub(hub)

        n_violate = 0

        for xi, yi in coords:
            if self.problem.feasibility_turbine(xi, yi) == 0:
                n_violate += 1

        if self.problem.feasibility_hub(hub[0], hub[1]) == 0:
            n_violate += 1

        return int(n_violate)

    def evaluate(self, x, hub):
        return {
            "f1": self.objective1(x),
            "f2": self.objective2(x),
            "f3": self.objective3(x, hub),
            "g1": self.constraint1(x),
            "g2": self.constraint2(x, hub),
        }