### evaluation.py defines how to evaluate the wind farm layout problem###

import pickle
import sys
import numpy as np
import xgboost as xgb
from scipy.stats import truncnorm, norm
from statistics import NormalDist
from instance import ProblemInstance
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import minimum_spanning_tree

class medClassifier:
    """
    Define median-based ensemble of surrogates.
    """
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
    '''
    Main wind farm problem (3 objectives + 3 constraints)
    '''
    def __init__(
        self,
        problem: ProblemInstance,
        ensemble_file: str = "Ensemble.pkl", # model used for obj1
        n_turbines: int = 5,
        nr_birds: int = 100,
        bird_mean: float = -25000, # bird_mean = -25000 m is the distance of the center of birds from the edge of the wind farm
        bird_angle: float = 270, # angle of bird_corridor in degrees (0-360), 270 degrees means the bird corridor is going down (south)
        x_sigma: float = 14, # decide std and bird group distribution
        rotor_diameter: float = 126,
        farm_length: float = 333.33 * 5,
        seed: int = 2026,
    ):
        self.problem = problem
        self.n_turbines = n_turbines
        self.nr_birds = nr_birds
        self.bird_mean = bird_mean
        self.bird_angle = bird_angle
        self.x_sigma = x_sigma
        self.rotor_diameter = rotor_diameter
        self.farm_length = farm_length
        self.seed = seed

        sys.modules["__main__"].medClassifier = medClassifier

        # Load ensemble surrogate model
        with open(ensemble_file, "rb") as f:
            self.ensemble = pickle.load(f)
    

    def _validate_x(self, x):
        '''
        check validatity of coordinates of the turbines
        '''
        x = np.asarray(x, dtype=float).reshape(-1)
        expected_dim = 2 * self.n_turbines
        if x.shape[0] != expected_dim:
            raise ValueError(f"x must have length {expected_dim}, got {x.shape[0]}")
        return x
  

    def _to_coords(self, x):
        '''
        turn input x into 2D coordinates
        '''
        x = self._validate_x(x)
        xs = x[:self.n_turbines]
        ys = x[self.n_turbines:]
        coords = np.column_stack((xs, ys))
        return coords
    
    
    def _validate_hub(self, hub):
        '''
        check validatity of coordinates of the hub
        '''
        hub = np.asarray(hub, dtype=float).reshape(-1)
        if hub.shape[0] != 2:
            raise ValueError("hub must be a 2D coordinate like [hub_x, hub_y].")
        return hub
    

    def objective1(self, x) -> float:
        '''
        Objective 1: use ensemble surrogate to predict the power generation
        '''
        x = self._validate_x(x)
        X = np.array([x], dtype=float)
        pred = self.ensemble.predict(X)
        return float(pred[0])

    def objective2(self, x) -> float:
        '''
        Objective 2: calculate the ratio of birds that are too close

        Note: scores are relative values between 0 and 1 only for this bird corridor,
        if the bird corridor changes, 0 and 1 will have different meanings
        '''
        x = np.asarray(x, dtype=float)
        coords = self._to_coords(x)

        bird_std = - self.bird_mean / self.x_sigma

        # calculate how many birds fly in the wind farm area

        # min and max distance that birds can be away from the corridor
        # while still crossing the wind farm area

        # calculate distance between corner point and where bird corridor touches wind farm
        # See https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
        # Line defined by point and angle
        # {\displaystyle \operatorname {distance} (P,\theta ,(x_{0},y_{0}))=|\cos(\theta )(P_{y}-y_{0})-\sin(\theta )(P_{x}-x_{0})|}
        if 0 <= self.bird_angle <= 90:
            Px = 1
            Py = 0 #x and y of point where bird corridor touches wind farm (corner point)
            x0 = 0
            y0 = 1 # furthest point from bird corridor within wind farm (corner point)
        elif 90 < self.bird_angle <= 180:
            Px = 1
            Py = 1
            x0 = 0
            y0 = 0
        elif 180 < self.bird_angle <= 270:
            Px = 0
            Py = 1
            x0 = 1
            y0 = 1
        elif 270 < self.bird_angle <= 360:
            Px = 0
            Py = 0
            x0 = 1
            y0 = 1
        else:
            raise ValueError(f"Angle must be between 0 and 360 degrees, got {self.bird_angle}")
        rad = self.bird_angle * np.pi / 180  # convert angle to radians
        # maximum distance for birds away from bird corridor, can be up to sqrt(2)*farm_length for 45 degrees
        interval_length = self.farm_length*(np.abs(np.cos(rad)*(Py-y0)-np.sin(rad)*(Px-x0)))

        # calculate distance between wind turbines and bird corridor
        BCdist = np.zeros(self.n_turbines) #distance from wind turbines to bird corridor
        for i in range(self.n_turbines):
            BCdist[i] = self.farm_length*(np.abs(np.cos(rad)*(Py-coords[i,1])-np.sin(rad)*(Px-coords[i,0])))


        # a truncated normal distribution to simulate the distribution of birds
        birds = truncnorm.rvs(
            self.x_sigma,                                  # lower bound, e.g. 4 standard deviations
            self.x_sigma + interval_length / bird_std,     # upper bound
            loc=self.bird_mean,                            # mean
            scale=bird_std,                                # std
            size=self.nr_birds,                            # nr. of samples
            random_state=self.seed                         # ramdom seed
        )

        # calculate distance
        d_closest =  np.min(BCdist)# the distance of the turbine closest to the bird corridor
        threshold = d_closest - self.rotor_diameter         # the distance of the threshold
        close_birds = np.sum(birds >= threshold) / self.nr_birds # ratio of close_birds
        return float(close_birds)
    

    def objective3(self, x, hub) -> float:
        '''
        Objective 3: total cable length using MST over [hub + turbines]
        '''
        coords = self._to_coords(x)
        hub = self._validate_hub(hub).reshape(1, 2)

        points = np.vstack([hub, coords]) # Point 0 = hub, points 1...N = turbines
        dist_matrix = cdist(points, points) # from point i to point j, calculating pairwise Euclidean distance between every pair of points
        mst = minimum_spanning_tree(dist_matrix) # generate Minimum spanning tree
        total_length = mst.sum() # Sum of edge lengths in normalized coordinates, then scale

        return float(total_length) * self.farm_length


    def constraint1(self, x) -> float:
        '''
        Constraint 1: distances between turbines should be bigger than 2 * rotor_diameter
        cv is the constraint violation
        '''
        coords = self._to_coords(x)

        min_dist = np.inf
        for turb in range(self.n_turbines - 1):
            dists = cdist([coords[turb]], coords[turb + 1:]) # Distances from the current turbine to all remaining turbines
            next_min = np.min(dists)
            if next_min < min_dist:
                min_dist = next_min

        cv = 2 * self.rotor_diameter - min_dist * self.farm_length
        return float(cv)
    
    
 
    def constraint2(self, x, hub) -> int:
        '''
        Constraint 2: check the feasibility of turbines and the hub 
        n_violate is the constraint violation
        '''
        coords = self._to_coords(x)
        hub = self._validate_hub(hub)

        n_violate = 0

        for xi, yi in coords:
            if self.problem.feasibility_turbine(xi, yi) == 0:
                n_violate += 1

        if self.problem.feasibility_hub(hub[0], hub[1]) == 0:
            n_violate += 1

        return int(n_violate)
    


    def constraint3(self, x, hub) -> int:
        '''
        Constraint 3: turbines and hub should not be too close to external reservoir platforms
        '''
        coords = self._to_coords(x)
        hub = self._validate_hub(hub)

        centres = getattr(self.problem, "external_reservoir_centres", [])
        radius = float(getattr(self.problem, "reservoir_centre_radius", 0.0))

        if radius <= 0.0 or not centres:
            return 0

        centres = np.asarray(centres, dtype=float)

        n_violate = 0

        turbine_dists = cdist(coords, centres)
        n_violate += int(np.sum(np.any(turbine_dists < radius, axis=1)))

        hub_dists = cdist(hub.reshape(1, 2), centres)
        if np.any(hub_dists < radius):
            n_violate += 1

        return int(n_violate)

  
    def evaluate(self, x, hub):
        '''
        Evaluate all objectives and constraints
        '''
        return {
            "f1": self.objective1(x),
            "f2": self.objective2(x),
            "f3": self.objective3(x, hub),
            "g1": self.constraint1(x),
            "g2": self.constraint2(x, hub),
            "g3": self.constraint3(x, hub),
        }