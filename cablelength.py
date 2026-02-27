from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_array

from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt

#x = [0.16583492632257624, 0.4871309875290023, 0.24153605985017246, 0.2954912222384124, 0.9558389075666868, 0.7992480932223422, 0.5400992985215289, 0.14902261675540462, 0.7592757901802544, 0.9983162571623986]
# x = [0.16, 0.48, 0.24, 0.29, 0.95, 0.79, 0.54, 0.14, 0.75, 0.99]
# x = [0.44, 0.02, 0.25, 0.63, 1.0, 0.0, 1.0, 0.46, 0.56, 0.0]
m = 5 #nr. of wind turbines
x = np.random.rand(1,2*m)

class objective_cablelength:
    def __init__(self, hub):
        self.m = 5 # number of wind turbines
        self.hub = hub # hub location for cables to go to. Need to give the hub when defining this class
        self.MST = None
        self.cable_length = []
        self.coords = []

    def evaluate(self,x):
        x = np.concatenate((x, self.hub), axis=1) # From now on, points include the hub location
        self.coords = np.resize(x, (m + 1, 2))  # Transform to 2D space
        dists = cdist(self.coords, self.coords, 'euclidean') # Calculate Euclidean distance between all points
        self.MST = minimum_spanning_tree(dists) # Calculate minimum spanning tree
        self.cable_length = np.sum(self.MST) # Calculate total cable length for minimum spanning tree
        return self.cable_length, self.MST

    def visualize(self):
        if self.MST==None:
            print('Warning: Use the evaluate function first')
        mst_coo = self.MST.tocoo() # Get MST coordinates, make sure to use the evaluate function first

        # Plot MST and points
        for i, j, weight in zip(mst_coo.row, mst_coo.col, mst_coo.data):
            # Plot line between coords[i] and coords[j]
            plt.plot([self.coords[i, 0], self.coords[j, 0]], [self.coords[i, 1], self.coords[j, 1]], 'r-', lw=0.3)
        for i in range(m):
            plt.scatter(self.coords[i, 0] - 0.02, self.coords[i, 1] + 0.025, marker="$" + str(i) + "$") # plot point labels
        plt.scatter(self.coords[0:m, 0], self.coords[0:m, 1])  # ,c=[0,1,2,3,4])

        # Plot hub
        plt.scatter(self.coords[m, 0], self.coords[m, 1], marker='s', s=50, color='red')

        plt.vlines(x=0, ymin=0, ymax=1)
        plt.hlines(y=0, xmin=0, xmax=1)
        plt.vlines(x=1, ymin=0, ymax=1)
        plt.hlines(y=1, xmin=0, xmax=1)
        plt.show()





if __name__ == "__main__":
    m = 5  # nr. of wind turbines
    x = np.random.rand(1, 2 * m)
    hub = 0.5 * np.random.randn(1, 2) + 0.5  # hub for the cables, can be outside search space. Generate one at a random location once
    cab = objective_cablelength(hub)
    CL, MST = cab.evaluate(x) # cable length in normalized space, multiply with 5*333.33 for cable length in meters
    CL = CL * 5*333.33
    print(CL)
    cab.visualize()
