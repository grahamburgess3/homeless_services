import numpy as np
import pandas as pd

class SolutionSpace():
    
    def __init__(self, solutions):
        self.solutions = [Solution(solutions[i]) for i in range(len(solutions))]
        self.Y0 = [[] for i in solutions]
        self.S2 = None
        self.active = np.array([True for i in solutions])
        self.eliminate = [0 for i in solutions]

    def optimise_rs(self, alpha, n0, delta, sim):
        # get first n0 solutions
        for x in range(len(self.solutions)):
            for rep in range(n0):
                cost = sim(self.solutions[x].solution)
                self.Y0[x].append(cost)

        # get initial variance
        self.Y0 = np.array(self.Y0)
        self.S2 = np.cov(self.Y0)

        # setup for iteration
        eta = 0.5*( (2*alpha/(len(self.solutions)-1))**(-2/(n0-1)) -1)
        h2 = 2*eta*(n0-1)
        Y_sum = np.sum(self.Y0, axis=1)
        r = n0
        k_iter = np.array([i for i in range(len(self.solutions))])

        # elimination loop
        while sum(self.active) > 1:
            r += 1
            a_temp = self.active.copy()

            # collect new data
            for i in k_iter[self.active]:
                cost = sim(self.solutions[x].solution)
                Y_sum[i] += cost

            # elimination
            for i in k_iter[self.active]:
                for j in k_iter[self.active]:
                    S2diff = self.S2[i,i] + self.S2[j,j] - 2 * self.S2[i,j]
                    W = max(0, (delta/2)*(h2*S2diff/delta**2 - r))
                    if Y_sum[i] > Y_sum[j] + W:
                        a_temp[i] = False
                        self.eliminate[i] = r
                        break
                        
            self.active = a_temp.copy()
        
class Solution():

    def __init__(self, solution):
        self.solution = solution
