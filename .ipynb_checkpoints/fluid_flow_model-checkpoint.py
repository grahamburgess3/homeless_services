import math
import numpy as np

class FluidFlowModel():

    def __init__(self, data, solution, T_a, T_b):
        """
        Initialise instance of fluid flow model - continuous approximation of M(t)/M/s(t) model
        
        Parameters
        ----------
        data : dict : includes 'initial_demand', 'initial_capacity', 'service_mean', 'arrival_rates'
        solution : dict(list) : includes annual build rates for 'housing' and 'shelter'

        Returns
        -------
        None

        """
        self.X_0 = data['initial_demand']
        self.h_0 = data['initial_capacity']['housing']
        self.s_0 = data['initial_capacity']['shelter']        
        self.h = self.get_daily_capacity(self.h_0, solution['housing'], T_a, T_b)
        self.s = self.get_daily_capacity(self.s_0, solution['shelter'], T_a, T_b)
        self.s_sq = [x**2 for x in self.s] # E[(num sheltered)^2] over time
        self.u = [self.X_0 - self.h_0 - self.s_0] # number unsheltered over time (Expected val)
        self.u_sq = [self.u[0]**2] # E[(num unsheltered)^2] over time
        self.mu0 = 1/(data['service_mean']['housing']*365)
        self.lambda_t = list(np.repeat(data['arrival_rates'],365))

    def get_daily_capacity(self, init, solution, T_a, T_b):
        annual = [init] + list(solution)
        diffs = list(np.repeat([(annual[i+1] - annual[i])/365 for i in range(int(T_a/365))],365))
        daily = [init + sum(diffs[0:i]) for i in range(1,len(diffs)+1)] + [list(solution)[-1]]*T_b
        return daily
                           
    def evaluate_queue_size(self, t):
        """
        Evaluate expected number unsheltered at time t
        
        Parameters
        ----------
        t : float : time (in years) evaluate queue size

        Returns
        -------
        unsh : float : E[number unsheltered at time t]

        """

        # compute u_t
        u_t = self.X_0 + sum(self.lambda_t[0:t]) - sum(self.h[0:t])*self.mu0 - self.h[t] - self.s[t]
        
        # return
        return u_t

    def analyse(self, T):
        """
        Evaluate Q performance measures for all times in T
        
        Parameters
        ----------
        T : list[float] : times (in units of days) to evaluate queue size

        Returns
        -------
        None

        """

        for t in T[1:len(T)]:
            unsh = self.evaluate_queue_size(t)
            self.u.append(unsh)
            self.u_sq.append(unsh**2)