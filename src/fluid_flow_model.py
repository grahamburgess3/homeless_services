#### NOTE: This is the cleanest way of computing the size of the unsheltered Q at some time t using the fluid model. This is, however, different from that used in the PSOR paper since in that version it had to be compatible with the IPOPT solver in Pyomo, and the below is not (due to making a list with the Pyomo Indexed Var, and due to using functions within numpy)

import math
import numpy as np
import matplotlib.pyplot as plt
import helper as hlp

import pdb

class FluidFlowModel():

    def __init__(self, data, solution):
        """
        Initialise instance of fluid flow model
        
        Parameters
        ----------
        data : dict
        solution : dict(list) : includes annual build rates for 'housing' and 'shelter'

        Returns
        -------
        None

        """
        self.T_a = data['T_a'] # decision horizon (years)
        self.T_b = data['T_b'] # extra modeling horizon (years)
        self.X_0 = data['initial_demand'] 
        self.h_0 = data['initial_capacity']['housing']
        self.s_0 = data['initial_capacity']['shelter']        
        self.h = hlp.get_daily_capacity(self.T_b, self.h_0, solution['housing'])
        self.s = hlp.get_daily_capacity(self.T_b, self.s_0, solution['shelter'])
        self.s_sq = [x**2 for x in self.s] # E[(num sheltered)^2] over time
        self.u = [self.X_0 - self.h_0 - self.s_0] # number unsheltered over time (Expected val)
        self.u_sq = [self.u[0]**2] # E[(num unsheltered)^2] over time
        self.mu0 = 1/(data['service_mean']['housing']*365) # daily service rate for one housing server
        self.lambda_t = list(np.repeat([x/365 for x in data['arrival_rates']],365))
                           
    def evaluate_queue_size(self, t):
        """
        Evaluate expected number unsheltered at time t
        
        Parameters
        ----------
        t : float : time (in days) evaluate queue size

        Returns
        -------
        unsh : float : E[number unsheltered at time t]

        """
        u_t = self.X_0 + sum(self.lambda_t[0:t]) - sum(self.h[0:t])*self.mu0 - self.h[t] - self.s[t]
        return u_t

    def analyse(self):
        """
        Evaluate Q performance measures (self.u and self.u_sq) for all times in modelling horizon
        
        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        T = [i for i in range((self.T_a+self.T_b)*365)]
        for t in T[1:len(T)]:
            unsh = self.evaluate_queue_size(t)
            self.u.append(unsh)
            self.u_sq.append(unsh**2)

    def plot(self):
        """
        Display Plot of solution performance

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        # general plotting setup
        fig, ax = plt.subplots()
        ymax = max(self.h + self.s + self.u)
        
        # plot solution
        x = [i/365 for i in range((self.T_a+self.T_b)*365)]
        ax.plot(x, self.h, color = 'green')
        ax.plot(x, self.s, color = 'orange')
        ax.plot(x, self.u, color = 'red')

        # formatting
        ax.set(xlabel='t (yrs)', ylabel='Number of people',
               title='Fluid flow model')
        ax.legend(["$h_t$", "$s_t$", "$u_t$"], loc="upper left")
        ax.grid()
        ax.set_ylim(0,ymax*1.05)
        
        # display
        plt.show()
