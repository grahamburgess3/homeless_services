#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 12:22:20 2023

@author: burges26

Queueing model of flow of people through homeless services
"""

# imports
import numpy as np
import math
import matplotlib.pyplot as plt
import numpy as np
import pdb
import helper as hlp

class Queue(object):
    """
    A queue to be modelled with numerical integration

    """
    
    def __init__(self, data, solution):
        """
        Initialise instance of queue

        Parameters
        ----------
        data : dict
        solution : dict(list) : includes annual build rates for 'housing' and 'shelter'

        Returns
        -------
        None.

        """
        self.p = None # prob of being in each state (num in system) at each time t
        self.p_q = None # prob of being in each state (num in queue) at each time t
        self.p_unsh = None # prob of being in each state (num unsheltered) at each time t
        self.p_sh = None # prob of being in each state (num sheltered) at each time t
        self.num_sys = None # expected val at each time t
        self.num_queue = None # expected val at each time t
        self.num_unsheltered = None # expected val at each time t
        self.num_sheltered = None # expected val at each time t
        self.num_unsheltered_avg = None
        self.num_sheltered_avg = None
        self.annual_arrival_rate = data['arrival_rates']
        self.mean_service_time = data['service_mean']['housing']
        self.servers_initial = data['initial_capacity']['housing']
        self.shelter_initial= data['initial_capacity']['shelter']
        self.server_build_rate = solution['housing']
        self.shelter_build_rate = solution['shelter']
        self.num_in_system_initial = data['initial_demand']
        self.max_in_system = data['max_in_system']
        self.d = data['delta_t']
        self.T = data['T_a'] + data['T_b']
        self.h = hlp.get_daily_capacity(data['T_b'], data['initial_capacity']['housing'], solution['housing'])
        self.s = hlp.get_daily_capacity(data['T_b'], data['initial_capacity']['shelter'], solution['shelter'])

    def arr_rate(self, t):
        """
        returns daily arrival rate at time t

        Parameters
        ----------
        t : float : time in years.

        Returns
        -------
        arr_rate : daily arrival rate at time t.

        """
        
        arr_rate = self.annual_arrival_rate[math.floor(t)]
        
        return arr_rate

    def serve_rate(self, t):
        """
        returns servie rate at time t - this is in fact constant.

        Parameters
        ----------
        t : float : time in years.

        Returns
        -------
        serve_rate : service rate at time t.

        """
        
        serve_rate = 1/self.mean_service_time
        
        return serve_rate

    def analyse(self):
        """
        Model the dynamics of the queue
        Assume 365 days in all years
        data['arrival_rate'] should include the same number of entries as the value of Y when calling model_dynamics(Y,d)
        
        Parameters
        ----------
        None

        Return
        -------
        None

        """
        # set up 
        d = self.d/365 # timestep size in years
        T = int(self.T/d) # number of time steps
        N = self.max_in_system
        n_0 = self.num_in_system_initial      
        
        # init state probabilities
        self.p = [[0 for i in range(T)] for j in range(N+1)]
        self.p[n_0][0] = 1 # enforce state at time 0
        self.p_q = [[0 for i in range(T)] for j in range(N+1)]
        self.p_q[max(0, n_0 - self.servers_initial)][0] = 1 # enforce state at time 0
        self.p_unsh = [[0 for i in range(T)] for j in range(N+1)]
        self.p_unsh[max(0, n_0 - self.servers_initial - self.shelter_initial)][0] = 1 # enforce state at time 0
        self.p_sh = [[0 for i in range(T)] for j in range(N+1)]
        self.p_sh[min(max(0, n_0 - self.servers_initial), self.shelter_initial)][0] = 1 # enforce state at time 0
        
        # init outputs
        self.num_sys = [0 for i in range(T)]
        self.num_queue = [0 for i in range(T)]        
        self.num_unsheltered = [0 for i in range(T)]
        self.num_sheltered = [0 for i in range(T)]
        self.h_t = [0 for i in range(T)]
        self.sh_t = [0 for i in range(T)]
        self.num_unsheltered_avg = 0 # average over time
        self.num_sheltered_avg = 0 # avg over time
        self.num_sys[0] = n_0
        self.num_queue[0] = max(0, n_0 - self.servers_initial)
        self.num_unsheltered[0] = max(0, n_0 - self.servers_initial - self.shelter_initial)
        self.num_sheltered[0] = self.num_queue[0] - self.num_unsheltered[0]
        self.h_t[0] = self.servers_initial
        self.sh_t[0] = self.shelter_initial

        # init number busy servers
        m = [0 for i in range(N+1)]
        
        # numerical integration - loop through t
        for t in range(1,T):
            
            # arrival/service rates and number servers after timestep
            lmbda = self.arr_rate((t-1)*d) 
            mu = self.serve_rate((t-1)*d)
            s = math.floor(self.h[t-1])

            # number of busy servers
            m[1] = min(1,s)
            m[N] = min(N,s)
            
            # prob of being in state 0 or (N-1)
            self.p[0][t] = (self.p[0][t-1] * (1-lmbda*d)) + (self.p[1][t-1] * (mu*m[1]*d))
            self.p[N][t] = (self.p[N][t-1] * (1-mu*m[N]*d)) + (self.p[N-1][t-1] * (lmbda*d))
            
            # loop through each state n
            for n in range(1,N):
                # number of servers busy in next state     
                m[n+1] = min(n+1,s) 
                # prob of being in other states
                self.p[n][t] = (self.p[n-1][t-1] * lmbda*d) + (self.p[n][t-1] * (1-lmbda*d-mu*m[n]*d)) + (self.p[n+1][t-1] * (mu*m[n+1]*d))
            
            # number of servers and shelters at current timestep
            s = math.floor(self.h[t])
            shelt = math.floor(self.s[t])
            
            for n in range(N+1):                
                # expected values for outputs
                self.num_sys[t] += n * self.p[n][t]
                extra_queue = max(0,n-s) * self.p[n][t]
                self.num_queue[t] += extra_queue
                extra_unsheltered = max(0,n-s-shelt) * self.p[n][t]
                self.num_unsheltered[t] += extra_unsheltered
                extra_sheltered = extra_queue - extra_unsheltered
                self.num_sheltered[t] += extra_sheltered
                
                # probs of number in q and unsheltered
                self.p_q[max(0,n-s)][t] += self.p[n][t]
                self.p_unsh[max(0,n-s-shelt)][t] += self.p[n][t]
                self.p_sh[min(max(0,n-s),shelt)][t] += self.p[n][t]
                
        # average over time of the expected value of number unshelterd - add up to point t
        self.num_unsheltered_avg = sum(self.num_unsheltered)/len(self.num_unsheltered)
        self.num_sheltered_avg = sum(self.num_sheltered)/len(self.num_sheltered)
                
    def plot(self, percentiles = {'low' : 0.1, 'high' : 0.9}):
        """
        create a fan chart for Q outputs
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
            
        """

        # get data
        upc = self.get_percentiles(percentiles, self.p_unsh)
        x = [i/365 for i in range(self.T*365)]

        # setup
        fig, ax = plt.subplots()

        # plot
        ax.plot(x, self.h, color = 'green')
        ax.plot(x, self.s, color = 'orange')
        ax.plot(x,self.num_unsheltered, color = 'red')
        ax.fill_between(x, upc['low'], upc['high'], facecolor = 'red', alpha = 0.3)
        
        # formatting
        ax.set(xlabel='t (yrs)', ylabel='Number of people',
               title='$M_t/M/h_t$ queueing model')
        ax.legend(["$h_t$", "$s_t$", "$u_t$"], loc="upper left")
        ax.grid()
        ymax = max(self.h + self.s + self.num_unsheltered)
        ax.set_ylim(0,ymax*1.05)

        # output
        plt.show()

    def get_percentiles(self, percentiles, data):
        """
        Parameters
        ----------
        T : int - time duration to analyse in days
        percentiles : dict(int) - two percentiles (one high and one low) must be of form {'low': int, 'high' : int}
        data : list - data on probabilities
    
        Returns
        -------
        out : dict(list) - percentiles of data in question
        """
        T = self.T*365
        percentiles_high = [t for t in range(T)]
        percentiles_low = [t for t in range(T)]
        
        for t in range(T):
            looking_high = True
            looking_low = True
            n = 0
            prob_cumulative = 0
            while looking_high:
                prob_cumulative += data[n][t]
                if prob_cumulative > percentiles['low']:
                    if looking_low == True:
                        percentiles_low[t] = n
                        looking_low = False
                    if prob_cumulative > percentiles['high']:
                        percentiles_high[t] = n
                        looking_high = False
                n += 1
                
        out = {'low' : percentiles_low, 'high' : percentiles_high}
        return out
