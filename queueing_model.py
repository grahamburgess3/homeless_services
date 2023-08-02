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

class queue(object):
    """
    A queue to be modelled with numerical integration
    """
    
    def __init__(self, annual_arrival_rate, mean_service_time, servers_initial, shelter_initial, server_build_rate, shelter_build_rate, num_in_system_initial, max_in_system):
        """
        Initialise instance of queue


        Returns
        -------
        None.

        """
        self.p = None
        self.num_sys = None
        self.num_queue = None
        self.num_unsheltered = None
        self.annual_arrival_rate = annual_arrival_rate
        self.mean_service_time = mean_service_time
        self.servers_initial = servers_initial
        self.shelter_initial= shelter_initial
        self.server_build_rate = np.repeat(server_build_rate, 6)
        self.shelter_build_rate = np.repeat(shelter_build_rate, 6)
        self.num_in_system_initial = num_in_system_initial
        self.max_in_system = max_in_system

    def arr_rate(self, t):
        """
        returns arrival rate at time t

        Parameters
        ----------
        t : float
            time.

        Returns
        -------
        arr_rate : arrival rate at time t.

        """
        
        arr_rate = self.annual_arrival_rate[math.floor(t)]
        
        return arr_rate

    def serve_rate(self, t):
        """
        returns servie rate at time t

        Parameters
        ----------
        t : float
            time.

        Returns
        -------
        serve_rate : service rate at time t.

        """
        
        serve_rate = 1/self.mean_service_time
        
        return serve_rate

    def num_serve(self, t):
        """
        returns number of servers at time t

        Parameters
        ----------
        t : float
            time.

        Returns
        -------
        num_serve: num servers at time t.

        """
        
        num_serve = self.servers_initial
        
        num_two_months = math.ceil(t/(1/6))
        
        for i in range(num_two_months):
            num_serve += self.server_build_rate[i]
        
        return num_serve

    def num_shelt(self, t):
        """
        returns number of shelters at time t

        Parameters
        ----------
        t : float
            time.

        Returns
        -------
        num_shelt: num shelters at time t.

        """
        
        num_shelt = self.shelter_initial
        
        num_two_months = math.ceil(t/(1/6))
        
        for i in range(num_two_months):
            num_shelt += self.shelter_build_rate[i]
        
        return num_shelt

    def model_dynamics(self, Y, d):
        """
        Model the dynamics of the queue
        
        Parameters
        ----------
        Y : float
            time horizon for analysis in integer years
        d : float
            width of time step in days

        Returnsserver_build_rate
        -------
        None.

        """
        
        # set up 
        d = d/365 # timestep size in years
        T = int(Y/d) # number of time steps
        N = self.max_in_system
        n_0 = self.num_in_system_initial        
        
        # init state probabilities
        self.p = [[0 for i in range(T)] for j in range(N+1)]
        self.p[n_0][0] = 1 # enforce state at time 0
        
        # init m, number of busy servers in each state
        m = [0 for i in range(N+1)]
        
        # init outputs
        self.num_sys = [0 for i in range(T)]
        self.num_queue = [0 for i in range(T)]        
        self.num_unsheltered = [0 for i in range(T)]
        
        self.num_sys[0] = n_0
        self.num_queue[0] = max(0, n_0 - self.num_serve(0))
        self.num_unsheltered[0] = max(0, n_0 - self.num_serve(0) - self.num_shelt(0))
        
        # numerical integration - loop through t
        for t in range(1,T):
            
            # arrival/service rates and number servers at prev timestep
            lmbda = self.arr_rate((t-1)*d) 
            mu = self.serve_rate((t-1)*d) 
            s = self.num_serve((t-1)*d)
            
            # number of busy servers
            m[1] = min(1,s)
            m[N] = min(N,s)
            
            # prob of being in state 0 or (N-1)
            self.p[0][t] = (self.p[0][t-1] * (1-lmbda*d)) + (self.p[1][t-1] * (mu*m[1]*d))
            self.p[N][t] = (self.p[N][t-1] * (1-mu*m[N]*d)) + (self.p[N-1][t-1] * (lmbda*d))
            
            # loop through each state n
            for n in range(1,N):
                # number of shelters busy in next state     
                m[n+1] = min(n+1,s) 
                
                # prob of being in other states
                self.p[n][t] = (self.p[n-1][t-1] * lmbda*d) + (self.p[n][t-1] * (1-lmbda*d-mu*m[n]*d)) + (self.p[n+1][t-1] * (mu*m[n+1]*d))
            
            # number of servers and shelters at current timestep
            s = self.num_serve(t*d)
            shelt = self.num_shelt(t*d)
            
            # expected values for outputs
            for n in range(N+1):                
                self.num_sys[t] += n * self.p[n][t]
                self.num_queue[t] += max(0,n-s) * self.p[n][t]
                self.num_unsheltered[t] += max(0,n-s-shelt) * self.p[n][t]
 
def mms_steadystate(lmbda, s, mu):
    """
    Compute steady state expected number in system and queue for an M/M/s queue

    Parameters
    ----------
    lmbda : float
        arrival rate.
    s : int
        number of servers.
    mu : float
        service rate.

    Returns
    -------
    num_sys : float
        expected number in system.
    num_q : float
        expected number in queue.

    """
    
    # traffic intensity
    rho = lmbda/(s*mu)
    
    # sum of terms
    summation = 0
    for n in range(s):
        summation += ((s*rho)**n)/math.factorial(n)
        
    summation += ((s*rho)**s)/(math.factorial(s)*(1-rho))
    
    # pi_0
    pi_0 = 1/summation
    
    # num_q
    num_q = rho*((s*rho)**s)*pi_0/(math.factorial(s)*((1-rho)**2))
    
    # num_sys
    num_sys = num_q + lmbda/mu
    
    return num_sys, num_q

def create_fanchart(arr, line, q):
    """
    create a fan chart using an array of arrays and a line

    Parameters
    ----------
    arr : np.array(np.array)
        simulation data over time for multiple simulation runs
    line : list
        bi-monthly data from the analytical queueing model
    q : queue
        the analytical queueing model

    Returns
    -------
    fix, ax : graph object

    """
    x = (np.arange(arr.shape[0]))/6
    percentiles = (60, 70, 80, 90)
    fig, ax = plt.subplots()
    for p in percentiles:
        low = np.percentile(arr, 100-p, axis=1)
        high = np.percentile(arr, p, axis=1)
        alpha = (100 - p) / 100
        ax.fill_between(x, low, high, color='green', alpha=alpha)
    thin, = ax.plot(np.arange(0,6,1/365), q.num_unsheltered, color = 'black', linestyle = 'dashed', linewidth = 0.5)
    thick, = ax.plot(x, line, color = 'black', linewidth = 0.8)
    plt.xlabel('Years')
    plt.ylabel('# Unsheltered')
    plt.title('Number of unsheltered people - simulation and queueing model results')
    
    first_legend = plt.legend([f'Simulation: {100-p}th - {p}th percentile' for p in percentiles])
    ax.add_artist(first_legend)
    ax.legend(handles=[thick,thin], labels=['Queueing model: bi-monthly','Queueing model: daily'], loc='upper right', bbox_to_anchor=(1,0.74))
    
    return fig, ax