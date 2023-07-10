#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 12:22:20 2023

@author: burges26

Queueing model of flow of people through homeless services
"""

def arr_rate(t):
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
    
    arr_rate = 1
    
    return arr_rate

def serve_rate(t):
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
    
    serve_rate = 1
    
    return serve_rate

def num_serve(t):
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
    
    num_serve = 1
    
    return num_serve

class queue(object):
    """
    A queue to be modelled with numerical integration
    """
    
    def __init__(self):
        """
        Initialise instance of queue with empty probability distribution date (self.p)


        Returns
        -------
        None.

        """
        self.p = None

    def model_dynamics(self, n_0, N, Y, d):
        """
        Model the dynamics of the queue
        
        Parameters
        ----------
        n_0 : int
            number of customers in the system at time t=0.
        N : int
            max number of customers in system. 
        Y : float
            time horizon for analysis in integer years
        d : float
            width of time step in days

        Returns
        -------
        None.

        """
        
        # set up 
        d = d/365 # timestep size in years
        T = int(Y/d) # number of time steps
        
        # init state probabilities
        self.p = [[0 for i in range(T)] for j in range(N+1)]
        self.p[n_0][0] = 1 # enforce state at time 0
        
        # init m, number of busy servers
        m = [0 for i in range(N+1)]
        
        # numerical integration
        for t in range(1,T):
            
            # arrival/service rates and number servers at prev timestep
            lmbda = arr_rate((t-1)*d) 
            mu = serve_rate((t-1)*d) 
            s = num_serve((t-1)*d)
            
            # number of busy servers
            m[1] = min(1,s)
            m[N] = min(N,s)
            
            # prob of being in state 0 or (N-1)
            self.p[0][t] = (self.p[0][t-1] * (1-lmbda*d)) + (self.p[1][t-1] * (mu*m[1]*d))
            self.p[N][t] = (self.p[N][t-1] * (1-mu*m[N]*d)) + (self.p[N-1][t-1] * (lmbda*d))
            
            for n in range(1,N):
                m[n+1] = min(n,s)           
                self.p[n][t] = (self.p[n-1][t-1] * lmbda*d) + (self.p[n][t-1] * (1-lmbda*d-mu*m[n]*d)) + (self.p[n+1][t-1] * (mu*m[n+1]*d))
 
q = queue(5)
q.model_dynamics(5,10,1,1)























