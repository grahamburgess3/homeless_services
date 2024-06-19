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
import pyomo.environ as pyo

class queue(object):
    """
    A queue to be modelled with numerical integration

    ##### Changes to the M(t)/M/H(t) queueing model

    Here $H(t)$ refers to the number of houses over time. We have changed this $H(t)$ function in order to make it more appropriate for this analysis. 
    
    Previously, $H(t)$ reflected the fact that every 2 months, an integer number of houses were built, and that bi-monthly build rate was constant within each year, but could change from year to year. Therefore, an annual build rate had to be a multiple of six and if it wasn't, the build rate was rounded down to the nearest six. The same was the case for the build rate for shelters. This is not suitable for this deterministic optimisation since for continuous-valued build functions, subsets of solutions will have the same objective value, which is not suitable for a solver which uses some gradient information. 
    
    We now construct $H(t)$ so that at at any time $t$, based on continuous-valued annual build rates, we calculate the number of whole houses ready for use at time $t$, and that dictates the total service rate until time $t + \Delta t$. For example, if we start with 20 houses and the annual house build rate is 40/year, then at time t = 60 days, we have $h(t=60) = \lfloor 20 + \frac{40}{365} * 60 \rfloor = 26$. We do the same for the number of shelters $s(t)$ at any time $t$. 
    
    This brings the queueing model further away from the simulation model, and more in line with the fluid model. However, it is not quite the same as the fluid model, since in the fluid model, a fraction of a house can contribute to an increased service rate. 
    
    The response $y(\boldsymbol{h},\boldsymbol{s})$ where $\boldsymbol{h},\boldsymbol{s}$ are the house and shelter annual build rates, respectively, using this amended queueing model will still be a step function, since an infinitessimally small change to the build rate functions will leave the number of houses and shelters, $H(t)$ and $S(t) \hspace{0.2cm} \forall t \in \{1, 1 + \Delta t, ... , T\}$ unchanged. It therefore remains to be seen whether this setup is suitable for the deterministic optimisation formulation. 
    
    **NOTE**: it  does not make sense to allow $H(t)$ and $S(t)$ to take non-integer values for time $t \in \{1, 1 + \Delta t, ... , T\}$, because our state space $n \in \{1, 2, ..., N\}$ where $n$ is the number of people in the system, takes integer values. 
    
    """
    
    def __init__(self, annual_arrival_rate, mean_service_time, initial_capacity, build_rates, num_in_system_initial, max_in_system):
        """
        Initialise instance of queue
        Note: there are no hardcoded elements in this class (apart from obvious things like the number of days in a week) but care should be taken if adapting some of the inputs, as detailed below: 
        - annual_arrival_rate should include the same number of entries as the value of Y when calling model_dynamics(Y,d)
        - server_build_rate and shelter_build_rate should have the same number of entries as num_annual_buildpoints, and this number should tally (roughly) with build_frequency_weeks. For example, if build_freq_weeks is 9, then it will take 54 weeks for this to happen 6 times, so things will start being out of sync after 5 years of model time, which is OK as long as you don't model so far ahead (or if the queue is always empty before 5 years, say.)
        - Care should also be taken if changing the step size d from 1 day, which is what it was intially set up for. 

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
        self.h_t = None # number of houses at time t
        self.sh_t = None # number of shelters at time t
        self.num_unsheltered_avg = None
        self.num_sheltered_avg = None
        self.annual_arrival_rate = annual_arrival_rate
        self.mean_service_time = mean_service_time['housing']
        self.servers_initial = initial_capacity['housing']
        self.shelter_initial= initial_capacity['shelter']
        # self.num_build_points_btwn_changes = round(time_btwn_changes_in_build_rate/time_btwn_building)
        # self.server_build_rate = np.repeat([int(i / self.num_build_points_btwn_changes) for i in build_rates['housing']], self.num_build_points_btwn_changes)
        # self.shelter_build_rate = np.repeat([int(i / self.num_build_points_btwn_changes) for i in build_rates['shelter']], self.num_build_points_btwn_changes)
        self.server_build_rate = build_rates['housing']
        self.shelter_build_rate = build_rates['shelter']
        # self.build_frequency_weeks = round(time_btwn_building*365/7)
        self.num_in_system_initial = num_in_system_initial
        self.max_in_system = max_in_system
        #self.m = m # number of busy servers in state n at time t

    def arr_rate(self, t):
        """
        returns arrival rate at time t

        Parameters
        ----------
        t : float
            time in years.

        Returns
        -------
        arr_rate : arrival rate at time t.

        """
        
        arr_rate = self.annual_arrival_rate[math.floor(t)]
        
        return arr_rate

    def serve_rate(self, t):
        """
        returns servie rate at time t - this is in fact constant.

        Parameters
        ----------
        t : float
            time in years.

        Returns
        -------
        serve_rate : service rate at time t.

        """
        
        serve_rate = 1/self.mean_service_time
        
        return serve_rate

    def num_serve(self, t):
        """
        returns weighted avg number of servers between time t and time t + 1 days

        Parameters
        ----------
        t : float
            time in years.

        Returns
        -------
        num_serve: num servers at time t.

        """

        num_serve = {'before' : 0, 'after' : 0}
        n = self.servers_initial
        
        # add complete years
        yrs = math.floor(t) # number of years passed
        for yr in range(yrs):
            n += self.server_build_rate[yr]
            
        # add fractional year
        n += (t % 1) * self.server_build_rate[yrs]

        num_serve['before'] = pyo.floor(n)
        num_serve['after'] = pyo.floor(n)
            
        return num_serve

    def num_serve_wtd_avg(self, t):
        """
        returns weighted avg number of servers between time t and time t + 1 days

        Parameters
        ----------
        t : float
            time in years.

        Returns
        -------
        num_serve: num servers at time t.

        """

        num_serve = {'before' : 0, 'after' : 0}
        n = self.servers_initial
        
        # add complete years
        yrs = math.floor(t) # number of years passed
        for yr in range(yrs):
            n += self.server_build_rate[yr]
            
        # add fractional year
        n += (t % 1) * self.server_build_rate[yrs]

        # how big is the half-built house?
        half_built_house = n - pyo.floor(n)
        
        # round to integer
        n = pyo.floor(n)
        num_serve['before'] = n
        
        # construct weighted avg
        weight_avg = 0
        T = 0
        while T < 1:
            time_til_next_build = (1-half_built_house)/(self.server_build_rate[yrs]/365)
            weight_avg += n * min(1-T,time_til_next_build)
            T += time_til_next_build
            n += 1
            half_built_house = 0
        
        num_serve['after'] = weight_avg
            
        return num_serve

    def num_shelt(self, t):
        """
        returns integer number of shelters at time t

        Parameters
        ----------
        t : float
            time in years.

        Returns
        -------
        num_shelt: num shelters at time t.

        """
        
        n = self.shelter_initial
        
        # add complete years
        yrs = math.floor(t) # number of years passed
        for yr in range(yrs):
            n += self.shelter_build_rate[yr]
            
        # add fractional year
        n += (t % 1) * self.shelter_build_rate[yrs]
        
        # round to integer
        n = pyo.floor(n)
        num_shelt = n
                    
        return num_shelt

    def model_dynamics(self, Y, d):
        """
        Model the dynamics of the queue
        Assume 365 days in all years
        
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
        self.p_q = [[0 for i in range(T)] for j in range(N+1)]
        self.p_q[max(0, n_0 - self.servers_initial)][0] = 1 # enforce state at time 0
        self.p_unsh = [[0 for i in range(T)] for j in range(N+1)]
        self.p_unsh[max(0, n_0 - self.servers_initial - self.shelter_initial)][0] = 1 # enforce state at time 0
        self.p_sh = [[0 for i in range(T)] for j in range(N+1)]
        self.p_sh[min(max(0, n_0 - self.servers_initial), self.shelter_initial)][0] = 1
        
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
        
        # numerical integration - loop through t
        for t in range(1,T):
            
            # arrival/service rates and number servers after timestep
            lmbda = self.arr_rate((t-1)*d) 
            mu = self.serve_rate((t-1)*d) 
            s = self.num_serve((t-1)*d)['after']

            m = [s for i in range(N+1)]
            
            # prob of being in state 0 or (N-1)
            self.p[0][t] = (self.p[0][t-1] * (1-lmbda*d)) + (self.p[1][t-1] * (mu*m[1]*d))
            self.p[N][t] = (self.p[N][t-1] * (1-mu*m[N]*d)) + (self.p[N-1][t-1] * (lmbda*d))
            
            # loop through each state n
            for n in range(1,N):           
                # prob of being in other states
                self.p[n][t] = (self.p[n-1][t-1] * lmbda*d) + (self.p[n][t-1] * (1-lmbda*d-mu*m[n]*d)) + (self.p[n+1][t-1] * (mu*m[n+1]*d))
            
            # number of servers and shelters at current timestep
            s = self.num_serve(t*d)['before']
            shelt = self.num_shelt(t*d)
            self.h_t[t] = s
            self.sh_t[t] = shelt
            
            for n in range(N+1):                
                # expected values for outputs
                self.num_sys[t] += n * self.p[n][t]
                #extra_queue = max(0,n-s) * self.p[n][t]
                #self.num_queue[t] += extra_queue
                #extra_unsheltered = max(0,n-s-shelt) * self.p[n][t]
                #self.num_unsheltered[t] += extra_unsheltered
                #extra_sheltered = extra_queue - extra_unsheltered
                #self.num_sheltered[t] += extra_sheltered
                
                # probs of number in q and unsheltered
                #self.p_q[max(0,n-s)][t] += self.p[n][t]
                #self.p_unsh[max(0,n-s-shelt)][t] += self.p[n][t]
                #self.p_sh[min(max(0,n-s),shelt)][t] += self.p[n][t]
                
        # average over time of the expected value of number unshelterd - add up to point t
        #self.num_unsheltered_avg = sum(self.num_unsheltered)/len(self.num_unsheltered)
        #self.num_sheltered_avg = sum(self.num_sheltered)/len(self.num_sheltered)
        self.y = (self.server_build_rate[0] - 25)**2
        self.num_sys_2 = self.num_sys[T-1]**2
        self.h_2 = self.server_build_rate[0]**2
                
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
    x=np.arange(arr.shape[0])*63/365 # every 9 weeks
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

def create_chart_prob_dists(data_sim,q,t,binwidth):
    """
    create a chart of probabilities distribtutions

    Parameters
    ----------
    data_sim : np.array
        data from simulation model  
    q : queue
        the analytical queueing model
    t : int
        time to look at in years
    binwidth : int
        bin width for histogram, in number of people unsheltered

    Returns
    -------
    fix, ax : graph object

    """
    x = np.arange(100)
    data_q = [q.p_unsh[i][t*378] for i in range(100)]
    fig, ax = plt.subplots()
    ax.hist(data_sim[t*6], bins=range(min(data_sim[t*6]), max(data_sim[t*6]) + binwidth, binwidth), density = True)
    line, = ax.plot(x, data_q, color = 'black', linewidth = 1)
    plt.xlabel('# Unsheltered')
    plt.ylabel('Probability')
    plt.title('Probability density for number unsheltered after year ' +  str(t))
    first_legend = plt.legend(['Analytical model','Simulation model'])
    ax.add_artist(first_legend)

    return fig, ax

def create_chart_comparing_percentiles(data_sim,q,percentiles, index_list):
    """
    create a chart of probabilities distribtutions

    Parameters
    ----------
    data_sim : np.array
        data from simulation model  
    q : queue
        the analytical queueing model
    percentiles : list
        list of percentiles to include in plot (should be 3 percentiles)
    index_list : int
        list of indeces of the data to include for the analytical queueing model (bi monthly to match simulation data)

    Returns
    -------
    fix, ax : graph object

    """
    # set up
    fig, ax = plt.subplots()
    
    # x axis data
    x = np.arange(data_sim.shape[0])*63/365
    
    # y axis data - simulation
    data_sim_quant = [[0 for i in range(data_sim.shape[0])] for j in range(len(percentiles))]
    for i in range(data_sim.shape[0]):
        for j in range(len(percentiles)):
            data_sim_quant[j][i]=np.percentile(data_sim[i],percentiles[j])
            
    # y axis data - analytical q model
    data_q_quant = [[0 for i in range(len(index_list))] for j in range(len(percentiles))]
    for j in range(len(percentiles)):
        for t in range(len(index_list)):
            i=0
            prob = 0.0
            looking = True
            while looking == True:
                prob += q.p_unsh[i][index_list[t]]
                if(prob) >= percentiles[j]*0.01:
                    looking=False
                else: 
                    i+=1
            data_q_quant[j][t]=i

    # lines
    line1, = ax.plot(x, data_sim_quant[2], color = 'red', linewidth = 1)
    line2, = ax.plot(x, data_q_quant[2], color = 'red', linewidth = 1, linestyle='dashed')
    
    line3, = ax.plot(x, data_sim_quant[1], color = 'orange', linewidth = 1)
    line4, = ax.plot(x, data_q_quant[1], color = 'orange', linewidth = 1, linestyle='dashed')
    
    line5, = ax.plot(x, data_sim_quant[0], color = 'green', linewidth = 1)
    line6, = ax.plot(x, data_q_quant[0], color = 'green', linewidth = 1, linestyle='dashed')
    
    plt.xlabel('Year')
    plt.ylabel('# Unsheltered')
    plt.title('Number unsheltered - Percentiles ' + str(percentiles[0]) + ', ' + str(percentiles[1]) +  ', ' + str(percentiles[2]))
    
    first_legend = plt.legend([str(percentiles[2]) + 'th percentile - Simulation',
                               str(percentiles[2]) + 'th percentile - Analytical',
                               str(percentiles[1]) + 'th percentile - Simulation',
                               str(percentiles[1]) + 'th percentile - Analytical',
                               str(percentiles[0]) + 'th percentile - Simulation',
                               str(percentiles[0]) + 'th percentile - Analytical'])
    ax.add_artist(first_legend)
    
    return fig, ax

def analytic_cdf(n, q, nmax, yr):
    """
    Function to give the analytical queue CDF values for an input array
    
    Parameters
    ----------
    n : np.array
        This refers to the empirical distribution of values for the number of people unsheltered
    q : queue
        This is the queue against which to compare the analytical CDF
    nmax : int
        This is the max number of unsheltered people to evaluate the analytical CDF at
    yr : int
        This is the point in time at which to evaluate the analytical CDF

    Returns
    -------
    out : list
        This is a list of corresponding CDF values for the given input n. 
    
    """
    data = [q.p_unsh[0][yr*378]]
    for i in range(1,nmax):
        data.append(data[i-1] + q.p_unsh[i][yr*378])
    out = [data[i] for i in n]
    return out

def compare_cdf(data_sim, q, nmax, yr):
    """
    Function to display a chart comparing CDFs of the analytical queuing model output for #unsheltered with ecdf
    
    Parameters
    ----------
    data_sim : nparray
        simulation output
    q : queue
        the queue object
    nmax : int
        max number of people unsheltered to consider in the plot
    yr : int
        the year to look at.       


    Returns
    -------
    fig, ax : plot objects
    
    """
    # analytical q data
    data = [q.p_unsh[0][yr*378]]
    for i in range(1,nmax):
        data.append(data[i-1] + q.p_unsh[i][yr*378])
    q_out = [data[i] for i in range(nmax)]
    x1 = np.arange(nmax)
    
    # simulation data
    x2 = np.sort(data_sim[yr*6])
    sim_out = np.arange(len(x2))/float(len(x2))
    
    fig, ax = plt.subplots()
    #ax.hist(data_sim[t*6], bins=range(min(data_sim[t*6]), max(data_sim[t*6]) + binwidth, binwidth), density = True)
    line1, = ax.plot(x1, q_out, color = 'black', linewidth = 1)
    line2, = ax.plot(x2, sim_out, color = 'blue', linewidth = 1)
    plt.xlabel('# Unsheltered')
    plt.ylabel('Probability')
    plt.title('CDF for number unsheltered after year ' +  str(yr))
    first_legend = plt.legend(['Analytical model','Simulation model'])
    ax.add_artist(first_legend)
    
    return fig, ax

def get_percentiles(decision_yr, T, percentiles, data, i, j):
    """
    Parameters
    ----------
    decision_yr : int - year at which additional accomm built
    T : int - time duration to analyse in days
    percentiles : dict(int) - two percentiles (one high and one low) must be of form {'low': int, 'high' : int}
    data : list - data on probabilities
    i : int - index in list of possible 'additional shelter' values to analyse
    j : int - index in list of possible housing 'service times' values to analyse
    
    Returns
    -------
    out : dict(list) - percentiles of data in question
    """
    percentiles_high = [t for t in range(T)]
    percentiles_low = [t for t in range(T)]

    for t in range(T):
        looking_high = True
        looking_low = True
        n = 0
        prob_cumulative = 0
        while looking_high:
            prob_cumulative += data[i][j][decision_yr][n][t]
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
