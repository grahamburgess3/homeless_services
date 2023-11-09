# external imports
import numpy as np
import pandas as pd
import math
from datetime import datetime

# internal imports
import queueing_model as qm

def generate_solution_space(build_rate_options, annual_budget, total_budgets, costs, simulation_length):
    """
    
    """

    previous_options = [{'housing' : [], 'shelter' : []}]
    # go through each year
    for yr in range(simulation_length):
        new_options = []
        for h in range(len(build_rate_options['housing'])):
            for s in range(len(build_rate_options['shelter'])):
                if build_rate_options['housing'][h]*costs['housing'] + build_rate_options['shelter'][s]*costs['shelter'] <= annual_budget:
                    for opt in previous_options:
                        if sum(opt['housing']) + build_rate_options['housing'][h] <= total_budgets['housing'] and sum(opt['shelter']) + build_rate_options['shelter'][s] <= total_budgets['shelter']:
                            new_opt = {'housing':opt['housing'].copy(),'shelter':opt['shelter'].copy()}
                            new_opt['housing'].append(build_rate_options['housing'][h])
                            new_opt['shelter'].append(build_rate_options['shelter'][s])
                            new_options.append(new_opt)
        previous_options = new_options.copy()
    return new_options

def generate_solution_space_generic(extra_shelter, build_rate_length_years, annual_budget):
    """
    return list of dictionaries, each dict giving a solution in terms of shelter and housing build rates
    """
    # solution space constraints
    build_rate_options = {'housing' : [int(annual_budget/3), int(annual_budget*2/3)], 
                          'shelter' : [int((annual_budget/3) + extra_shelter), int((annual_budget*2/3) + (2 * extra_shelter))]}
    accommodation_budgets = {'housing' : int((annual_budget*2/3) * (build_rate_length_years - 0.5)), 
                             'shelter' : int((annual_budget*2/3 + 2 * extra_shelter) * (build_rate_length_years - 0.5))}
    costs = {'housing': 1, 
             'shelter': annual_budget / (annual_budget + (3 * extra_shelter))}

    # geneate solution space
    sols = generate_solution_space(build_rate_options,
                                   annual_budget, 
                                   accommodation_budgets, 
                                   costs, 
                                   build_rate_length_years)

    sols = keep_max_budget(sols, costs, annual_budget)
    
    return(sols)

def keep_max_budget(sols, costs, annual_budget):
    """
    return a subset of solutions in which the annual budget is used up every year
    """
    sol_subset = []
    for i in range(len(sols)):
        keep = True
        for j in range(len(sols[0]['housing'])):
            if sols[i]['housing'][j]*costs['housing'] + sols[i]['shelter'][j]*costs['shelter'] < annual_budget:
                keep = False
        if keep == True:
            sol_subset.append(sols[i])
    return sol_subset

class SolutionSpace():
    """
    A class to represent a set of solutions making up a solution space, which can be tested for cost using stochastic simulation

    Attributes (Instance)
    ----------
    solutions : list(Solution)
       the set of solutions in question
    costs : list(list)
       for each solution, a list of the initial realisations of cost, from the simulation
    covar : np.array
       the covariance matrix for the costs of each solution
    active : np.array(bool)
       True for each solution if it is still in consideration to be the best
    eliminate : list
       Details the iteration at which each solution was rejected (0 if not rejected)
        
        """
    def __init__(self, solutions):
        """
        Constructs the initial attributes for an instance of SolutionSpace

        Parameters
        ----------
        solutions : list(dict(list))
           a dictionary for each solution, each containing a list of the build rates for each type of accommodation, for that solution

        """
        self.solutions = [Solution(solutions[i]) for i in range(len(solutions))]
        self.costs = [[] for i in solutions]
        self.costs_initial_reps = [[] for i in solutions]
        self.covar = None
        self.active = np.array([True for i in solutions])
        self.eliminate = [0 for i in solutions]
        self.true_best = {}
        self.true_costs = {}
        self.true_outputs_sh = []
        self.true_outputs_unsh = []

    def model_analytically(self, data_as_is, data_as_is_analytical, analysis_horizon, service_mean_housing):
        """
        get data from the analytical model for each solution in the solution space, with a specified service time info
        """
        for s in range(len(self.solutions)):
            q = qm.queue(data_as_is['arrival_rates'],
                         {'housing' : service_mean_housing, 'shelter' : 0}, 
                         data_as_is['initial_capacity'], 
                         self.solutions[s].solution, 
                         data_as_is['initial_demand'], 
                         data_as_is_analytical['max_in_system'],
                         data_as_is['time_btwn_changes_in_build_rate'],
                         data_as_is['time_btwn_building'])
            q.model_dynamics(analysis_horizon,
                             data_as_is_analytical['delta_t'])
            self.true_outputs_sh.append(q.num_sheltered_avg)
            self.true_outputs_unsh.append(q.num_unsheltered_avg)

    def find_true_best(self, obj_function, obj_function_desc):
        """
        based on a given objective function, find the true best solution based on the data collected from analytical model
        """
        costs = []
        for i in range(len(self.solutions)):
            costs.append(obj_function(self.true_outputs_sh[i],self.true_outputs_unsh[i]))

        best_one_index = costs.index(min(costs))
        self.true_best[obj_function_desc] = [i for i in range(len(costs)) if costs[i] == costs[best_one_index]]
            
    def run_initial_reps(self, n0, sim, **kwargs):
        # get first n0 solutions
        for x in range(len(self.solutions)):
            for rep in range(n0):
                cost = sim(self.solutions[x].solution, **kwargs)
                self.costs_initial_reps[x].append(cost)

        # get initial covariance
        self.covar = np.cov(np.array(self.costs_initial_reps))
        
    def optimise_rs(self, alpha, n0, delta, sim, initial_reps_needed, print_progress, **kwargs):
        """
        Find an optimal solution using the KN Ranking & Selection algorithm (see Nelson and Pei, 2021, Chapter 9)
        Callin this function updates the self.active attribute to leave only one True element - this is the optimal solution

        Parameters
        ----------
        alpha : float
           (1-alpha) desired confidence. Assuming the true optimal is at least delta better than all the rest, this routine will select the best with probability 1-alpha
        n0 : int
           The number of initial replications of each solution to simulate before elimination begins
        delta : float
           The indifference zone - i.e. the smallest difference in expected cost which is of practical importance.
        sim : function
           A function which takes as input self.solutions[i] for i in range(len(self.solutions)) and returns the cost of that solution
        initial_reps_needed : bool
           True if the initial simulation is neeeded (for example if this has already been done)
        print_progress : bool
           Indicate whether print statements of progress in the optimisation is desired
        """
        # start
        if print_progress == True:
            print('starting routine at time  ' + str(datetime.now()))

        # get first n0 solutions (if needed)
        if initial_reps_needed == True:
            self.run_initial_reps(n0, sim, **kwargs)
            if print_progress == True:
                print('done init reps at time  ' + str(datetime.now()))

        # setup for iteration
        self.active = np.array([True for i in self.solutions])
        self.eliminate = [0 for i in self.solutions]
        self.costs = [self.costs_initial_reps[i].copy() for i in range(len(self.costs_initial_reps))]
        eta = 0.5*( (2*alpha/(len(self.solutions)-1))**(-2/(n0-1)) -1)
        t2 = 2*eta*(n0-1)
        costs_sum = np.sum(np.array(self.costs), axis=1) # sum of costs over each replication, for each solution
        r = n0 # iteration number
        sol_index = np.array([i for i in range(len(self.solutions))]) # represent the index of each solution

        # elimination loop
        num_active_old = np.sum(self.active)
        num_active_new = num_active_old
        while np.sum(self.active) > 1:
            num_active_diff = num_active_old-num_active_new
            if (num_active_diff > 0 and print_progress == True):
                print('start iteration ' + str(r+1) + ' with ' + str(np.sum(self.active)) + ' active solutions out of initial ' + str(len(self.solutions)) + ' at time ' + str(datetime.now()))
            num_active_old = np.sum(self.active)                
            r += 1
            a_temp = self.active.copy()

            # collect new data
            for i in sol_index[self.active]:
                cost = sim(self.solutions[i].solution, **kwargs)
                self.costs[i].append(cost)
                costs_sum[i] += cost

            # elimination
            for i in sol_index[self.active]:
                for j in sol_index[self.active]:
                    covar_diff = self.covar[i,i] + self.covar[j,j] - 2 * self.covar[i,j]
                    W = max(0, (delta/2)*(t2*covar_diff/delta**2 - r))
                    if costs_sum[i] > costs_sum[j] + W:
                        a_temp[i] = False
                        self.eliminate[i] = r
                        break
                        
            self.active = a_temp.copy()
            num_active_new = np.sum(self.active)
        
class Solution():
    """
    A class to represent a single solution, which can be tested for cost using stochastic simulation

    Attributes (Instance)
    ----------
    solution : dict
       A dictionary which can be the argument to the function which runs the stochastic simulation

    """
    def __init__(self, solution):
        """
        Constructs the initial attributes for an instance of Solution

        Parameters
        ----------
        solution : dict(list)
           a list of the build rates for each type of accommodation

        """
        self.solution = solution

def InventorySystem(x, n=1, RandomSeed=-1):
  # simulates the (s,S) inventory example of Koenig & Law
  # x in {1,2,...,1600} is the system index
  # n = number of replications
  # output is average cost for 30 periods
  
    littleS = math.ceil(x/40)
    bigS = littleS + x - (littleS - 1)*40
  
    Y = []
    for j in range(n):
        InvtPos = bigS
        Cost = 0
        for period in range(30):
            Demand = np.random.poisson(lam=25,size=1)[0]
            if InvtPos < littleS:
                INext = bigS
                Cost = Cost + 32 +3*(bigS - InvtPos)
            else:
                INext = InvtPos
            if (INext - Demand >= 0):
                Cost = Cost + INext - Demand
            else:
                Cost = Cost + 5*(Demand - INext) 
            InvtPos = INext - Demand
        Y.append(Cost/30)
    return Y

def get_sS_system(x):
    """
    Returns the (two dimensional s,S parameters for solution x
    """
    littleS = math.ceil(x/40)
    bigS = littleS + x - (littleS - 1)*40
    return littleS, bigS
