# Imports
from __future__ import division
import pyomo.environ as pyo
from pyomo.core import Var
from pyomo.opt import SolverFactory
import matplotlib.pyplot as plt
import math
import numpy as np

# Internal imports
import fluid_flow_model as fl

# constraint funcs
def budget_constraint(problem):
    costs=0
    for t in problem.T:
        costs += problem.h[t] * problem.costs_accomm['housing']
        costs += problem.s[t] * problem.costs_accomm['shelter']
    return costs <= problem.budget

def annual_budget_constraint(problem,n):
    return problem.h[n] * problem.costs_accomm['housing'] + problem.s[n] * problem.costs_accomm['shelter'] <= problem.annual_budget[n]

def min_house_build(problem,n):
    return problem.h[n]>=problem.baseline_build

def min_shelter_build(problem,n):
    return problem.s[n]>=problem.baseline_build

def h_up(problem,n):
    return (problem.h[n+1] >= problem.h[n])

def s_up(problem,n):
    return (problem.s[n+1] >= problem.s[n])

def s_down(problem,n):
    return (problem.s[n+1] <= problem.s[n])

# objective functions    
def y0(problem):
    " objective function for problem Phi0 "
    fluid_model = run_model(problem)
    avg_unsh = sum(fluid_model.model.unsh_t)/len(fluid_model.model.unsh_t)
    return(avg_unsh)

def y1(problem):
    " objective function for problem Phi1 "
    fluid_model = run_model(problem)
    avg_unsh = sum(fluid_model.model.unsh_t)/len(fluid_model.model.unsh_t)
    avg_sh = sum(fluid_model.model.sh_t)/len(fluid_model.model.sh_t)
    return(avg_unsh + (problem.c * avg_sh))

def y2(problem):
    " objective function for problem Phi2 and Phi3 "
    fluid_model = run_model(problem)
    avg_unsh_2 = sum(fluid_model.model.unsh_sq_t)/len(fluid_model.model.unsh_sq_t)
    avg_sh_2 = sum(fluid_model.model.sh_sq_t)/len(fluid_model.model.sh_sq_t)
    return(avg_unsh_2 + (problem.c * avg_sh_2))

def y4(problem):
    " objective function for problem Phi4"
    fluid_model = run_model(problem)
    avg_unsh_2 = sum(fluid_model.model.unsh_sq_t)/len(fluid_model.model.unsh_sq_t)
    weight_avg_sh_2 = sum([fluid_model.model.sh_sq_t[i]*problem.c[i] for i in range(int(problem.horizon/problem.timestep))])
    return(avg_unsh_2 + weight_avg_sh_2)
    
# helper functions
def run_model(problem):
    solution = {'housing' : problem.h, 'shelter' : problem.s}
    fluid_model = problem.selected_model(problem.data, solution)
    fluid_model.analyse(problem.horizon, problem.timestep)
    return(fluid_model)

def get_linear_weight_func_int1(m, horizon, timestep):
    ' weight function which integrates to one'
    T = horizon/timestep
    w1max = 2/(T**2)
    w1 = m*w1max
    w0 = (1/T) - ((w1*T)/2)
    w = [i*w1 + w0 for i in range(int(horizon/timestep))]
    return(w)

def get_linear_weight_func(m, horizon, timestep):
    ' weight function which increases to a maximum which is level of unweighted avg'
    T = horizon/timestep
    w1max = 1/(T**2)
    w1 = m*w1max
    w0 = (1/T) - w1*T
    w = [i*w1 + w0 for i in range(int(horizon/timestep))]
    return(w)
    
def plot_weight_func(w, horizon, timestep):
    x = list(np.arange(0,horizon,timestep))
    w = [i*365 for i in w]
    fig, ax = plt.subplots()
    ax.plot(x,w)
    ax.set_ylim(bottom=0)
    ax.set(xlabel='time (yrs)', ylabel='weight',
           title='weight of penalty on n_shelters^2')
    ax.grid()
    plt.show()

class Problem():

    objective_funcs = {'phi0' : y0,
                       'phi1' : y1,
                       'phi2' : y2,
                       'phi3' : y2,
                       'phi4' : y4}
    
    def __init__(self, modeling_options):
        
        # Create Abstract Model
        self.problem = pyo.AbstractModel()

        # Attributes
        self.problem.horizon = modeling_options['horizon']
        
        # Levels
        self.problem.T = pyo.RangeSet(0, self.problem.horizon-1)
        
        # Variables
        self.problem.h = Var(self.problem.T, domain=pyo.NonNegativeReals)
        self.problem.s = Var(self.problem.T, domain=pyo.NonNegativeReals)

    def solve(self, solver):
        self.opt=SolverFactory(solver)
        self.instance=self.problem.create_instance()
        self.results=self.opt.solve(self.instance)
        self.h_opt=[self.instance.h[i].value for i in range(self.problem.horizon)]
        self.s_opt=[self.instance.s[i].value for i in range(self.problem.horizon)]

    def print_results(self):
        surplus_budget = self.problem.budget - self.problem.horizon*self.problem.baseline_build*(self.problem.costs_accomm['housing']+self.problem.costs_accomm['shelter'])
        extra_houses = sum([n - self.problem.baseline_build for n in self.h_opt])
        extra_housing_spending = extra_houses*self.problem.costs_accomm['housing']
        surp_prop_h = extra_housing_spending/surplus_budget
        extra_shelters = sum([n - self.problem.baseline_build for n in self.s_opt])
        extra_shelter_spending = extra_shelters*self.problem.costs_accomm['shelter']
        surp_prop_s = extra_shelter_spending/surplus_budget
        print('------- Optimal solution -------')
        print('Number of housing units to build annually: ' + str([round(i,2) for i in self.h_opt]))
        print('Number of shelter units to build annually: ' + str([round(i,2) for i in self.s_opt]))
        print('Total budget: ' + str(self.problem.budget))
        print('Number of housing/shelter units which must be built annually: ' + str(self.problem.baseline_build))
        print('Surplus budget: ' + str(surplus_budget))
        print('Proportion of surplus budget spent on housing: ' + '{:.0%}'.format(surp_prop_h) + ': ' + str(round(extra_houses,2)) + ' extra houses.')
        print('Proportion of surplus budget spent on shelter: ' + '{:.0%}'.format(surp_prop_s) + ': ' + str(round(extra_shelters,2)) + ' extra shelters.')
        print('Optimal objective val: ' + str(round(self.instance.OBJ(),2)))

    def print_results_phi5(self):
        print('------- Optimal solution -------')
        print('Number of housing units to build annually: ' + str([round(i,2) for i in self.h_opt]))
        print('Number of shelter units to build annually: ' + str([round(i,2) for i in self.s_opt]))
        print('Proportion of annual budget spent on housing: ' + str(['{:.0%}'.format((self.h_opt[i]*self.problem.costs_accomm['housing'])/(self.problem.annual_budget[i])) for i in range(len(self.h_opt))]))
        print('Proportion of annual budget spent on shelter: ' + str(['{:.0%}'.format((self.s_opt[i]*self.problem.costs_accomm['shelter'])/(self.problem.annual_budget[i])) for i in range(len(self.s_opt))]))
        print('Optimal objective val: ' + str(round(self.instance.OBJ(),2)))
    
    def plot_opt(self):
        # run model at optimal
        solution = {'housing' : self.h_opt, 'shelter' : self.s_opt}
        model = self.problem.selected_model(self.problem.data, solution)
        model.analyse(self.problem.horizon, self.problem.timestep)

        # general plotting
        x = math.floor(self.problem.horizon*365)/365
        fig, ax = plt.subplots(1,2, figsize=(12, 5))
        ymax = max(model.model.h_t + model.model.sh_t + model.model.unsh_t)
        
        # plot optimal solution
        ax[0].plot(np.arange(0,x,1/365), model.model.h_t, color = 'blue')
        ax[0].plot(np.arange(0,x,1/365), model.model.sh_t, color = 'green')
        ax[0].set(xlabel='time (yrs)', ylabel='# units',
               title='Accommodation stock for optimal solution')
        ax[0].legend(["houses", "shelters"], loc="lower right")
        ax[0].grid()
        ax[0].set_ylim(0,ymax*1.05)

        # plot objective function for optimal solution
        ax[1].plot(np.arange(0,x,1/365), model.model.sh_t, color = 'green')
        ax[1].plot(np.arange(0,x,1/365), model.model.unsh_t, color = 'red')
        ax[1].set(xlabel='time (yrs)', ylabel='# in queue',
               title='Expected number in queue')
        ax[1].legend(["sheltered", "unsheltered"], loc="lower right")
        ax[1].grid()
        ax[1].set_ylim(0,ymax*1.05)
        
        # general
        fig.subplots_adjust(wspace = 0.5)
        plt.show()

    def plot_obj_curve(self, n):

        # get data
        surp = self.problem.budget - self.problem.horizon*self.problem.baseline_build*(self.problem.costs_accomm['housing']+self.problem.costs_accomm['shelter'])
        out = []
        for i in range(n+1):
            solution = {'housing' : [self.problem.baseline_build]*self.problem.horizon, 'shelter' : [self.problem.baseline_build]*self.problem.horizon}
            solution['housing'][0] += ((i/n)*surp)/self.problem.costs_accomm['housing']
            solution['shelter'][0] += ((1-i/n) * surp)/self.problem.costs_accomm['shelter']
            model = self.problem.selected_model(self.problem.data, solution)
            model.analyse(self.problem.horizon, self.problem.timestep)
            avg_unsh_2 = sum(model.model.unsh_sq_t)/len(model.model.unsh_sq_t)
            avg_sh_2 = sum(model.model.sh_sq_t)/len(model.model.sh_sq_t)
            out.append(avg_unsh_2 + (self.problem.c * avg_sh_2))
        
        # plot
        x = [i/n for i in range(n+1)]
        fig, ax = plt.subplots()
        ax.scatter(x,out)
        ax.set(xlabel='Proportion surplus spent on housing in yr 1', ylabel='Objective function',
               title='Objective function for different solutions')
        plt.show()

class Phi(Problem):

    def __init__(self, data, modeling_options, obj, c=1):
        super(Phi, self).__init__(modeling_options)

        # Set up model
        self.problem.data = {key: data[key] for key in data.keys() & {'initial_capacity', 'initial_demand', 'service_mean', 'arrival_rates'}}
        self.problem.timestep = modeling_options['timestep']
        self.problem.selected_model = modeling_options['model']
        self.problem.budget = data['budget']
        self.problem.costs_accomm = data['costs_accomm']
        self.problem.baseline_build = data['baseline_build']
        self.problem.c = c

        # Objective function
        self.problem.OBJ = pyo.Objective(rule=self.objective_funcs[obj])

    def add_total_budget_constraint(self):
        # Budget constraint
        self.problem.BUDGET = pyo.Constraint(rule=budget_constraint)

    def add_annual_budget_constraint(self, proportions):
        self.problem.annual_budget = [self.problem.budget*p for p in proportions]
        self.problem.BUDGET = pyo.Constraint(self.problem.T, rule=annual_budget_constraint)
    
    def add_baseline_build_constraint(self):
        # Baseline build constraints
        self.problem.h_base = pyo.Constraint(self.problem.T,rule=min_house_build)
        self.problem.s_base = pyo.Constraint(self.problem.T,rule=min_shelter_build)

    def add_housing_increase(self):
        self.problem.T_housing = pyo.RangeSet(0, self.problem.horizon-2)
        self.problem.h_increase = pyo.Constraint(self.problem.T_housing, rule = h_up)
            
    def add_shelter_increase_decrease(self, shelter_mode):
        self.problem.T_shelter_first = pyo.RangeSet(0, shelter_mode-2)
        self.problem.T_shelter_second = pyo.RangeSet(shelter_mode-1, self.problem.horizon-2)
        self.problem.s_increase = pyo.Constraint(self.problem.T_shelter_first, rule = s_up)
        self.problem.s_decrease = pyo.Constraint(self.problem.T_shelter_second, rule = s_down)

class FluidModel():

    def __init__(self, data, solution):
        self.model = fl.FluidFlowModel(data, solution)
        
    def analyse(self, horizon, timestep):
        self.T = [i*timestep for i in range(int(horizon/timestep))]
        self.model.analyse(self.T)