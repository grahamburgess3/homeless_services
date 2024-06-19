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
import queueing_model as qm

class FluidModel():

    def __init__(self, data, horizon, timestep, max_in_system, solution):
        self.model = fl.FluidFlowModel(data, horizon, timestep, solution)
        
    def analyse(self, horizon, timestep):
        self.T = [i*timestep for i in range(int(horizon/timestep))]
        self.model.analyse(self.T)
        self.model.num_sys = self.model.n_t

class AnalyticalQueueModel():

    def __init__(self, data, horizon, timestep, max_in_system, solution):
        self.model = qm.queue(data['arrival_rates'], data['service_mean'], data['initial_capacity'], solution, data['initial_demand'], max_in_system)
        
    def analyse(self, horizon, timestep):
        self.model.model_dynamics(horizon,
                                  timestep*365)
        
# objective functions    
def y0(problem):
    " objective function for problem Phi0 "
    solution = {'housing' : problem.h, 'shelter' : problem.s}
    model = run_model(problem, solution)
    avg_unsh = sum(model.model.unsh_t)/len(model.model.unsh_t)
    return(avg_unsh)

def y1(problem):
    " objective function for problem Phi1 "
    solution = {'housing' : problem.h, 'shelter' : problem.s}
    model = run_model(problem, solution)
    avg_unsh = sum(model.model.unsh_t)/len(model.model.unsh_t)
    avg_sh = sum(model.model.sh_t)/len(model.model.sh_t)
    return(avg_unsh + (problem.c * avg_sh))

def y2(problem):
    " objective function for problem Phi2 and Phi3 "
    solution = {'housing' : problem.h, 'shelter' : problem.s}
    model = run_model(problem, solution)
    avg_unsh_2 = sum(model.model.unsh_sq_t)/len(model.model.unsh_sq_t)
    avg_sh_2 = sum(model.model.sh_sq_t)/len(model.model.sh_sq_t)
    return(avg_unsh_2 + (problem.c * avg_sh_2))

def y4(problem):
    " objective function for problem Phi4"
    solution = {'housing' : problem.h, 'shelter' : problem.s_positive}
    model = run_model(problem, solution)
    avg_unsh_2 = sum(model.model.unsh_sq_t)/len(model.model.unsh_sq_t)
    weight_avg_sh_2 = sum([model.model.sh_sq_t[i]*problem.c[i] for i in range(int(problem.horizon_model/problem.timestep))])
    return(avg_unsh_2 + weight_avg_sh_2)

def y5(problem):
    " objective function for problem Phi4"
    solution = {'housing' : problem.h, 'shelter' : problem.s}
    model = run_model(problem, solution)
    avg_unsh_2 = sum(model.model.unsh_sq_t)/len(model.model.unsh_sq_t)
    weight_avg_sh_2 = sum([model.model.sh_sq_t[i]*problem.c[i] for i in range(int(problem.horizon_model/problem.timestep))])
    return(avg_unsh_2 + weight_avg_sh_2)

def y6(problem):
    " objective function for problem Phi6"
    solution = {'housing' : problem.h, 'shelter' : problem.s}
    model = run_model(problem, solution)
    return(model.model.y)

def y7(problem):
    " objective function for problem Phi6"
    solution = {'housing' : problem.h, 'shelter' : problem.s}
    model = run_model(problem, solution)
    return(model.model.h_2 + 0.1*model.model.num_sys_2)

class Phi():

    objective_funcs = {'phi0' : y0,
                       'phi1' : y1,
                       'phi2' : y2,
                       'phi3' : y2,
                       'phi4' : y4,
                       'phi5' : y5,
                       'phi6' : y6,
                       'phi7' : y7}

    def __init__(self, data, modeling_options, obj, c=1, q_model_options = {'bigM' : 0, 'max_in_system' : 0}):
        
        # Create Abstract Model
        self.problem = pyo.AbstractModel()

        # Attributes
        self.problem.horizon_model = modeling_options['horizon_model']
        self.problem.horizon_decision = modeling_options['horizon_decision']
        self.problem.timestep = modeling_options['timestep']
        self.problem.selected_model = modeling_options['model']
        self.problem.M = q_model_options['bigM']
        self.problem.max_in_system = q_model_options['max_in_system']
        self.problem.data = data
        self.problem.budget = data['budget']
        self.problem.costs_accomm = data['costs_accomm']
        self.problem.baseline_build = data['baseline_build']
        self.problem.c = c
        
        # Levels
        self.problem.T = pyo.RangeSet(0, self.problem.horizon_decision-1)
        
        # Variables
        self.problem.h = Var(self.problem.T, domain=pyo.NonNegativeReals)
        self.problem.s = Var(self.problem.T, domain=pyo.Reals)
        self.problem.s_positive = Var(self.problem.T, domain=pyo.NonNegativeReals)

        # Objective function
        self.problem.OBJ = pyo.Objective(rule=self.objective_funcs[obj])

    def add_total_budget_constraint(self):
        self.problem.BUDGET = pyo.Constraint(rule=budget_constraint)

    def add_annual_budget_constraint(self, proportions):
        self.problem.annual_budget = [self.problem.budget*p for p in proportions]
        self.problem.BUDGET = pyo.Constraint(self.problem.T, rule=annual_budget_constraint)
    
    def add_baseline_build_constraint(self):
        self.problem.h_base = pyo.Constraint(self.problem.T,rule=min_house_build)
        self.problem.s_base = pyo.Constraint(self.problem.T,rule=min_shelter_build)

    def add_housing_increase(self):
        self.problem.T_housing = pyo.RangeSet(0, self.problem.horizon_decision-2)
        self.problem.h_increase = pyo.Constraint(self.problem.T_housing, rule = h_up)
            
    def add_shelter_increase_decrease(self, shelter_mode):
        self.problem.T_shelter_first = pyo.RangeSet(0, shelter_mode-1)
        self.problem.T_shelter_second = pyo.RangeSet(shelter_mode, self.problem.horizon_decision-2)
        self.problem.s_increase = pyo.Constraint(self.problem.T_shelter_first, rule = s_up)
        self.problem.s_decrease = pyo.Constraint(self.problem.T_shelter_second, rule = s_down)

    def add_shelter_positive_negative(self, shelter_mode):
        self.problem.T_shelter_positive = pyo.RangeSet(0, shelter_mode-1)
        self.problem.T_shelter_incr = pyo.RangeSet(0, shelter_mode-2)
        self.problem.T_shelter_negative = pyo.RangeSet(shelter_mode, self.problem.horizon_decision-1)
        self.problem.s_positive = pyo.Constraint(self.problem.T_shelter_positive, rule = s_plus)
        self.problem.s_increase = pyo.Constraint(self.problem.T_shelter_incr, rule = s_up)
        self.problem.s_negative = pyo.Constraint(self.problem.T_shelter_negative, rule = s_minus)
        self.problem.s_not_too_negative = pyo.Constraint(self.problem.T_shelter_negative, rule = s_not_too_minus)
        self.problem.budget_checked_annually = pyo.Constraint(self.problem.T, rule = budget_constraint_checked_annually)
    
    def add_queue_model_constraints(self):
        self.problem.m_lessthan_n = pyo.Constraint(self.problem.dT, self.problem.n, rule = m_lessthan_n)
        self.problem.m_lessthan_s = pyo.Constraint(self.problem.dT, self.problem.n, rule = m_lessthan_s)
        self.problem.m_morethan_n = pyo.Constraint(self.problem.dT, self.problem.n, rule = m_morethan_n)
        self.problem.m_morethan_s = pyo.Constraint(self.problem.dT, self.problem.n, rule = m_morethan_s)

    def add_no_build_constraints(self):
        self.problem.s_build = pyo.Constraint(self.problem.T,rule=no_shelter_build)
        self.problem.T_minus_1 = pyo.RangeSet(1, self.problem.horizon_decision-1)
        self.problem.h_build = pyo.Constraint(self.problem.T_minus_1,rule=no_housing_build)

    def solve(self, solver):
        self.opt=SolverFactory(solver)
        self.instance=self.problem.create_instance()
        self.results=self.opt.solve(self.instance)
        self.h_opt=[self.instance.h[i].value for i in range(self.problem.horizon_decision)]
        self.s_opt=[self.instance.s[i].value for i in range(self.problem.horizon_decision)]

    def print_results(self):
        surplus_budget = self.problem.budget - self.problem.horizon_decision*self.problem.baseline_build*(self.problem.costs_accomm['housing']+self.problem.costs_accomm['shelter'])
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

    def print_results_phi3(self):
        print('------- Optimal solution -------')
        print('Number of housing units to build annually: ' + str([round(i,2) for i in self.h_opt]))
        print('Number of shelter units to build annually: ' + str([round(i,2) for i in self.s_opt]))
        print('Proportion of total budget spent on housing: ' + str(['{:.0%}'.format((self.h_opt[i]*self.problem.costs_accomm['housing'])/(self.problem.budget)) for i in range(len(self.h_opt))]))
        print('Proportion of total budget spent on shelter: ' + str(['{:.0%}'.format((self.s_opt[i]*self.problem.costs_accomm['shelter'])/(self.problem.budget)) for i in range(len(self.s_opt))]))
        print('Optimal objective val: ' + str(round(self.instance.OBJ(),2)))
    
    def print_results_phi4(self):
        print('------- Optimal solution -------')
        print('Number of housing units to build annually: ' + str([round(i,2) for i in self.h_opt]))
        print('Number of shelter units to build annually: ' + str([round(i,2) for i in self.s_positive_opt]))
        print('Proportion of annual budget spent on housing: ' + str(['{:.0%}'.format((self.h_opt[i]*self.problem.costs_accomm['housing'])/(self.problem.annual_budget[i])) for i in range(len(self.h_opt))]))
        print('Proportion of annual budget spent on shelter: ' + str(['{:.0%}'.format((self.s_positive_opt[i]*self.problem.costs_accomm['shelter'])/(self.problem.annual_budget[i])) for i in range(len(self.s_positive_opt))]))
        print('Optimal objective val: ' + str(round(self.instance.OBJ(),2)))
    
    def plot_opt(self, solution):
        # run model at optimal
        model = self.problem.selected_model(self.problem.data, self.problem.horizon_decision, self.problem.timestep, self.problem.max_in_system, solution)
        model.analyse(self.problem.horizon_model, self.problem.timestep)

        # general plotting
        x = math.floor(self.problem.horizon_model*365)/365
        fig, ax = plt.subplots()
        ymax = max(model.model.h_t + model.model.sh_t + model.model.unsh_t)
        
        # plot optimal solution
        ax.plot(np.arange(0,x,1/365), model.model.h_t, color = 'green')
        ax.plot(np.arange(0,x,1/365), model.model.sh_t, color = 'orange')
        ax.plot(np.arange(0,x,1/365), model.model.unsh_t, color = 'red')
        ax.set(xlabel='Time (yrs)', ylabel='Number of people',
               title='Number of people housed/sheltered/unsheltered')
        ax.legend(["$h_t$", "$s_t$", "$u_t$"], loc="lower right")
        ax.grid()
        ax.set_ylim(0,ymax*1.05)
        
        # general
        plt.show()

    def plot_obj_curve(self, n):

        # get data
        surp = self.problem.budget - self.problem.horizon_decision*self.problem.baseline_build*(self.problem.costs_accomm['housing']+self.problem.costs_accomm['shelter'])
        out = []
        for i in range(n+1):
            solution = {'housing' : [self.problem.baseline_build]*self.problem.horizon_decision, 'shelter' : [self.problem.baseline_build]*self.problem.horizon_decision}
            solution['housing'][0] += ((i/n)*surp)/self.problem.costs_accomm['housing']
            solution['shelter'][0] += ((1-i/n) * surp)/self.problem.costs_accomm['shelter']
            model = self.problem.selected_model(self.problem.data, self.problem.horizon_decision, self.problem.timestep, self.max_in_system, solution)
            model.analyse(self.problem.horizon_model, self.problem.timestep)
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

class Phi0(Phi):
    def __init__(self, data, modeling_options, obj):
        super(Phi0, self).__init__(data, modeling_options, obj)
        self.add_total_budget_constraint()
        self.add_baseline_build_constraint()

    def solve(self, solver):
        super(Phi0, self).solve(solver)
        self.print_results()
        solution = {'housing' : self.h_opt, 'shelter' : self.s_opt}
        self.plot_opt(solution)

class Phi1(Phi):
    def __init__(self, data, modeling_options, obj, c):
        super(Phi1, self).__init__(data, modeling_options, obj, c)
        self.add_total_budget_constraint()
        self.add_baseline_build_constraint()

    def solve(self, solver):
        super(Phi1, self).solve(solver)
        self.print_results()
        solution = {'housing' : self.h_opt, 'shelter' : self.s_opt}
        self.plot_opt(solution)

class Phi2(Phi1):
    def __init__(self, data, modeling_options, obj, c):
        super(Phi2, self).__init__(data, modeling_options, obj, c)

    def solve(self, solver):
        super(Phi2, self).solve(solver)

class Phi3(Phi):
    def __init__(self, data, modeling_options, obj, c, shelter_mode):
        super(Phi3, self).__init__(data, modeling_options, obj, c)
        self.problem.del_component(self.problem.s_positive)
        self.add_total_budget_constraint()
        self.add_housing_increase()
        self.add_shelter_positive_negative(shelter_mode) 
    
    def solve(self, solver):
        super(Phi3, self).solve(solver)
        self.print_results_phi3()
        solution = {'housing' : self.h_opt, 'shelter' : self.s_opt}
        self.plot_opt(solution)

class Phi4(Phi):
    def __init__(self, data, modeling_options, obj, c, budget_props):
        super(Phi4, self).__init__(data, modeling_options, obj, c)
        self.add_annual_budget_constraint(budget_props)

    def solve(self, solver):
        super(Phi4, self).solve(solver)
        self.s_positive_opt=[self.instance.s_positive[i].value for i in range(self.problem.horizon_decision)]
        self.print_results_phi4()
        solution = {'housing' : self.h_opt, 'shelter' : self.s_positive_opt}
        self.plot_opt(solution)

class Phi5(Phi3):
    def __init__(self, data, modeling_options, obj, c, shelter_mode):
        super(Phi5, self).__init__(data, modeling_options, obj, c, shelter_mode)
    
    def solve(self, solver):
        super(Phi5, self).solve(solver)

class Phi6(Phi):
    def __init__(self, data, modeling_options, obj, q_model_options):
        super(Phi6, self).__init__(data, modeling_options, obj, q_model_options = q_model_options)
        self.problem.dT = pyo.Set(initialize = range(int(round(self.problem.horizon_model / modeling_options['timestep'],0))))
        self.problem.n = pyo.Set(initialize = range(self.problem.max_in_system))
        self.problem.dT_n = pyo.Set(initialize = self.problem.dT*self.problem.n)
        self.problem.y = Var(self.problem.dT_n, domain=pyo.Binary)
        self.problem.m = Var(self.problem.dT_n, domain = pyo.NonNegativeReals)
        self.add_total_budget_constraint()
        self.add_no_build_constraints() # ensure no shelter is built at all, and housing only built in year 1. 

    def solve(self, solver):
        super(Phi6, self).solve(solver)
        self.print_results()

class Phi7(Phi6):
    def __init__(self, data, modeling_options, obj, q_model_options):
        super(Phi6, self).__init__(data, modeling_options, obj, q_model_options = q_model_options)

    def solve(self, solver):
        super(Phi6, self).solve(solver)

# constraint funcs
def budget_constraint(problem):
    costs=0
    for t in problem.T:
        costs += problem.h[t] * problem.costs_accomm['housing']
        costs += problem.s[t] * problem.costs_accomm['shelter']
    return costs <= problem.budget

def budget_constraint_checked_annually(problem,n):
    return sum([problem.h[t] * problem.costs_accomm['housing'] + problem.s[t] * problem.costs_accomm['shelter'] for t in range(n+1)]) <= problem.budget

def annual_budget_constraint(problem,n):
    return problem.h[n] * problem.costs_accomm['housing'] + problem.s_positive[n] * problem.costs_accomm['shelter'] <= problem.annual_budget[n]

def min_house_build(problem,n):
    return problem.h[n]>=problem.baseline_build

def min_shelter_build(problem,n):
    return problem.s[n]>=problem.baseline_build

def no_shelter_build(problem,n):
    return problem.s[n] == 0

def no_housing_build(problem,n):
    return problem.h[n] == 0

def h_up(problem,n):
    return (problem.h[n+1] >= problem.h[n])

def s_up(problem,n):
    return (problem.s[n+1] >= problem.s[n])

def s_down(problem,n):
    return (problem.s[n+1] <= problem.s[n])

def s_plus(problem,n):
    return (problem.s[n] >= 0)

def s_minus(problem,n):
    return (problem.s[n] <= 0)

def s_not_too_minus(problem,n):
    stock = problem.data['initial_capacity']['shelter'] + sum([problem.s[t] for t in range(n)])
    return (problem.s[n] >= -stock)

def m_lessthan_n(problem, t, n):
    return problem.m[t][n] <= n

def m_lessthan_s(problem, t, n):
    return problem.m[t][n] <= num_serve(problem, t)

def m_morethan_n(problem, t, n):
    return problem.m[t][n] >= n - problem.M*(1-problem.y)

def m_morethan_s(problem, t, n):
    return problem.m[t][n] >= num_serve(problem, t) - problem.M*problem.y
    
# helper functions
def run_model(problem, solution):
    model = problem.selected_model(problem.data, problem.horizon_decision, problem.timestep, problem.max_in_system, solution)
    model.analyse(problem.horizon_model, problem.timestep)
    return(model)

def num_serve(problem, t):
    """
    returns number of servers at time t days

    Parameters
    ----------
    t : float
        time in days.

    Returns
    -------
    num_serve: num servers at time t.

    """

    t = t*problem.timestep # time in years
    num_serve = 0
    n = problem.data['initial_capacity']['housing']
    
    # add complete years
    yrs = math.floor(t) # number of years passed
    for yr in range(yrs):
        n += problem.h[yr]
        
    # add fractional year
    n += (t % 1) * problem.h[yrs]

    n = pyo.floor(n)
        
    return n

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