# Imports
from __future__ import division
import pyomo.environ as pyo
from pyomo.core import Var
from pyomo.opt import SolverFactory

# Internal imports
import fluid_flow_model as fl

class Problem():

    def __init__(self, horizon):
    
        # Create Abstract Model
        self.problem = pyo.AbstractModel()
    
        # Levels
        self.problem.T = pyo.RangeSet(0, horizon-1)

        # Variables
        self.problem.h = Var(self.problem.T, domain=pyo.NonNegativeReals)
        self.problem.s = Var(self.problem.T, domain=pyo.NonNegativeReals)

    def solve(self, solver):
        self.opt=SolverFactory('glpk')
        self.instance=self.problem.create_instance()
        self.results=self.opt.solve(self.instance)
        self.h_opt=[self.instance.h[i].value for i in range(self.horizon)]
        self.s_opt=[self.instance.s[i].value for i in range(self.horizon)]
        
class Phi1(Problem):

    # Problem:
    # min TimeAvg(E[unsh(t)])
    # s.t. total budget constraint
    #      annual minimum build constraint
    
    def __init__(self, data, timestep, horizon, budget, costs_accomm, baseline_build):
        super(Phi1, self).__init__(horizon)

        # Set up model
        self.data = data
        self.timestep = timestep
        
        # Budget constraint
        self.budget = budget
        self.costs_accomm = costs_accomm
        self.problem.BUDGET = pyo.Constraint(rule=budget_constraint)

        # Baseline build constraints
        self.baseline_build = baseline_build
        self.problem.h_base = pyo.Constraint(self.problem.T,rule=min_house_build)
        self.problem.s_base = pyo.Constraint(self.problem.T,rule=min_shelter_build)

        # Objective function
        self.problem.OBJ = pyo.Objective(rule=objective_function)        
        
class FluidModel():

    def __init__(self, data, solution):
        self.model = fl.FluidFlowModel(data_as_is, solution)
        
    def analyse(self, horizon, timestep):
        self.T = [i*timestep for i in range(int(horizon/timestep))]
        self.model.analyse(self.T)

# Set as is data
data_as_is = {'initial_capacity' : {'housing':40, 'shelter':15},
              'initial_demand' : 180,
              'service_mean' : {'housing': 4.0, 'shelter': 0.0},
              'arrival_rates' : [100,100,100,100]}

# Model setup
horizon = 4
timestep = 1/365
budget = 132
baseline_build = 12
costs_accomm = {'housing' : 1.0, 'shelter' : 0.5}
model = FluidModel

# constraint funcs
def budget_constraint(problem):
    costs=0
    for t in problem.T:
        costs += problem.h[t] * costs_accomm['housing']
        costs += problem.s[t] * costs_accomm['shelter']
    return costs <= budget

def min_house_build(problem,n):
    return problem.h[n]>=baseline_build

def min_shelter_build(problem,n):
    return problem.s[n]>=baseline_build
        
def objective_function(problem):
    solution = {'housing' : problem.h, 'shelter' : problem.s}
    fluid_model = model(data_as_is, solution)
    fluid_model.analyse(horizon, timestep)
    return(sum(fluid_model.model.unsh_t)/len(fluid_model.model.unsh_t))

# Set up problem and solve
problem = Phi1(data_as_is, timestep, horizon, budget, costs_accomm, baseline_build)
problem.solve('glpk')

# Outputs
print('House building solution per year: ' + str(problem.h_opt))
# print('Shelter building solution per year: ' + str(problem.s_opt))
# print('Optimal objective Val: ' + str(problem.instance.OBJ()))
