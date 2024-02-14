# Imports
from __future__ import division
from pyomo.environ import *
from pyomo.core import Var
from pyomo.opt import SolverFactory

# Internal imports
import fluid_flow_model as fl

# Set as is data
data_as_is = {'initial_capacity' : {'housing':40, 'shelter':15},
              'initial_demand' : 180,
              'service_mean' : {'housing': 4.0, 'shelter': 0.0},
              'arrival_rates' : [100,100,100,100]}

# Model setup
horizon = 4
budget = 132
baseline_build = 12
cost = {'housing' : 1.0, 'shelter' : 0.5}

# Create Abstract Model
model = AbstractModel()
    
# Levels
model.T = RangeSet(0, horizon-1)

# Variables
model.h = Var(model.T, domain=NonNegativeReals)
model.s = Var(model.T, domain=NonNegativeReals)

# Constraints - budget
def set_budget(model):
    costs=0
    for i in model.T:
        costs += model.h[i] * cost['housing']
        costs += model.s[i] * cost['shelter']
    return costs <= budget

model.BUDGET=Constraint(rule=set_budget)

# Constraints: Lower Bound on build rates
def min_house_build(model,n):
    return model.h[n]>=baseline_build

def min_shelter_build(model,n):
    return model.s[n]>=baseline_build

model.h_base=Constraint(model.T,rule=min_house_build)
model.s_base=Constraint(model.T,rule=min_shelter_build)

# Objective function
def obj_expression(model):
    solution = {'housing' : model.h, 'shelter' : model.s}
    fluid_model = fl.FluidFlowModel(data_as_is, solution)
    T = [i/365 for i in range(int(horizon*365))]
    fluid_model.analyse(T)     
    output = sum(fluid_model.unsh_t)/len(fluid_model.unsh_t)
    return output

model.OBJ = Objective(rule=obj_expression)

# Solve
opt=SolverFactory('glpk')
instance=model.create_instance()
results=opt.solve(instance)

# Optimal sol
h_opt=[0 for i in range(horizon)]
s_opt=[0 for i in range(horizon)]
for i in range(horizon):
    h_opt[i]=instance.h[i].value
    s_opt[i]=instance.s[i].value

# Outputs
print('House building solution per year: ' + str(h_opt))
print('Shelter building solution per year: ' + str(s_opt))
print('Optimal objective Val: ' + str(instance.OBJ()))
