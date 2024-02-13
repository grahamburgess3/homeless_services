# Imports
from __future__ import division
from pyomo.environ import *
from pyomo.core import Var
from pyomo.opt import SolverFactory

# Internal imports
import fluid_flow_model as fl

# Set as is data
data_as_is = {'initial_capacity' : {'housing':40, 'shelter':15},
              'initial_demand' : 120,
              'service_mean' : {'housing': 4.487179487179487, 'shelter': 0},
              'arrival_rates' : [36.55542857142857, 43.520228571428575, 47.76822857142857, 47.76822857142857, 43.12542857142857, 38.92062857142857]}

# Model setup
horizon = 5
budget = 100
cost = {'housing' : 1.0, 'shelter' : 0.5}

# Create Abstract Model
model = AbstractModel()
    
# Levels
model.T = RangeSet(0, horizon-1)

# Variables
model.h = Var(model.T, domain=NonNegativeReals)
model.s = Var(model.T, domain=NonNegativeReals)

# Constraints
def set_budget(model):
    costs=0
    for i in model.T:
        costs += model.h[i] * cost['housing']
        costs += model.s[i] * cost['shelter']
    return costs <= budget

model.BUDGET=Constraint(rule=set_budget)

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
