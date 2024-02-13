# Imports
from __future__ import division
from pyomo.environ import *
from pyomo.core import Var
import matplotlib.pyplot as plt
import numpy as np;
from pyomo.opt import SolverFactory

# Internal imports
import fluid_flow_model as fl

# Model setup
horizon = 5
budget = 100
cost = {'housing' : 1.0, 'shelter' : 0.5}

# Create Abstract Model
model = AbstractModel()
    
# Levels
model.horizon = Param(default=horizon, mutable = False)
model.T = RangeSet(0, model.horizon-1)

# Variables
model.h = Var(model.T, domain=NonNegativeReals)
model.s = Var(model.T, domain=NonNegativeReals)

# Constraints
def budget(model):
    costs=0
    for i in model.T:
        costs += model.h[i] * cost['housing']
        costs += model.s[i] * cost['shelter']
    return costs < budget

model.BUDGET=Constraint(rule=budget)

# Objective function
def obj_expression(model):
    fluid_model = fl.FluidFlowModel(data_as_is, model.h, model.s)
    T = [i/365 for i in range(int(model.horizon*365))]
    fluid_model.analyse(T)     
    output = sum(fluid_model.unsh_t)/len(fluid_model.unsh_t)
    return output

model.OBJ = Objective(rule=obj_expression)

# Solve
opt=SolverFactory('glpk')
instance=model.create_instance()
instance.preprocess()
results=opt.solve(instance)

# Optimal sol
h_opt=np.zeros(horizon)
g_opt=np.zeros(horizon)
for i in range(horizon):
    h_opt[i]=instance.h[i].value
    s_opt[i]=instance.s[i].value

# Outputs
print('House building solution per year: ' + str(h_opt))
print('Objective Val: ' + str(instance.OBJ()))
