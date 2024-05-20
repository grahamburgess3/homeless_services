# Imports
from __future__ import division
import pyomo.environ as pyo
from pyomo.core import Var
from pyomo.opt import SolverFactory

# Internal imports
import fluid_flow_model as fl

class Problem():

    def __init__(self, horizon):

        # Attributes
        self.horizon = horizon
        
        # Create Abstract Model
        self.problem = pyo.AbstractModel()
    
        # Levels
        self.problem.T = pyo.RangeSet(0, horizon-1)
        
        # Variables
        self.problem.h = Var(self.problem.T, domain=pyo.NonNegativeReals)
        self.problem.s = Var(self.problem.T, domain=pyo.NonNegativeReals)

    def solve(self, solver):
        self.opt=SolverFactory(solver)
        self.instance=self.problem.create_instance()
        self.results=self.opt.solve(self.instance)
        self.h_opt=[self.instance.h[i].value for i in range(self.horizon)]
        self.s_opt=[self.instance.s[i].value for i in range(self.horizon)]
        
class Phi(Problem):

    def __init__(self, data, timestep, horizon, budget, costs_accomm, baseline_build, budget_constraint, min_house_build, min_shelter_build, objective_function):
        super(Phi, self).__init__(horizon)

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

    def add_housing_increase(self, h_increase):
        self.problem.T_housing = pyo.RangeSet(0, self.horizon-2)
        self.problem.h_increase = pyo.Constraint(self.problem.T_housing, rule = h_increase)
            
    def add_shelter_increase_decrease(self, shelter_mode, s_increase, s_decrease):
        self.problem.T_shelter_first = pyo.RangeSet(0, shelter_mode-1)
        self.problem.T_shelter_second = pyo.RangeSet(shelter_mode, self.horizon-2)
        self.problem.s_increase = pyo.Constraint(self.problem.T_shelter_first, rule = s_increase)
        self.problem.s_decrease = pyo.Constraint(self.problem.T_shelter_second, rule = s_decrease)

class FluidModel():

    def __init__(self, data, solution):
        self.model = fl.FluidFlowModel(data, solution)
        
    def analyse(self, horizon, timestep):
        self.T = [i*timestep for i in range(int(horizon/timestep))]
        self.model.analyse(self.T)
