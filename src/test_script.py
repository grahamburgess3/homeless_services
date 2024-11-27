import queueing_model as qm
import fluid_flow_model as fl
import simulation_model as sm
import json

# Data
with open('../data/data.json') as json_file:
    data = json.load(json_file)

solution = data['solution']

# Fluid
f = fl.FluidFlowModel(data, solution)
f.analyse()
f.plot()

# Queue
q = qm.Queue(data, solution)
q.analyse()
q.plot()

# DES
s = sm.SimulationModel(data, solution)
s.analyse()
s.plot()
