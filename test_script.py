# imports
import ranking_and_selection as rs
import queueing_model as qm
import math
import numpy as np

# simulation options
number_reps = 1
initial_build_time = 63/365 # 9 weeks in years
end_of_simulation = 5 + initial_build_time - 0.25 # in years
initial_demand = 120
initial_capacity = {'housing' : 40, 'shelter' : 15}
arrival_rates = [35.0400, 42.0048, 46.2528, 46.2528, 41.6100] # in 1/year. One constant rate per year.
service_mean = {'housing' : (1/52)*(0+300+400)/3, 'shelter' : 0.0} # in years

# adjust arrival rates to include re-entries
reentry_rate = 0.17 # the proportion of those leaving accommodation which re-enter the system some time later
arrival_rate_reentries = (initial_capacity['housing']*reentry_rate)/service_mean['housing'] # assuming re-entries from the initial number of servers
arrival_rates = [i+arrival_rate_reentries for i in arrival_rates]#
time_btwn_changes_in_build_rate = (6*63)/365 # in years
time_btwn_building = 63/365 # in years. 63/365 years = 9 weeks.
reentry_rate = 0 # set this to zero now we have accounted for re-entries using an uplift to arrival rate

# solution options
build_rate_options = {'housing' : [25, 50], 'shelter' : [25,50]}
annual_budget = 75
accommodation_budgets = {'housing' : 200, 'shelter' : 200}
simulation_length = 5

# generate sols
sols = rs.generate_solution_space(build_rate_options, annual_budget, accommodation_budgets, simulation_length)
#sols = [{'housing': [3,6,7,10,8,4], 'shelter' : [2,2,0,-2,-1,-1]}]

# additional params for analytical model
max_in_system = 1000
num_annual_buildpoints = 6
build_frequency_weeks = 9
d = 1 # days

# model analytically
outputs = []
for s in range(len(sols)):
    q = qm.queue(arrival_rates, service_mean['housing'], initial_capacity['housing'], initial_capacity['shelter'], sols[s]['housing'], sols[s]['shelter'], initial_demand, max_in_system, num_annual_buildpoints, build_frequency_weeks)
    q.model_dynamics(simulation_length, d)
    outputs.append(q.num_unsheltered_avg)
    print('done ' + str(s))

print(outputs)
