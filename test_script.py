#import ranking_and_selection as rs
import math
import numpy as np

build_rate_options = {'housing' : [25, 50], 'shelter' : [25,50]}
annual_budget = 75
accommodation_budgets = {'housing' : 200, 'shelter' : 200}
simulation_length = 5

#sols = rs.generate_solution_space(build_rate_options, annual_budget, accommodation_budgets, simulation_length)

#possible_solutions = 2**(simulation_length*2)

#for i in range(len(sols)):
#    print(sols[i])
#print(str(len(sols)) + ' feasible solutions of possible ' + str(possible_solutions))

def bigS(x):
    littleS = math.ceil(x/40)
    bigS = littleS + x - (littleS - 1)*40
    return bigS

x = range(1600)
y = []

for i in range(1600):
    y.append(bigS(i))

out = np.array([3,1,2,0])
for i in out:
    print (i)
