import ranking_and_selection as rs

build_rate_options = {'housing' : [25, 50], 'shelter' : [25,50]}
annual_budget = 75
accommodation_budgets = {'housing' : 200, 'shelter' : 200}
simulation_length = 5

sols = rs.generate_solution_space(build_rate_options, annual_budget, accommodation_budgets, simulation_length)

possible_solutions = 2**(simulation_length*2)

for i in range(len(sols)):
    print(sols[i])
print(str(len(sols)) + ' feasible solutions of possible ' + str(possible_solutions))
