import simpy
import random
import math
import collections

# Inputs
housing_capacity_initial = 40
shelter_capacity_initial = 15
housing_build_frequency = 1/6 # in years
end_of_simulation = 5 # in years
service_mean = {'housing' : (1/52)*(0+300+400)/3, 'shelter' : 0.0} # in years
arrival_rates = [35.0400, 42.0048, 46.2528, 46.2528, 41.6100, 37.4052] # expected number per year
build_rates = {'housing' : [3,6,7,10,8,4], 'shelter' : [2,2,0,-2,-1,-1]}
# build_rates = {'housing' : [0,0,0,0,0,0], 'shelter' : [0,0,0,0,0,0]}
current_demand = 120

def get_arrival_rate(arrival_rates, t):
        """
        returns arrival rate at time t

        Parameters
        ----------
        t : float
            time in years.

        Returns
        -------
        arr_rate : arrival rate at time t.

        """
        
        arr_rate = arrival_rates[math.floor(t)]
        
        return arr_rate


def get_build_rate(build_rates, t):
        """
        returns build rate at time t

        Parameters
        ----------
        t : float
            time in years.

        Returns
        -------
        build_rate : build rate at time t.

        """
        
        build_rate = build_rates[math.floor(t)]
        
        return build_rate
    
class Accommodation():
    # tracks the next id to be assigned to a house
    next_id = {'housing' : 1, 'shelter' : 1}

    def __init__(self, type):
        self.id = Accommodation.next_id[type]
        self.type = type
        Accommodation.next_id[type] += 1

class Customer():
    # tracks the next id to be assigned to a customer
    next_id = 1

    def __init__(self):
        self.id = Customer.next_id
        Customer.next_id += 1

class HousingStock():

    def __init__(self, env, initial_stock, initial_stock_shelter):

        self.env = env
        self.houses = simpy.FilterStore(env, capacity = initial_stock + initial_stock_shelter)
        self.houses.items = [Accommodation('housing') for i in range(initial_stock)] + [Accommodation('shelter') for j in range(initial_stock_shelter)]
        print(str(self.houses.items))
        for i in range(4):
                print(str(self.houses.items[i].type))
        self.data_queue = []
 
    def add_house(self, type):
            house = Accommodation(type)
            self.houses.put(house)
            
    def remove_house(self, type):
        if len(self.houses.items) > 0:
            house = yield self.houses.get(lambda house: house.type == type)
        else:
            pass

def gen_arrivals(env, housing_stock, service_mean, arrival_rates, current_demand):
        # generate arrivals for those initially in system (current demand)
        for _ in range(current_demand):
            c = Customer()
            print('customer ' + str(c.id) + ' arrives at time t = ' + str(env.now))
            print('current queue: ' + str(len(housing_stock.houses.get_queue)))            
            env.process(find_housing(env, c, housing_stock, service_mean))
            
        # generate arrivals with non-homogeneous Poisson process using 'thinning'
        arrival_rate_max = max(arrival_rates)
        while True:
            arrival_rate = get_arrival_rate(arrival_rates, env.now)
            U = random.uniform(0,1)
            t = random.expovariate(arrival_rate)
            yield env.timeout(t)
            if U <= arrival_rate / arrival_rate_max:
                    c = Customer()
                    print('customer ' + str(c.id) + ' arrives at time t = ' + str(env.now))
                    print('current queue: ' + str(len(housing_stock.houses.get_queue)))
                    env.process(find_housing(env, c, housing_stock, service_mean))

def find_housing(env, c, housing_stock, service_mean):

        # First look for shelter
        accomm_type = 'shelter'
        print('customer ' + str(c.id) + ' looks for ' +  str(accomm_type) + ' at time ' + str(env.now)) 
        shelter = yield housing_stock.houses.get(lambda house: house.type == accomm_type)
        print('customer ' + str(c.id) + ' enters ' +  str(accomm_type) + ' at time ' + str(env.now)) 
        if service_mean[accomm_type] > 0:
            time_in_housing = random.expovariate(1/service_mean[accomm_type])
        else:
            time_in_housing = 0
        yield env.timeout(time_in_housing)

        # When done in shelter (but before leaving shelter) look for housing
        accomm_type1 = 'housing'        
        print('customer ' + str(c.id) + ' looks for ' +  str(accomm_type1) + ' at time ' + str(env.now))
        housing = yield housing_stock.houses.get(lambda house: house.type == accomm_type1)

        # When found housing, leave shelter and spend time in housing
        housing_stock.houses.put(shelter)
        print('customer ' + str(c.id) + ' leaves ' +  str(accomm_type) + ' at time ' + str(env.now)) 
        print('customer ' + str(c.id) + ' enters ' +  str(accomm_type1) + ' at time ' + str(env.now)) 
        if service_mean[accomm_type1] > 0:
            time_in_housing = random.expovariate(1/service_mean[accomm_type1])
        else:
            time_in_housing = 0
        yield env.timeout(time_in_housing)

        # Finally, leave housing
        housing_stock.houses.put(housing)
        print('customer ' + str(c.id) + ' leaves ' +  str(accomm_type1) + ' at time ' + str(env.now)) 
        
def development_sched(env, housing_stock, housing_build_frequency, build_rates):
    while True:
        housing_stock.data_queue.append(len(housing_stock.houses.get_queue)) # this includes sheltered Queue
        t = housing_build_frequency
        yield env.timeout(t)
        for accomm_type in ['shelter','housing']:
                build_rate = get_build_rate(build_rates[accomm_type], env.now)
                if (build_rate > 0):
                        for _ in range(build_rate):
                                housing_stock.add_house(accomm_type)
                elif (build_rate < 0):
                        for _ in range(abs(build_rate)):
                                env.process(housing_stock.remove_house(accomm_type))
                
# run the model
env = simpy.Environment()
housing_stock = HousingStock(env, housing_capacity_initial, shelter_capacity_initial)
env.process(gen_arrivals(env, housing_stock, service_mean, arrival_rates, current_demand))
env.process(development_sched(env, housing_stock, housing_build_frequency, build_rates))
env.run(until=end_of_simulation)
print(housing_stock.data_queue)
