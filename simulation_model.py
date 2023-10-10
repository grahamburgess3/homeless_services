import simpy
import random
import math
import collections
import numpy as np
import matplotlib.pyplot as plt

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


def get_new_buildings(build_rates, leftover, build_freq, change_freq, t_end):
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
        t_start = t_end - build_freq
        t = t_start
        buildings = leftover

        while t < t_end:
                print('START: ' + str(t_start) + ' ' + str(t) + ' ' + str(t_end))
                way_through = t/change_freq
                build_rate = build_rates[math.floor(way_through)]
                time_to_tend = t_end - t
                if way_through % 1 > 0:
                        time_to_next_rate = (math.ceil(way_through) - way_through)*change_freq
                elif way_through % 1 == 0:
                        time_to_next_rate = change_freq
                time_to_build_at_this_rate = min(time_to_tend, time_to_next_rate) 
                buildings += build_rate * time_to_build_at_this_rate
                t += time_to_build_at_this_rate
                print('END: ' + str(t_start) + ' ' + str(t) + ' ' + str(t_end))
                
        return buildings

class HousingFilterStore(simpy.FilterStore):
        def __init__(self, *args, **kwargs):
                self.queue = {'housing' : 0, 'shelter' : 0}
                super().__init__(*args, **kwargs)

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

    def __init__(self, env, initial_stock):

        self.env = env
        self.houses = HousingFilterStore(env)
#        self.houses = simpy.FilterStore(env)
        self.houses.items = [Accommodation('housing') for i in range(initial_stock['housing'])] + [Accommodation('shelter') for j in range(initial_stock['shelter'])]
        print(str(self.houses.items))
        for i in range(4):
                print(str(self.houses.items[i].type))
        self.data_queue = []
        self.data_queue_shelter = []
        self.data_queue_housing = []
 
    def add_house(self, type):
            house = Accommodation(type)
            self.houses.put(house)
            
    def remove_house(self, type):
        if len(self.houses.items) > 0:
            house = yield self.houses.get(filter = lambda house: house.type == type)
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
        print('get queue is ' + str(len(housing_stock.houses.get_queue)) + str(type(housing_stock.houses.get_queue)))
        housing_stock.houses.queue[accomm_type]+=1
        shelter = yield housing_stock.houses.get(filter = lambda house: house.type == accomm_type)
        housing_stock.houses.queue[accomm_type]-=1
        
        print('customer ' + str(c.id) + ' enters ' +  str(accomm_type) + ' at time ' + str(env.now)) 
        if service_mean[accomm_type] > 0:
            time_in_housing = random.expovariate(1/service_mean[accomm_type])
        else:
            time_in_housing = 0
        yield env.timeout(time_in_housing)

        # When done in shelter (but before leaving shelter) look for housing
        accomm_type1 = 'housing'        
        print('customer ' + str(c.id) + ' looks for ' +  str(accomm_type1) + ' at time ' + str(env.now))

        housing_stock.houses.queue[accomm_type1] += 1
        housing = yield housing_stock.houses.get(filter = lambda house: house.type == accomm_type1)
        housing_stock.houses.queue[accomm_type1] -= 1

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
        
def development_sched(env, housing_stock, housing_build_frequency, build_rate_change_frequency, build_rates):
    leftover = {'shelter' : 0, 'housing' : 0}
    while True:
        housing_stock.data_queue.append(len(housing_stock.houses.get_queue)) # this includes sheltered Queue
        housing_stock.data_queue_shelter.append(housing_stock.houses.queue['shelter'])
        housing_stock.data_queue_housing.append(housing_stock.houses.queue['housing'])
        t = housing_build_frequency
        yield env.timeout(t)
        for accomm_type in ['shelter','housing']:
                new_buildings = get_new_buildings(build_rates[accomm_type], leftover[accomm_type], housing_build_frequency, build_rate_change_frequency, env.now)
                if (new_buildings > 0):
                        new_buildings_integer = math.floor(new_buildings)
                        leftover[accomm_type] = new_buildings - new_buildings_integer                        
                        for _ in range(new_buildings_integer):
                                housing_stock.add_house(accomm_type)
                elif (new_buildings < 0):
                        new_buildings_integer = math.ceil(new_buildings)
                        leftover[accomm_type] = new_buildings - new_buildings_integer
                        for _ in range(abs(new_buildings_integer)):
                                env.process(housing_stock.remove_house(accomm_type))

def simulate(end_of_simulation, number_reps, housing_build_frequency, build_rate_change_frequency, capacity_initial, service_mean, arrival_rates, build_rates, current_demand, seed):
        results = []
        random.seed(seed)
        for rep in range(number_reps):
                env = simpy.Environment()
                housing_stock = HousingStock(env, capacity_initial)
                env.process(gen_arrivals(env, housing_stock, service_mean, arrival_rates, current_demand))
                env.process(development_sched(env, housing_stock, housing_build_frequency, build_rate_change_frequency, build_rates))
                env.run(until=end_of_simulation)
                results.append(np.array(housing_stock.data_queue_shelter))
        results = np.array(results).T
        return(results)

def create_fanchart(arr):
    """
    create a fan chart using an array of arrays and a line

    Parameters
    ----------
    arr : np.array(np.array)
        simulation data over time for multiple simulation runs

    Returns
    -------
    fix, ax : graph object

    """
    x=np.arange(arr.shape[0])*63/365 # every 9 weeks
    percentiles = (60, 70, 80, 90)
    fig, ax = plt.subplots()
    for p in percentiles:
        low = np.percentile(arr, 100-p, axis=1)
        high = np.percentile(arr, p, axis=1)
        alpha = (100 - p) / 100
        ax.fill_between(x, low, high, color='green', alpha=alpha)
    plt.xlabel('Years')
    plt.ylabel('# Unsheltered')
    plt.title('Number of unsheltered people - simulation and queueing model results')
    
    first_legend = plt.legend([f'Simulation: {100-p}th - {p}th percentile' for p in percentiles])
    ax.add_artist(first_legend)
    
    return fig, ax
