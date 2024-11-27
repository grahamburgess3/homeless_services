import simpy
import random
import math
import collections
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import helper as hlp
import pdb

class Customer():
        """
        A class to represent an individual customer arrival to the system

        Attributes (Class)
        ----------
        next_id : dict(int)
           unique id for the next customer to be initialised
        prob_re_entry : float
           the probability that the customer will re-enter the system upon leaving housing (defined in simulate() function)

        Attributes (Instance)
        ----------
        id : int
           unique ID for this type of accommodation
        proceed : bool
           defines whether or not the customer proceeds to look for accommodation
           this always starts as True, and if it remains as True once they leave system, they re-enter system and look for accommodation again
        rand_re_entry : float
           random number between 0 and 1 which dicates whether or not this customer will re-enter the system the next time it leaves the system. 
        
        """
        next_id = 1

        def __init__(self):
                """
                Constructs the initial attributes for an instance of Customer

                """
                self.id = Customer.next_id
                self.proceed = True
                self.rand_re_entry = random.uniform(0,1)
                Customer.next_id += 1 # advance the class attribute accordingly

class Accommodation():
        """
        A class to represent an individual unit of accommodation

        Attributes (Class)
        ----------
        next_id : dict(int)
           unique ids for the next unit of each accommodation type to be initialised

        Attributes (Instance)
        ----------
        id : int
           unique ID for this type of accommodation
        type : str
           type of accommodation unit
        
        """
        next_id = {'housing' : 1, 'shelter' : 1}

        def __init__(self, type):
                """
                Constructs the initial attributes for an instance of Accommodation

                Parameters
                ----------
                type : str
                   the type of accommodation
                """
                self.id = Accommodation.next_id[type]
                self.type = type
                Accommodation.next_id[type] += 1 # advance the class attribute accordingly

class AccommodationFilterStore(simpy.FilterStore):
        """
        A class to represent a set of different types of accommodation units (inherits simpy's built'in FilterStore)
        The items attribute (inherited from simpy.FilterStore) contains details of the accommodation units in this store

        Attributes
        ----------
        queue : dict(int)
           the size of the queue for shelter and housing - this changes as each simulation run progresses

        """
        def __init__(self, *args, **kwargs):
                """
                Constructs the initial attributes for an instance of AccommodationFilterStore

                Parameters
                ----------
                env : simpy.Environment
                   the environment in which to set up an instance of AccommodationFilterStore
                """
                self.queue = {'housing' : 0, 'shelter' : 0}
                super().__init__(*args, **kwargs)

class AccommodationStock():
        """
        A class to represent a stock of accommodation units. The items contained in this stock are stored in an instance of 'AccommodationFilterStore'

        Attributes (Instance)
        ----------
        env : simpy.Environment
           the environment in which this stock lives
        store : AccommodationFilterStore
           the store which contains units of different types of accommodation
        data_queue_shelter : list
           stored data over time: the number of customers waiting for store.get() event (shelter only)
        data_queue_housing : list
           stored data over time: the number of customers waiting for store.get() event (housing only)
        
        """
        def __init__(self, env, initial_stock):
                """
                Constructs the initial attributes for an instance of AccommodationStock

                Parameters
                ----------
                env : simpy.Environment
                   the environment in which to place this stock
                initial_stock : dict(int)
                   the initial amount of accommodation units to place in the store
                """
                self.env = env
                self.store = AccommodationFilterStore(env)
                self.store.items = [Accommodation('housing') for i in range(initial_stock['housing'])] + [Accommodation('shelter') for j in range(initial_stock['shelter'])]
                self.data_queue_shelter = [] # this is appended to at regular intervals
                self.data_queue_housing = [] # this is appended to at regular intervals
                self.data_queue_avg = {'housing':{'running_avg' : 0, 'time_last_updated' : 0},
                                       'shelter':{'running_avg' : 0, 'time_last_updated' : 0}}# this is updated whenever the queue changes

        def update_stats(self, t, accomm_type, up_down):
                """
                Update the queue statistics

                Parameters
                ----------
                t : float
                   time at which the change is being made to queue stats
                accomm_type : str
                   accommodation type in question
                up_down : int (-1 or 1)
                   1 if adding to queue, -1 if removing from queue

                """
                self.data_queue_avg[accomm_type]['running_avg'] += self.store.queue[accomm_type] * (t - self.data_queue_avg[accomm_type]['time_last_updated'])
                self.store.queue[accomm_type] += up_down * 1
                self.data_queue_avg[accomm_type]['time_last_updated'] = t
                
        def add_accommodation(self, type):
                """
                Add an accommodation unit to the store

                Parameters
                ----------
                type : str
                   the type of accommodation unit to add
                """
                accommodation = Accommodation(type)
                self.store.put(accommodation)
            
        def remove_accommodation(self, type):
                """
                Add an accommodation unit to the store

                Parameters
                ----------
                type : str
                   the type of accommodation unit to add
                """
                if len(self.store.items) > 0:
                        accommodation = yield self.store.get(filter = lambda accommodation: accommodation.type == type)
                else:
                        pass

def get_arrival_rate(arrival_rates, t):
        """
        returns arrival rate at time t, given a list of annual arrival rates

        Parameters
        ----------
        t : float
            time in years.
        arrival_rates : list
            annual arrival rates to be simulated

        Returns
        -------
        arr_rate : arrival rate at time t.

        """
        arr_rate = arrival_rates[math.floor(t)]

        return arr_rate

def gen_arrivals(env, accommodation_stock, service_mean, arrival_rates, initial_demand, warm_up_time, initial_capacity, service_dist):
        """
        Generate customer arrivals according to the initial demand and the future arrival rates and for each arrival generate a 'find_accommodation' process. Non-homogeneous Poisson arrivals. 

        Parameters
        ----------
        env : simpy.Environment
           the environment in which to generate processes for the new arrivals
        accommodation_stock : AccommodationStock
           the stock of accommodation where the customer arrivals will go to look for accommodation
        service_mean : dict(float)
           the mean service time for stays in different types of accommodation
        arrival_rates : list
           annual customer arrival rates
        initial_demand : dict(int)
           the number of customers to exist in the environment at time t = 0
        warm_up_time : float
           building time before new arrivals enter system
        initial_capacity : int
           number of houses/shetlers in system initially
        service_dist : dict
           Triangle dist params for housing service time distribution


        """
        # wait for warm up (while initial building taking place)
        yield env.timeout(warm_up_time)
        
        # generate arrivals for those initially in system (current demand)
        for i in range(initial_capacity['housing']):
                c = Customer()
                env.process(process_straight_to_housing(env, c, accommodation_stock, service_mean, warm_up_time, service_dist))
        for i in range(max(initial_demand - initial_capacity['housing'], 0)):
                c = Customer()
                env.process(process_find_accommodation(env, c, accommodation_stock, service_mean, warm_up_time, service_dist))

        # generate arrivals with non-homogeneous Poisson process using 'thinning'
        arrival_rate_max = max(arrival_rates)
        while True:
            arrival_rate = get_arrival_rate(arrival_rates, env.now-warm_up_time)
            U = random.uniform(0,1)
            t = random.expovariate(arrival_rate_max)
            yield env.timeout(t)
            if U <= arrival_rate / arrival_rate_max:
                    c = Customer()
                    env.process(process_find_accommodation(env, c, accommodation_stock, service_mean, warm_up_time, service_dist))
        
def process_straight_to_housing(env, c, accommodation_stock, service_mean, warm_up_time, service_dist):
        """
        Using yield statements and store.get() functions, this process advances the simulation clock until desired accommodation is available. Shelter is not exited until Housing becomes available.

        Parameters
        ----------
        env : simpy.Environment
           the environment in which to generate processes for the new arrivals
        c : Customer
           the customer looking for housing
        accommodation_stock : AccommodationStock
           the stock of accommodation where the customer arrivals will go to look for accommodation
        service_mean : dict(float)
           the mean service time for stays in different types of accommodation
        service_dist : dict
           Triangle dist params for housing service time distribution

        """
        # Look for housing
        accomm_type_next = 'housing'
        accommodation_stock.update_stats(env.now-warm_up_time, accomm_type_next, 1)        
        housing = yield accommodation_stock.store.get(filter = lambda accomm: accomm.type == accomm_type_next)
        accommodation_stock.update_stats(env.now-warm_up_time, accomm_type_next, -1)        

        # When found housing, spend time in housing
        time_in_accomm = np.random.triangular(service_dist['low'], service_dist['mid'], service_dist['high'])
        yield env.timeout(time_in_accomm)

        # Finally, leave housing
        accommodation_stock.store.put(housing)

        # Re enter system?
        if c.rand_re_entry >= c.prob_re_entry:
                c.proceed = False
        else:
                c.rand_re_entry = random.uniform(0,1)
                env.process(process_find_accommodation(env, c, accommodation_stock, service_mean, warm_up_time, service_dist))

def process_find_accommodation(env, c, accommodation_stock, service_mean, warm_up_time, service_dist):
        """
        Using yield statements and store.get() functions, this process advances the simulation clock until desired accommodation is available. Shelter is not exited until Housing becomes available.

        Parameters
        ----------
        env : simpy.Environment
           the environment in which to generate processes for the new arrivals
        c : Customer
           the customer looking for housing
        accommodation_stock : AccommodationStock
           the stock of accommodation where the customer arrivals will go to look for accommodation
        service_mean : dict(float)
           the mean service time for stays in different types of accommodation
        service_dist : dict
           Triangle dist params for housing service time distribution

        """
        while c.proceed == True:
                # First look for shelter
                accomm_type = 'shelter'
                accommodation_stock.update_stats(env.now-warm_up_time, accomm_type, 1)
                shelter = yield accommodation_stock.store.get(filter = lambda accomm: accomm.type == accomm_type)
                accommodation_stock.update_stats(env.now-warm_up_time, accomm_type, -1)
                if service_mean[accomm_type] > 0:
                        time_in_accomm = random.expovariate(1/service_mean[accomm_type])
                else:
                        time_in_accomm = 0
                yield env.timeout(time_in_accomm)

                # When done in shelter (but before leaving shelter) look for housing
                accomm_type_next = 'housing'
                accommodation_stock.update_stats(env.now-warm_up_time, accomm_type_next, 1)        
                housing = yield accommodation_stock.store.get(filter = lambda accomm: accomm.type == accomm_type_next)
                accommodation_stock.update_stats(env.now-warm_up_time, accomm_type_next, -1)        

                # When found housing, leave shelter and spend time in housing
                accommodation_stock.store.put(shelter)
                time_in_accomm = np.random.triangular(service_dist['low'], service_dist['mid'], service_dist['high'])
                yield env.timeout(time_in_accomm)

                # Finally, leave housing
                accommodation_stock.store.put(housing)

                # Re enter system?
                if c.rand_re_entry >= c.prob_re_entry:
                        c.proceed = False
                else:
                        c.rand_re_entry = random.uniform(0,1)
                        
def gen_development_sched(env, accommodation_stock, accomm_build_time, warm_up_time, h, s):
        """
        Using yield statements and store.get() and store.put() functions, this process advances the simulation clock until new accommodation is to be built. 

        Parameters
        ----------
        env : simpy.Environment
           the environment in which to generate processes for the new arrivals
        accommodation_stock : AccommodationStock
           the stock of accommodation where the customer arrivals will go to look for accommodation
        accomm_build_time : float
           the time (in years) in between building new accommodation
        warm_up_time : float
           building time before new arrivals enter 
        h : list
           housing stock over time
        s : list
           shelter stock over time

        """
        # initialise
        new_buildings = {'shelter' : 0, 'housing' : 0}
        
        # advance through warm up time
        yield env.timeout(warm_up_time)

        # continuously build accommodation
        while True:
                # collect data
                accommodation_stock.data_queue_shelter.append(accommodation_stock.store.queue['shelter'])
                accommodation_stock.data_queue_housing.append(accommodation_stock.store.queue['housing'])

                # advance time
                yield env.timeout(accomm_build_time)
                
                # number of servers and shelters at current timestep
                t = int(round((env.now-warm_up_time)*365,0))
                new_buildings['housing'] = math.floor(h[t]) - math.floor(h[t-1])
                new_buildings['shelter'] = math.floor(s[t]) - math.floor(s[t-1])

                # build accommodation
                for accomm_type in ['shelter','housing']:
                        # either add or remove, depending on sign of new_buildings
                        if (new_buildings[accomm_type] > 0):
                                for i in range(new_buildings[accomm_type]):
                                        accommodation_stock.add_accommodation(accomm_type)
                        elif (new_buildings[accomm_type] < 0):
                                for i in range(abs(new_buildings[accomm_type])):
                                        env.process(accommodation_stock.remove_accommodation(accomm_type))

class SimulationModel(object):

        """
        Model object
        """
        
        def __init__(self, data, solution):
                """
                Initialise the model object
                
                Data contains the following:
                end_of_simulation : float
                the simulation time (in years) at which to stop simulating
                number_reps : int
                the number of simulations replications to perform
                accomm_build_time : float
                the time (in years) in between building new accommodation
                capacity_initial : dict(int)
                the initial amount of accommodation units to place in the store
                service_mean : dict(float)
                the mean service time for stays in different types of accommodation
                service_dist : dict
                Triangle dist params for housing service time distribution
                arrival_rates : list
                annual customer arrival rates
                solution : dict(list)
                housing and shelter stock
                initial_demand : dict(int)
                the number of customers to exist in the environment at time t = 0
                warm_up_time : float
                building time before new arrivals enter system
                prob_re_entry : float
                the chance that a customer will re-enter the system once it has left housing
                """
                self.end_of_simulation = data['T_a'] + data['T_b'] + data['simulation_build_time']
                self.number_reps = data['simulation_reps']
                self.accomm_build_time = data['time_btwn_building']
                self.capacity_initial = data['initial_capacity']
                self.service_mean = data['service_mean']
                self.service_dist = data['service_dist_triangle']
                self.arrival_rates = data['arrival_rates']
                self.solution = solution
                self.initial_demand = data['initial_demand']
                self.warm_up_time = data['simulation_build_time']
                self.prob_re_entry = data['reentry_rate']
                self.h = hlp.get_daily_capacity(data['T_b'], data['initial_capacity']['housing'], solution['housing'])
                self.s = hlp.get_daily_capacity(data['T_b'], data['initial_capacity']['shelter'], solution['shelter'])

        def analyse(self, percentile = 90):
                """
                Given a set of inputs and a random seed, simulate the system multiple times over a fixed period of simulation time
                """
                Customer.prob_re_entry = self.prob_re_entry
                self.results = {'unsheltered_q_over_time' : [], 'unsheltered_q_avg' : [], 'time_taken' : 0}
                start = datetime.now()
                for rep in range(self.number_reps):
                        env = simpy.Environment()
                        accommodation_stock = AccommodationStock(env, self.capacity_initial)
                        env.process(gen_arrivals(env, accommodation_stock, self.service_mean, self.arrival_rates, self.initial_demand, self.warm_up_time, self.capacity_initial, self.service_dist))
                        env.process(gen_development_sched(env, accommodation_stock, self.accomm_build_time, self.warm_up_time, self.h, self.s))
                        env.run(until=self.end_of_simulation)
                        self.results['unsheltered_q_over_time'].append(np.array(pd.concat([pd.Series([self.initial_demand - self.capacity_initial['shelter']-self.capacity_initial['housing']]), pd.Series(accommodation_stock.data_queue_shelter[1:])])))
                        for accomm_type in ['housing', 'shelter']:
                                accommodation_stock.data_queue_avg[accomm_type]['running_avg'] += accommodation_stock.store.queue[accomm_type] * (self.end_of_simulation - self.warm_up_time - accommodation_stock.data_queue_avg[accomm_type]['time_last_updated'])
                                accommodation_stock.data_queue_avg[accomm_type]['running_avg'] = accommodation_stock.data_queue_avg[accomm_type]['running_avg'] / (self.end_of_simulation - self.warm_up_time)
                        self.results['unsheltered_q_avg'].append(accommodation_stock.data_queue_avg['shelter']['running_avg'])
                end = datetime.now()
                self.results['unsheltered_q_over_time'] = np.array(self.results['unsheltered_q_over_time']).T
                self.results['time_taken'] = end-start
                self.low = list(np.percentile(self.results['unsheltered_q_over_time'], 100-percentile, axis=1))
                self.median = list(np.percentile(self.results['unsheltered_q_over_time'], 50, axis = 1))
                self.high = list(np.percentile(self.results['unsheltered_q_over_time'], percentile, axis=1))

        def plot(self, percentile = 90):
                """
                create a fan chart using an array of arrays
                
                Parameters
                ----------
                percentile : int : percentile to plot on fan chart
                
                """
                # initialise
                fig, ax = plt.subplots()

                # x - axis
                x = [i/365 for i in range(self.end_of_simulation*365)]

                # y - axis
                alpha = (100 - percentile) / 100
                ax.plot(x, self.h, color = 'green')
                ax.plot(x, self.s, color = 'orange')
                ax.plot(x, self.median, color = 'red')
                ax.fill_between(x, self.low, self.high, color='red', alpha=alpha)
                ax.set(xlabel='t (yrs)', ylabel='Number of people', title='DES model')
                ax.legend(["$h_t$", "$s_t$", "$u_t$"], loc="upper left")
                ax.grid()
                ymax = max(self.h + self.s + high)
                ax.set_ylim(0,ymax*1.05)
                
                # display
                plt.show()
