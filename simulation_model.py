import simpy
import random
import math
import collections
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

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

def gen_arrivals(env, accommodation_stock, service_mean, arrival_rates, initial_demand, warm_up_time):
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

        """
        # wait for warm up (while initial building taking place)
        yield env.timeout(warm_up_time)
        
        # generate arrivals for those initially in system (current demand)
        for i in range(initial_demand):
            c = Customer()
            env.process(process_find_accommodation(env, c, accommodation_stock, service_mean, warm_up_time))

        # generate arrivals with non-homogeneous Poisson process using 'thinning'
        arrival_rate_max = max(arrival_rates)
        while True:
            arrival_rate = get_arrival_rate(arrival_rates, env.now-warm_up_time)
            U = random.uniform(0,1)
            t = random.expovariate(arrival_rate_max)
            yield env.timeout(t)
            if U <= arrival_rate / arrival_rate_max:
                    c = Customer()
                    env.process(process_find_accommodation(env, c, accommodation_stock, service_mean, warm_up_time))
        
def process_find_accommodation(env, c, accommodation_stock, service_mean, warm_up_time):
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
                if service_mean[accomm_type_next] > 0:
                        time_in_accomm = random.expovariate(1/service_mean[accomm_type_next])
                else:
                        time_in_accomm = 0
                yield env.timeout(time_in_accomm)

                # Finally, leave housing
                accommodation_stock.store.put(housing)

                # Re enter system?
                if c.rand_re_entry >= c.prob_re_entry:
                        c.proceed = False
                else:
                        c.rand_re_entry = random.uniform(0,1)
                        

def get_new_accommodation(build_rates, leftover, build_time, change_time, t_end):
        """
        returns the number of accommodation units (shelter or housing) to be built at time t_end, given the building rate between (t_end - change_time) and t_end, which may have changed in that time period, and given the 'leftover' from the previous build (which arises when a non-integer build rate leads to a non-integer number of houses being built)

        Parameters
        ----------
        build_rates : dict(list)
            for each accommodation type, the number of accommodation units to be built over the course of the time period (in years) given by 'change_time'
        leftover : float
            the non-integer amount of accommodation units leftover from the previous building time
        build_time : float
            the time, in years, between building new accommodation units
        change_time : float
            the time, in years, at which point the building rate may be changed
        t_end : float
           this is the simulation time (in years) at which building is taking place. 

        Returns
        -------
        buildings : float
           the number of buildings to be built at time t_end

        """
        # initialise
        t_start = t_end - build_time
        t = t_start
        buildings = leftover

        # integrate the build rate function over the time window t_start to t_end
        # given the build rate function is piecewise constant, do this integration in chunks, ending each chunk either when the build rate changes or when we reach t_end
        while t < t_end:
                # get the build rate
                way_through = t/change_time # how far through the build rate function we are at time t (as a proportion of the time between changing the build rate)
                build_rate = build_rates[math.floor(way_through)] # the build rate at time t

                # get the time until t_end (as proportion of change_time)
                time_to_tend = (t_end - t)/change_time

                # get the time until the build rate may change (as a proportion of change_time)
                if way_through % 1 > 0:
                        time_to_next_rate = (math.ceil(way_through) - way_through)
                elif way_through % 1 == 0:
                        time_to_next_rate = 1

                # get the time for this chunk of integration
                time_to_build_at_this_rate = min(time_to_tend, time_to_next_rate)

                # integration
                buildings += build_rate * time_to_build_at_this_rate

                # advance time
                t += time_to_build_at_this_rate * change_time
                
        return buildings

def gen_development_sched(env, accommodation_stock, accomm_build_time, time_btwn_build_rate_changes, build_rates, warm_up_time):
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
        time_btwn_build_rate_changes : float
           the time (in years) in between points where the build_rate may change
        build_rates : dict(list)
           the build rates for each type of accommodation. The time difference between each build rate is given by time_btwn_build_rate_changes
        warm_up_time : float
           building time before new arrivals enter system

        """
        # initialise
        leftover = {'shelter' : 0, 'housing' : 0}

        # advance through warm up time
        yield env.timeout(warm_up_time)

        # continuously build accommodation
        while True:
                # collect data
                accommodation_stock.data_queue_shelter.append(accommodation_stock.store.queue['shelter'])
                accommodation_stock.data_queue_housing.append(accommodation_stock.store.queue['housing'])
                
                # build accommodation
                for accomm_type in ['shelter','housing']:
                        # compute the number of units to build at this time (based on the build rate function since we last built)
                        new_buildings = get_new_accommodation(build_rates[accomm_type], leftover[accomm_type], accomm_build_time, time_btwn_build_rate_changes, env.now)

                        # either add or remove, depending on sign of new_buildings
                        if (new_buildings > 0):
                                new_buildings_integer = math.floor(new_buildings)
                                leftover[accomm_type] = new_buildings - new_buildings_integer # keep track of leftover
                                for i in range(new_buildings_integer):
                                        accommodation_stock.add_accommodation(accomm_type)
                        elif (new_buildings < 0):
                                new_buildings_integer = math.ceil(new_buildings)
                                leftover[accomm_type] = new_buildings - new_buildings_integer # keep track of leftover
                                for i in range(abs(new_buildings_integer)):
                                        env.process(accommodation_stock.remove_accommodation(accomm_type))
                                        
                # advance time
                yield env.timeout(accomm_build_time)

def simulate(end_of_simulation,
             number_reps,
             accomm_build_time,
             time_btwn_build_rate_changes,
             capacity_initial,
             service_mean,
             arrival_rates,
             build_rates,
             initial_demand,
             warm_up_time,
             prob_re_entry):
        """
        Given a set of inputs and a random seed, simulate the system multiple times over a fixed period of simulation time

        Parameters
        ----------
        end_of_simulation : float
           the simulation time (in years) at which to stop simulating
        number_reps : int
           the number of simulations replications to perform
        accomm_build_time : float
           the time (in years) in between building new accommodation
        time_btwn_build_rate_changes : float
           the time (in years) in between points where the build_rate may change
        capacity_initial : dict(int)
           the initial amount of accommodation units to place in the store
        service_mean : dict(float)
           the mean service time for stays in different types of accommodation
        arrival_rates : list
           annual customer arrival rates
        build_rates : dict(list)
           the build rates for each type of accommodation. The time difference between each build rate is given by time_btwn_build_rate_changes
        initial_demand : dict(int)
           the number of customers to exist in the environment at time t = 0
        warm_up_time : float
           building time before new arrivals enter system
        prob_re_entry : float
           the chance that a customer will re-enter the system once it has left housing
           
        Returns
        -------
        results : np.array
           the size of the unsheltered queue at discrete time points, for each simulation run.         
        timetaken : datetime.timedelta
           the time taken for all the simulation replications to be run. 

        """
        Customer.prob_re_entry = prob_re_entry
        results = {'unsheltered_q_over_time' : [], 'unsheltered_q_avg' : [], 'time_taken' : 0}
        start = datetime.now()
        for rep  in range(number_reps):
                env = simpy.Environment()
                accommodation_stock = AccommodationStock(env, capacity_initial)
                env.process(gen_arrivals(env, accommodation_stock, service_mean, arrival_rates, initial_demand, warm_up_time))
                env.process(gen_development_sched(env, accommodation_stock, accomm_build_time, time_btwn_build_rate_changes, build_rates, warm_up_time))
                env.run(until=end_of_simulation)
                results['unsheltered_q_over_time'].append(np.array(pd.concat([pd.Series([initial_demand - capacity_initial['shelter']-capacity_initial['housing']]), pd.Series(accommodation_stock.data_queue_shelter[1:])])))
                for accomm_type in ['housing', 'shelter']:
                        accommodation_stock.data_queue_avg[accomm_type]['running_avg'] += accommodation_stock.store.queue[accomm_type] * (end_of_simulation - warm_up_time - accommodation_stock.data_queue_avg[accomm_type]['time_last_updated'])
                        accommodation_stock.data_queue_avg[accomm_type]['running_avg'] = accommodation_stock.data_queue_avg[accomm_type]['running_avg'] / (end_of_simulation - warm_up_time)
                results['unsheltered_q_avg'].append(accommodation_stock.data_queue_avg['shelter']['running_avg'])
        end = datetime.now()
        results['unsheltered_q_over_time'] = np.array(results['unsheltered_q_over_time']).T
        results['time_taken'] = end-start
        return(results)

def simulate_as_is(build_rates, data_as_is, data_as_is_simulation):
        """ One replication with as-is variables apart from build rates"""
        number_reps = 1

        output = simulate(data_as_is['analysis_horizon'] + data_as_is_simulation['initial_build_time'],
                          number_reps, 
                          data_as_is['time_btwn_building'], 
                          data_as_is['time_btwn_changes_in_build_rate'], 
                          data_as_is['initial_capacity'], 
                          data_as_is['service_mean'], 
                          data_as_is['arrival_rates'], 
                          build_rates, 
                          data_as_is['initial_demand'], 
                          data_as_is_simulation['initial_build_time'],
                          data_as_is_simulation['reentry_rate'])
    
        output = output['unsheltered_q_avg'][0]
    
        return output

def create_fanchart(arr):
        """
        create a fan chart using an array of arrays

        Parameters
        ----------
        arr : np.array(np.array)
        simulation data over time for multiple simulation runs
        
        Returns
        -------
        fix, ax : graph object
        
        """
        # initialise
        percentiles = (60, 70, 80, 90)
        fig, ax = plt.subplots()

        # x - axis
        x=np.arange(arr.shape[0])*63/365 # every 9 weeks

        # y - axis
        for p in percentiles:
                low = np.percentile(arr, 100-p, axis=1)
                high = np.percentile(arr, p, axis=1)
                alpha = (100 - p) / 100
                ax.fill_between(x, low, high, color='green', alpha=alpha)

        # labels
        plt.xlabel('Years')
        plt.ylabel('# Unsheltered')
        plt.title('Number of unsheltered people - SimPy simulation model results')

        # legend
        first_legend = plt.legend([f'Simulation: {100-p}th - {p}th percentile' for p in percentiles])
        ax.add_artist(first_legend)
    
        return fig, ax

def compare_cdf(data_simpy, data_simio, yr):
    """
    Function to display a chart comparing CDFs of the analytical queuing model output for #unsheltered with ecdf
    
    Parameters
    ----------
    data_simpy : nparray
        simulation output from simpy
    data_simio : nparray
        simulation output from simio
    yr : int
        the year to look at.       

    Returns
    -------
    fig, ax : plot objects
    
    """
    
    # simulation data simpy
    x1 = np.sort(data_simpy[yr*6])
    simpy_out = np.arange(len(x1))/float(len(x1))

    # simulation data simio
    x2 = np.sort(data_simio[yr*6])
    simio_out = np.arange(len(x2))/float(len(x2))
    
    fig, ax = plt.subplots()
    line1, = ax.plot(x1, simpy_out, color = 'black', linewidth = 1)
    line2, = ax.plot(x2, simio_out, color = 'blue', linewidth = 1)
    plt.xlabel('# Unsheltered')
    plt.ylabel('Probability')
    plt.title('CDF for number unsheltered after year ' +  str(yr))
    first_legend = plt.legend(['SimPy model','Simio model'])
    ax.add_artist(first_legend)
    
    return fig, ax

def plot_hist(data, num_bins, xlab, ylab):
        plt.hist(data, num_bins)
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        return plt
