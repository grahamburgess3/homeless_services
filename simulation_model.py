import simpy
import random
import math

# Inputs
housing_capacity_initial = 40
housing_build_frequency = 1/6
end_of_simulation = 5 # in years
housing_service_mean = (1/52)*(0+300+400)/3 # in years
arrival_rates = [35.0400, 42.0048, 46.2528, 46.2528, 41.6100, 37.4052] # expected number per year
build_rates = [3,6,7,10,8,4]

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
    
# Arrivals
def source(env, arrival_rates, housing_service_mean, housing, end_of_simulation):
    i = 0
    while(env.now < end_of_simulation):
        i += 1
        arrival_rate = get_arrival_rate(arrival_rates, env.now)
        t = random.expovariate(arrival_rate)
        yield env.timeout(t)
        c = customer(env,
                     'Customer%02d' % i,
                     housing,
                     housing_service_mean)
        env.process(c)

def customer(env, name, housing, housing_service_mean):
    arrive = env.now
    print('%7.4f %s: Here I am' % (arrive, name))
    
    with housing.request() as req:
        # wait for resource to be availabile
        yield req
        wait = env.now - arrive
        print('%7.4f %s: Waited %6.3f' % (env.now, name, wait))

        # spend time in housing
        time_in_housing = random.expovariate(1/housing_service_mean)
        yield env.timeout(time_in_housing)
        print('%7.4f %s: Waited %6.3f' % (env.now, name, wait))

class House():
    # tracks the next id to be assigned to a house
    next_id = 1

    def __init__(self):
        self.id = House.next_id
        House.next_id += 1

class Customer():
    # tracks the next id to be assigned to a customer
    next_id = 1

    def __init__(self):
        self.id = Customer.next_id
        Customer.next_id += 1

class HousingStock():

    def __init__(self, env, initial_stock):

        self.env = env
        self.houses = simpy.Store(env)
        self.houses.items = [House() for _ in range(initial_stock)]
 
    def add_house(self):
            house = House()
            self.houses.put(house)
            
    def remove_house(self):
        if len(self.houses.items) > 0:
            house = yield self.house.get()
        else:
            pass

    def get(self):
        house_req = self.houses.get()
        return house_req

    def put(self, house):
        self.houses.put(house)

def gen_arrivals(env, houses):
    while True:
        arrival_rate = get_arrival_rate(arrival_rates, env.now)
        t = random.expovariate(arrival_rate)
        yield env.timeout(t)
        c = Customer()
        env.process(find_housing(env, c, houses))

def find_housing(env, c, houses):
    house = yield houses.get()
    time_in_housing = random.expovariate(1/housing_service_mean)
    yield env.timeout(time_in_housing)
    houses.put(house)
    
def development_sched(env, houses):
    while True:
        t = housing_build_frequency
        yield env.timeout(t)
        build_rate = get_build_rate(build_rates, env.now)
        if (build_rate > 0):
            for _ in range(build_rate):
                houses.add_house()
        elif (build_rate < 0):
            for _ in range(abs(build_rate)):
                env.process(houses.remove_house())
                
# run the model
env = simpy.Environment()
houses = HousingStock(env, housing_capacity_initial)
env.process(gen_arrivals(env, houses))
env.process(development_sched(env, houses))
env.run(until=end_of_simulation)
