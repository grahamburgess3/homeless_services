import math

class FluidFlowModel():

    def __init__(self, data, solution):
        """
        Initialise instance of fluid flow model - continuous approximation of M(t)/M/s(t) model
        
        Parameters
        ----------
        data : dict : includes 'initial_demand', 'initial_capacity', 'service_mean', 'arrival_rates'
        solution : dict(list) : includes annual build rates for 'housing' and 'shelter'

        Returns
        -------
        None

        """
        self.n0 = data['initial_demand']
        self.h0 = data['initial_capacity']['housing']
        self.s0 = data['initial_capacity']['shelter']
        self.mu0 = 1/data['service_mean']['housing']
        self.lambda_t = data['arrival_rates']
        self.h_t = solution['housing']
        self.s_t = solution['shelter']
        self.n_t = [] # number in system over time (Expected val)
        self.sh_t = [] # number sheltered over time (Expected val)       
        self.unsh_t = [] # number unsheltered over time (Expected val)
        self.n_var_t = [] # Var[num in system] over time
        self.unsh_var_t = [] # Var[num unsheltered] over time
        self.n_sq_t = [] # E[(num in system)^2] over time
        self.unsh_sq_t = [] # E[(num unsheltered)^2] over time

    def evaluate_queue_size(self, t):
        """
        Evaluate expected number in system, number sheltered and number unsheltered at time t
        
        Parameters
        ----------
        t : float : time (in years) evaluate queue size

        Returns
        -------
        n : float : E[number in system at time t]
        sh : float : E[number sheltered at time t]
        unsh : float : E[number unsheltered at time t]

        """

        # init quantities
        fluid_in = 0
        fluid_out = 0
        houses = self.h0
        shelters = self.s0

        # add complete years
        yrs = math.floor(t) # number of years passed
        for yr in range(yrs):
            fluid_in += self.lambda_t[yr]
            fluid_out += self.mu0 * (self.h0 + sum([self.h_t[i] for i in range(yr)]) + self.h_t[yr]/2)
            houses += self.h_t[yr]
            shelters += self.s_t[yr]
            
        # add fractional year
        fluid_in += (t % 1) * self.lambda_t[yrs]
        fluid_out += (t % 1) * self.mu0 * (self.h0 + sum([self.h_t[i] for i in range(yrs)]) + (t % 1) * self.h_t[yrs]/2)
        houses += (t % 1) * self.h_t[yrs]
        shelters += (t % 1) * self.s_t[yrs]

        # calculate queue lengths (expected values)
        n = max(0, self.n0 + fluid_in - fluid_out)
        sh = min(shelters, max(0, n - houses))
        unsh = max(0, n - houses - shelters)

        # calculate queue lengths (variance)
        n_var = fluid_in + fluid_out
        unsh_var = fluid_in + fluid_out

        # calculate squared queue lengths (expected vals)
        n_sq = n**2 + n_var
        unsh_sq = unsh**2 + unsh_var
        
        # return
        return n, sh, unsh, n_var, unsh_var, n_sq, unsh_sq

    def analyse(self, T):
        """
        Reset and evaluate self.n_t, self.unsh_t, self.sh_t for all times in T
        
        Parameters
        ----------
        T : list[float] : times (in years) to evaluate queue size

        Returns
        -------
        None

        """
        
        # Reset current values for Q lengths
        self.n_t = [] # number in system over time (Expected val)
        self.sh_t = [] # number sheltered over time (Expected val)       
        self.unsh_t = [] # number unsheltered over time (Expected val)
        self.n_var_t = [] # Var[num in system] over time
        self.unsh_var_t = [] # Var[num unsheltered] over time
        self.n_sq_t = [] # E[(num in system)^2] over time
        self.unsh_sq_t = [] # E[(num unsheltered)^2] over time
        
        # Set starting values for Q lengths
        self.n_t.append(self.n0)
        self.sh_t.append(min(self.s0, max(0, self.n0 - self.h0)))
        self.unsh_t.append(max(0,self.n0 - self.h0 - self.s0))
        self.n_var_t.append(0)
        self.unsh_var_t.append(0)
        self.n_sq_t.append(self.n0**2)
        self.unsh_sq_t.append(self.unsh_t[0]**2)

        # Set remaining values for Q lengths, for t in T
        # (not include first element of T which should be t=0 and already accounted for
        for t in T[1:len(T)]:
            n, sh, unsh, n_var, unsh_var, n_sq, unsh_sq = self.evaluate_queue_size(t)
            self.n_t.append(n)
            self.sh_t.append(sh)
            self.unsh_t.append(unsh)
            self.n_var_t.append(n_var)
            self.unsh_var_t.append(unsh_var)
            self.n_sq_t.append(n_sq)
            self.unsh_sq_t.append(unsh_sq)
