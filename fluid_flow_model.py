import math

class FluidFlowModel():

    def __init__(self, n0, h0, s0, mu0, lambda_t, h_t, s_t):
        self.n0 = n0
        self.h0 = h0
        self.s0 = s0
        self.mu0 = mu0
        self.lambda_t = lambda_t
        self.h_t = h_t
        self.s_t = s_t
        self.mu_t = []
        self.n_t = []
        self.unsh_t = []
        self.sh_t = []

    def analyse_model(self, T, dt):
        """
        Evaluate self.n_t, self.unsh_t, self.sh_t from t=0 to t=T in steps dt
        """
        self.n_t.append(self.n0)
        self.unsh_t.append(max(0,self.n0 - self.h0 - self.s0))
        
        for t in range(int(T/dt)-1):
            rate_flow_in = self.lambda_t[math.floor(t*dt)]
            fluid_in = rate_flow_in*dt

            houses = self.h0
            shelters = self.s0
            for yr in range(math.floor(t*dt)):
                houses += self.h_t[yr]
                shelters += self.s_t[yr]
            houses += ((t+1)%(int(1/dt)))*dt*self.h_t[math.floor(t*dt)]
            shelters +=((t+1)%(int(1/dt)))*dt*self.s_t[math.floor(t*dt)]

            rate_flow_out = houses*self.mu0
            fluid_out = rate_flow_out*dt
            self.n_t.append(self.n_t[t] + fluid_in - fluid_out)
            self.unsh_t.append(max(0,self.n_t[t+1] - houses - shelters))
