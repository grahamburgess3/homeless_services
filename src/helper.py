import numpy as np

def get_daily_capacity(T_b, init, solution):
    """
    Get Daily housing/shelter capacity (from annual capacity)
        
    Parameters
    ----------
    T_b : int : extra years to model after decision horizon
    init : int : initial capacity
    solution : list : annual capacity (housing or shelter)

    Returns
    -------
    daily : list : daily capacity (housing or shelter)

    """
    annual = [init] + list(solution)
    diffs_annual = list(np.diff(annual))
    diffs_daily = [x/365 for x in diffs_annual]
    diffs = np.repeat(diffs_daily,365)
    daily = [init + sum(diffs[0:i]) for i in range(1,len(diffs)+1)] + [list(solution)[-1]]*(T_b*365)
    return daily

# triangular inverse cdf
def ticdf(u,c):
    if u<c:
        return(np.sqrt(c*u))
    else:
        return(1-np.sqrt((1-u)*(1-c)))

# now suppose we don't
def sample_remaining_time(rng,c):
    u0 = rng.uniform(0,1)
    U = rng.uniform(u0,1)
    X = ticdf(U,c)
    return(X)
