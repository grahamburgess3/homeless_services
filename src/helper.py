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
