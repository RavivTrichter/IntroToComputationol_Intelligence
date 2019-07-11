"""
@author: ofersh@telhai.ac.il
"""

import numpy as np
from Ex1 import computeTourLength, variate

"""
This function will define T if T has got to his minimum and we have more calls for the objective function
This is a Restart to the Temperature and it is affected by the number of iterations we already did,
the T returned is lower as eval_cntr is bigger.
"""
def ReStart(eval_cntr,max_evals):
    print("Eval_cntr : ", eval_cntr)
    if eval_cntr <= max_evals/4:
        return 110
    elif eval_cntr <= max_evals/2:
        return 90
    elif eval_cntr <= max_evals*3/4:
        return 64
    else:
        return 30
"""
:param xmin the starting permutation
:param max_evals is the number of iterations
:param Graph
:param Variate function that changes the given xmin 
:param func computes f(x) - in our case the length of the route
"""

def basicSimulatedAnnealing(xmin, Graph,max_evals, variation=variate, func=computeTourLength):
    T_min = 1e-20
    T_init = 128
    best_call = -1
    eval_cntr = 1
    history = []
    alpha = 0.999  # Decreased alpha from 0.99 to 0.999 so the descent of T will be slower
    f_lower_bound = 6110  # real shortest path in the Graph
    eps_satisfactory = 1e-5
    local_state = np.random.RandomState(seed=None)
    max_internal_runs = 50
    xbest = np.copy(xmin)
    fmin = func(xmin, Graph)
    fbest = fmin
    T = T_init
    history.append(fmin)
    while T > T_min and eval_cntr < max_evals:
        for _ in range(max_internal_runs):
            x = variation(xmin)
            f_x = func(x, Graph)
            eval_cntr += 1
            dE = f_x - fmin
            if dE < 0:  # if we got a better solution
                fbest = f_x
                xbest = np.copy(x)
                best_call = eval_cntr
            if dE <= 0 or local_state.uniform(size=1) < np.exp(-np.abs(dE)/T):  # in a small prob. take a worse solution
                xmin = np.copy(x)
                fmin = f_x
            history.append(fmin)
            if np.mod(eval_cntr, int(max_evals/10)) == 0:  # prints every 10^5 iterations the current fmin
                print(eval_cntr," evals: fmin=", fmin)
            if fbest < f_lower_bound+eps_satisfactory:  # if we got to the optimal solution
                T = T_min
                break
        T *= alpha  # Decreasing T every iteration
        if T <= T_min:
            T = ReStart(eval_cntr,max_evals)  # Restarting the program by re-initializing the Temperature
    return xbest, fbest, history, best_call