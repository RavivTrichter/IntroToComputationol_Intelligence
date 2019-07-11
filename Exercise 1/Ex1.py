# -*- coding: utf-8 -*-
"""
Raviv Trichter  204312086
"""
import numpy as np
import os
import matplotlib.pyplot as plt
import SimulatedAnnealing as SA

"""
Computes the length of a given route by summing the costs on each edge in the graph
This function assures this is a valid route
"""
def computeTourLength(perm, Graph):
    tlen = 0.0
    for i in range(len(perm)):
        tlen += Graph[perm[i], perm[np.mod(i + 1, len(perm))]]
    return tlen

"""
Variate - my definiton of a Neighnour to a given permutation s
s' will be a neighbour of s if there are 4 cities that are different in the route
It means that exactly 2 swaps happened
"""
def variate(perm):
    tmp = np.copy(perm)
    num_of_swaps = 2

    for _ in range(num_of_swaps):
        i, j = np.random.choice(len(perm), 2)  # Returns two different indexes between [0,130)
        tmp[i], tmp[j] = tmp[j], tmp[i]  # Swapping two cities in the permutation
    return tmp


if __name__ == "__main__":
    dirname = ""
    fname = os.path.join(dirname, "hachula130.dat")
    data = []
    NTrials = 10**6

    with open(fname) as f:
        for line in f:
            data.append(line.split())
    n = len(data)
    G = np.empty([n, n])
    for i in range(n):
        for j in range(i, n):
            G[i, j] = np.linalg.norm(
                np.array([float(data[i][1]), float(data[i][2])]) - np.array([float(data[j][1]), float(data[j][2])]))
            G[j, i] = G[i, j]

    tourStat = []
    best_tourStat = []
    f_best = 0
    x_best = []
    num_call = 0
    num_try = 0
    calls_to_function = 1 # I initialized to 30 and took the best one
    #  in the following loop, we will call 30 times the SA with different random permutations and save the best solution
    for i in range(calls_to_function):
        perm_x = np.random.permutation(n)  # create permutation
        perm_f = computeTourLength(perm_x, G)  # compute it's length
        print("Length of starting route (first randomized permutation) :", perm_f)
        if i == 0:  # first time initialize f_best
            f_best = perm_f
        tourStat.append(perm_f)
        x_min, f_min, history, call = SA.basicSimulatedAnnealing(perm_x, G, NTrials)
        tourStat.extend(np.copy(history))  # appending to tourStat list the function results of SA
        history.clear()
        if f_min < f_best:  # if we got a better solution keep it and it's attributes
            f_best = f_min
            x_best = np.copy(x_min)
            num_call = call
            num_try = i+1
            best_tourStat = np.copy(tourStat)
        tourStat.clear()
    print("Best Run : ", f_best)
    print("Best Route : ", x_best)
    print("Number Of Call to Function : ", num_call, "At call No' : ", num_try)
    plt.semilogy(best_tourStat)
    plt.savefig('data.png')
    plt.show()
