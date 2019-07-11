import numpy as np
import os

#   Computes the length of a given permutation (perm) as a route for TSP in the Graph
def computeTourLength(perm, Graph):
    tlen = 0.0
    for i in range(len(perm)):
        tlen += Graph[perm[i], perm[np.mod(i + 1, len(perm))]]  # assures the route is consistent
    return tlen

"""
Variate gets a permutation (perm) and changes the order of 4 random cities in perm and returns the new permutation  
This is my Definition for a neighbour of a permutation  
"""
def variate(perm):
    tmp = np.copy(perm)

    for _ in range(2):
        i, j = np.random.choice(len(perm), 2)  # Choice function returns two different indexes from [0,130)
        tmp[i], tmp[j] = tmp[j], tmp[i]  # Swapping two cities in the permutation
    return tmp


if __name__ == "__main__":
    dirname = ""
    fname = os.path.join(dirname, "hachula130.dat")
    data = []
    NTrials = 5000  # Number of trials to check
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
    #
    tourStat = []
    perm = np.random.permutation(n)  # A random permutation
    length_of_first = computeTourLength(perm, G)
    sum = 0
    #  In the loop, creating 5000 neighbours of perm and adding the differences between their length minus perms length
    for k in range(NTrials):
        tmp_perm = variate(perm)
        sum += computeTourLength(tmp_perm, G) - length_of_first
    print(abs(sum / NTrials))   # Dividing by Ntrials to get the average difference between perm and his neighbours
