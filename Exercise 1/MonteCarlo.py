# -*- coding: utf-8 -*-
"""
@author: ofersh@telhai.ac.il
"""
import numpy as np
import os
import matplotlib.pyplot as plt


def computeTourLength(perm, Graph):
    tlen = 0.0
    for i in range(len(perm)):
        tlen += Graph[perm[i], perm[np.mod(i + 1, len(perm))]]
    return tlen


if __name__ == "__main__":
    dirname = ""
    fname = os.path.join(dirname, "hachula130.dat")
    data = []
    NTrials = 10 ** 6
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
    for k in range(NTrials):
        permut = np.random.permutation(n)
        f_perm = computeTourLength(permut, G)
        if k == 0:
            f_min = f_perm
        if f_perm < f_min:
            f_min = f_perm
            x_best = np.copy(permut)
        tourStat.append(f_min)

    print(x_best)
    print("Best Value : ",f_min)
    plt.semilogy(tourStat)
    plt.savefig('MonteCarlo.png')
    plt.show()