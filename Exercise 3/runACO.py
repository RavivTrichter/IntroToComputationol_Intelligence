# -*- coding: utf-8 -*-
"""
@author: ofersh@telhai.ac.il
"""
from ACO import AntColony as ACO
import numpy as np
import os

def print_sol(path):
    sol = []
    row = [path[0][0]]
    sum_of_row = path[0][0]
    for arc in path:
        if sum_of_row+arc[1] > 150:
            sol.append(row)
            sum_of_row = int(arc[1])
            row = [arc[1]]
        else:
            row.append(arc[1])
            sum_of_row += int(arc[1])
    if sum_of_row:  # checks if a new tow has started
        sol.append(row)
    return sol



if __name__ == "__main__" :
    Niter = 30
    Nant = 200
    filename1 = "rotterdam.txt"
    print(filename1 + " :")
    ant_colony = ACO(filename1, Nant, Niter, 4, rho=0.95, alpha=1, beta=3)
    shortest_path, fitness = ant_colony.run()
    sol = print_sol(shortest_path)
    print("manged to organize in ", fitness, " rows")
    print(sol)
    print("population = ", Nant,"     num_of_iter = ", Niter)
    print("calculated the fitness ", str(Niter*Nant), " times\n\n")

    filename2 = "roskilde.txt"
    print(filename2 + " :")
    ant_colony = ACO(filename2, Nant, Niter, 4, rho=0.95, alpha=1, beta=3)
    shortest_path, fitness = ant_colony.run()
    sol = print_sol(shortest_path)
    print("manged to organize in ", fitness, " rows")
    print(sol)
    print("population = ", Nant,"     num_of_iter = ", Niter)
    print("calculated the fitness ", str(Niter*Nant), " times\n\n")