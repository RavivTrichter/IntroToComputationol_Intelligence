# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 09:42:08 2018

@author: ofersh@telhai.ac.il
Based on code by <github/Akavall>
"""
import numpy as np

MAX_ROW = 150
class AntColony(object) :
    def __init__(self, filename, Nant, Niter, size_of_best_g, rho,alpha=1, beta=1, seed=None):
        self.dic = self.getMapGroups(filename)  # dictionary [ key = group_size, value = num_of_appearances ]
        self.list_of_groups = sorted(self.dic.keys()) # a sorted list of the group_sizes
        self.Nant = Nant
        self.Niter = Niter
        self.rho = rho
        self.size_of_best_g = size_of_best_g  # the proportion of good groups we pick in every iteration
        self.alpha = alpha
        self.beta = beta
        self.pheromone = self.initPheromones()
        self.local_state = np.random.RandomState(seed)

    # this function reads from the input file and returns a dict with all the different sizes and counts how many appearances they have
    # the dictionary key : size of group, value : number of appearances
    def getMapGroups(self, filename):
        data = []
        with open(filename) as f:
            for line in f:
                data.append(line.split())
        n = len(data)
        groups = []
        for i in range(n):
            groups.append(int(data[i][0]))
        sz_multiple = list(set([(x, groups.count(x)) for x in groups])) # returns a lst of a set of groups and their multiple
        my_dict = dict()
        for i in range(len(sz_multiple)):
            my_dict[sz_multiple[i][0]] = sz_multiple[i][1]
        return my_dict

    def run(self) :
        #Book-keeping: best arrangement of groups
        shortest_path = None
        best_path = ("TBD", np.inf)
        for i in range(self.Niter):
            all_paths = self.constructColonyPaths()
            self.depositPheromones(all_paths)
            self.normalize() # normalize the pheromones matrix
            shortest_path = min(all_paths, key=lambda x: x[1])
            print(i+1, ": ", shortest_path[1])
            if shortest_path[1] < best_path[1]:
                best_path = shortest_path
            self.pheromone *= self.rho  #evaporation
        return best_path


    # this function erases all the edges in the graph between groups that thier sum is larger then the bin's size
    def initPheromones(self):
        dic_len = len(self.dic)
        pheromones = np.ones(shape=(dic_len, dic_len)) / (dic_len * 1000)
        for i in range(dic_len):
            for j in range(dic_len):
                if self.list_of_groups[i] + self.list_of_groups[j] > MAX_ROW:
                    pheromones[i][j] = 0
        return pheromones

    def depositPheromones(self, all_paths):
        sorted_paths = sorted(all_paths, key=lambda x: x[1])
        Nsel = int(self.Nant / self.size_of_best_g)  # keeping a small group of good solutions
        curr_bin = []
        curr_bin_sum = 0
        for solution, fitness in sorted_paths[:Nsel]:  # takes only the good solutions
            curr_bin.append(solution[0][0]) # every path is a list of tuples (string)
            curr_bin_sum += solution[0][0]
            """ every arc is a tuple of two groups that are connected in the graph
             when we take arc[1] we are ignoring duplicates - by that we count the sum of bins we have (curr_bin_sum) and
                deposite pheromones by the strengh of the solution  """
            for arc in solution :
                if curr_bin_sum + arc[1] > MAX_ROW:
                    self.depositRow(curr_bin, curr_bin_sum)  # update pheromones to every row together
                    curr_bin = [arc[1]]
                    curr_bin_sum = arc[1]
                else:
                    curr_bin.append(arc[1])
                    curr_bin_sum += arc[1]
            if curr_bin_sum < MAX_ROW :
                self.depositRow(curr_bin, curr_bin_sum)

        # deposit pheromones for all of the row members according to the fitness of the entire solution and according
        # to how good the bin is
    def depositRow(self, bin, bin_sum):
        for i in range(len(bin)):
            for j in range(len(bin)):
                if i != j:
                    x, y = self.list_of_groups.index(bin[i]), self.list_of_groups.index(bin[j])
                    self.pheromone[x][y] += ((bin_sum / MAX_ROW) ** 2)

# counting bins for a given solution
    def fitness(self, solution):
        res = 0
        row = solution[0][0]
        for arc in solution:
            row += arc[1]
            if row == MAX_ROW :
                res += 1
                row = 0
            if row > MAX_ROW:  # means we filled a bin and have to start a new one
                row = arc[1]
                res += 1
        if row:  # means there is still a row to add
            res += 1
        return res

    """
    given the dictionary, inserts the biggest group in a bin and fill according to the pheromones
    and continuing to add groups till the bin is full 
    """
    def constructSolution(self):
        curr_dict = self.dic.copy()
        solution = []
        visited = set()
        prev = max(curr_dict.keys())
        curr_dict[prev] -= 1
        curr_bin = prev
        while len(curr_dict) > 0:
            move = self.nextMove(self.pheromone[self.list_of_groups.index(prev)], visited)
            if (move + curr_bin) > MAX_ROW:
                group = self.find_best_fit(curr_bin, curr_dict)  # returns a list of suitable groups
                for g in group: # fills the bin up in an sub optimal way
                    solution.append((prev, g))
                    prev = g
                    curr_dict[g] -= 1
                    if curr_dict[g] == 0:
                        visited.add(self.list_of_groups.index(g))
                        del curr_dict[g]
                move = self.find_next_max(curr_dict, solution, prev, visited)
                curr_bin = move  # starting a new bin
                prev = move
            elif (move + curr_bin) == MAX_ROW: # optimal
                solution.append((prev, move))
                curr_dict[move] -= 1
                if curr_dict[move] == 0:
                    visited.add(self.list_of_groups.index(move))
                    del curr_dict[move]
                prev = move
                move = self.find_next_max(curr_dict, solution, prev, visited)
                curr_bin = move  # starting a new row
                prev = move
            else:
                solution.append((prev, move))
                curr_bin += move
                curr_dict[move] -= 1
                if curr_dict[move] == 0:
                    visited.add(self.list_of_groups.index(move))
                    del curr_dict[move]
                prev = move
        return solution

    # when we start a new line finds the biggest group that hasn't been chosen yet and update dict
    def find_next_max(self, curr_dict, path, prev, visited):
        if len(curr_dict) > 0:  # avoiding exception
            move = max(curr_dict.keys())
            path.append((prev, move))
            curr_dict[move] -= 1
            if curr_dict[move] == 0:
                visited.add(self.list_of_groups.index(move))
                del curr_dict[move]
            return move
        return 0

    def constructColonyPaths(self):
        all_paths = []
        for i in range(self.Nant):
            path = self.constructSolution()
            # constructing pairs: first is the order of groups, second is the number of rows needed
            all_paths.append((path, self.fitness(path)))
        return all_paths

    def nextMove(self, pheromone, visited):
        pheromone = np.copy(pheromone) #Careful: python passes arguments "by-object"; pheromone is mutable
        pheromone[list(visited)] = 0
        norm_row = pheromone / pheromone.sum()
        move = self.local_state.choice(range(len(self.list_of_groups)), 1, p=norm_row)[0]
        return self.list_of_groups[move]

    def find_best_fit(self, curr_row, curr_dict) :
        fit_group = []
        if MAX_ROW - curr_row > self.list_of_groups[0]:
            return fit_group
        for group in sorted(curr_dict.keys(), reverse=True):
            if curr_row + group <= MAX_ROW:
                fit_group.append(group)
                curr_row += group
        return fit_group

    def normalize(self):
        max_val = self.pheromone.sum()
        for i in range(len(self.pheromone)):
            for j in range(len(self.pheromone)):
                self.pheromone[i][j] /= max_val




