"""
@authors: Raviv Trichter & Ido Dan
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb

def GA(n, max_evals, selectfct, fitnessfct,crossoverfct,mutationfct, seed=None) :
    eval_cntr = 0
    history = []
    max_attainable = comb(n,2)
    #
    #GA params
    population = 100
    pc = 0.37
    pm = 2/n
    local_state = np.random.RandomState(seed)
    Genome = []
    for _ in range(population): # inserting random permutations into Genome as the size of a population
        Genome.append(np.random.permutation(n))
    fitness, calls_to_calc = fitnessfct(Genome,max_attainable,True)
    eval_cntr += calls_to_calc
    fcurr_best = fmax = np.max(fitness)
    xmax = Genome[np.argmax(fitness)]
    history.append(fmax)
    how_far = 100000 # a variable for printing every 100,000 iterations
    while (eval_cntr < max_evals) :
#       Generate offspring population (recombination, mutation)
        newGenome = []
#        1. sexual selection + 1-point recombination
        for i in range(int(population/2)) :
            p1 = selectfct(Genome,fitness,local_state)
            p2 = selectfct(Genome,fitness,local_state)
            if local_state.uniform() < pc : #recombination (Crossover)
                Xnew1, Xnew2 = crossoverfct(p1,p2)
            else : #no recombination; two parents are copied as are
                Xnew1 = np.copy(p1)
                Xnew2 = np.copy(p2)
#        2. mutation
            if local_state.uniform()< pm:
                Xnew1 = mutationfct(Xnew1)
            if local_state.uniform() < pm:
                Xnew2 = mutationfct(Xnew2)
#
            newGenome.append(Xnew1)
            newGenome.append(Xnew2)
        #The best individual of the parental population is kept
        newGenome.append(np.copy(Genome[np.argmax(fitness)]))
        Genome = np.copy(newGenome)
        fitness, calls_to_calc = fitnessfct(Genome,max_attainable)
        eval_cntr += calls_to_calc
        fcurr_best = np.max(fitness)
        if fmax < fcurr_best :
            fmax = fcurr_best
            xmax = Genome[np.argmax(fitness)]
        history.append(fcurr_best)
        if eval_cntr > how_far :
            print(how_far," evals: fmax=",fmax) # printing approximatly every 10**5 iterations
            how_far += 100000
        if fmax == max_attainable :
            print(eval_cntr," evals: fmax=",fmax,"; done!")
            break
    return xmax, fmax, history, eval_cntr


# Suming the right diagonal as the equality :  Phenotype[i] + i for all i
# Suming the left diagonal as the equality :  Phenotype[i] - i for all i
def calculateFitness(Phenotype, isDiagRight):
    diag = []
    for i in range(len(Phenotype)):
        if isDiagRight:
            diag.append(Phenotype[i] + i)
        else:
            diag.append(Phenotype[i] - i)
    diag.sort()
    left = cnt = 0
    right = 1
    while right < len(diag):
        if diag[left] != diag[right]:
            left += 1
            right += 1
        else:
            while right < len(diag) and diag[left] == diag[right]:
                right += 1
            cnt += comb (right - left,2) # Gives us exactly the number of pairs (Combinatorial)
            left = right
            if right != len(diag) - 1:
                right += 1
    return cnt

# Getting a matrix -> Genome. evaluating every row in the matrix (individual)
# Returning a vector that each element i in the vector describes the fitness of Genome[i]
def fitnessFunction(Genome,Max_Fitness,printOPT = False):
    fitness = []
    if printOPT:
        print("The Optimum is : ", Max_Fitness)
    remmember_dict = dict()
    eval_cnt = 0
    for gene in Genome:
        # avoiding calculating the fittens more than once for the same permutation
        if tuple(gene) not in remmember_dict.keys():
            fit = Max_Fitness - calculateFitness(gene,True) - calculateFitness(gene,False) # Objective function
            eval_cnt += 1
            fitness.append(fit)
            remmember_dict[tuple(gene)] = fit
        else:
            fitness.append(remmember_dict[tuple(gene)])
          
    return np.copy(fitness) , eval_cnt

def OurMutation(Gene):
    i, j = np.random.choice(len(Gene), 2, replace=False) # swapping 2 different indexes
    Gene[i], Gene[j] = Gene[j],Gene[i]
    return np.copy(Gene)


def OurRecombination(Parent1, Parent2) :
    big, small = sorted(np.random.choice(len(Parent1),2,replace=False))
    child1 = [-1]*len(Parent1)
    child2 = np.copy(child1)
    child1[small:big] = Parent1[small:big]
    child2[small:big] = Parent2[small:big]
    missing_child1 = [x for x in Parent2 if x not in child1]
    missing_child2 = [x for x in Parent1 if x not in child2]
    j = 0
    for i in range(len(child1)):
        if j < len(missing_child1) and child1[i] == -1:
            child1[i] = missing_child1[j]
            j += 1
    j = 0
    for i in range(len(child2)):
        if j < len(missing_child2) and child2[i] == -1:
            child2[i] = missing_child2[j]
            j += 1
    return  child1,child2

#Roulette
def select_proportional(Genome,fitness,rand_state) :
    ''' RWS: Select one individual out of a population Genome with fitness values fitness using proportional selection.''' 
    cumsum_f = np.cumsum(fitness)
    r = sum(fitness) * rand_state.uniform()
    idx = np.ravel(np.where(r<cumsum_f))[0]
    return Genome[idx]



    
    
if __name__ == "__main__" :
    n=64
    evals=10**6
    Nruns= 1
    fbest = []
    xbest = []
    for i in range(Nruns) :
        xmax,fmax,history, eval_cnt = GA(n,evals,selectfct=select_proportional,fitnessfct=fitnessFunction,
                               crossoverfct=OurRecombination,mutationfct=OurMutation)
        plt.semilogy(np.array(history))
        plt.show()
        print("maximal N-Queen Problem found ", fmax," at location \n", xmax.T)
        fbest.append(fmax)
        xbest.append(xmax)
    print("====\n Best ever: ",max(fbest),"x*=",xbest[fbest.index(max(fbest))].T)