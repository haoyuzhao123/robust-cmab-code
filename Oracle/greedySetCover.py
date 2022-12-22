''' Implements greedy heuristic for IC model [1]

[1] -- Wei Chen et al. Efficient Influence Maximization in Social Networks (Algorithm 1)
'''
__author__ = 'ivanovsergey'

from Tool.priorityQueue import PriorityQueue as PQ
from IC.IC import runICmodel_single_step
import heapq

def greedySetCover(G, k, p):
    ''' Finds initial seed set S using general greedy heuristic
    Input: G -- networkx Graph object
    k -- number of initial nodes needed
    p -- propagation probability
    Output: S -- initial set of k nodes to propagate
    '''
    import time
    start = time.time()
    R = 1 # number of times to run Random Cascade
    S = [] # set of selected nodes
    T = [] # set of activated nodes 
    # add node to S if achieves maximum propagation for current chosen + this node
    for i in range(k):
        s = PQ() # priority queue
        for v in G.nodes():
            if v not in S:
                reward = 0
                for j in range(R): # run R times Random Cascade
                    r, T_new, _ = runICmodel_single_step(G, v, p, T+[v])
                    reward += r
                s.add_task(v, -reward/R, T_new) # add normalized spread value
        task, priority, T = s.pop_item()
        S.append(task)
        # print(i, k, time.time() - start)
    return S