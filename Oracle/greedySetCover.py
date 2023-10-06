''' Implements greedy heuristic for IC model [1]

[1] -- Wei Chen et al. Efficient Influence Maximization in Social Networks (Algorithm 1)
'''
__author__ = 'ivanovsergey'

from Tool.priorityQueue import PriorityQueue as PQ
from IC.IC import runICmodel_single_step
import heapq
import random

def greedySetCover(G, k, P):
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
        max_reward = -1
        max_reward_keys = []
        s = {}
        # s = PQ()
        for v in G.nodes():
            if v not in S:
                reward = 0
                for j in range(R): # run R times Random Cascade
                    r, T_new, _, p = runICmodel_single_step(G, v, P, list(set(T).union(set([v]))) )
                    reward += r
                s[v] = [reward/R, T_new, p]

                if reward/R > max_reward:
                    max_reward = reward/R
                    max_reward_keys = [v]
                elif reward/R == max_reward:
                    max_reward_keys.append(v)

                # s.add_task(v, -reward/R, T_new) # add normalized spread value

            # if v == 509912501 and v in s.keys():
            #     print(s[v][2])

        max_reward_keys = sorted(max_reward_keys, key=lambda x: s[x][2], reverse=True)

        # print(max_reward)
        # print(max_reward_keys)
        # print("K", s[max_reward_keys[0]][2])

        task = max_reward_keys[0]
        # task = random.sample(max_reward_keys, 1)[0]
        priority, T, _ = s[task]

        # for v in max_reward_keys:
        #     print(v, s[v][2])

        S.append(task)

        # print("S", S)
        # print("T", T)
    return S