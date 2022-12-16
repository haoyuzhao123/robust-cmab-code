''' Independent cascade model for influence propagation
'''
__author__ = 'ivanovsergey'
from copy import deepcopy
from random import random
import numpy as np

def runICmodel_n (G, v_added, P, T=[]):
    ''' Runs independent cascade model.
    Input: G -- networkx graph object
    v_added -- chosen set of vertices
    p -- propagation probability
    Output: T -- resulted influenced set of vertices (including S)

    S -- set of chosen and observed nodes
    T -- set of all observed nodes so far
    '''
    reward = 0
    if type(v_added).__name__ != 'list':
        S = [v_added]
    else:
        S = deepcopy(v_added)

    if type(T).__name__ == 'list' and len(T) > 0:
        T_new = deepcopy(T)
    else:
        T_new = deepcopy(S)

    E = {}

    for v_new in S:
        for v in G[v_new]: # for neighbors of a selected node                
            # w = G[T[i]][v]['weight'] # count the number of edges between two nodes (weight is ther weight of the edge between the two nodes, always 1)
            if random() <= P[v_new][v]['weight']: # if at least one of edges propagate influence
                # print T[i], 'influences', v
                if v not in T_new: # if this node wasn't previously explored
                    S.append(v)
                    T_new.append(v)
                if (v_new, v) in E:
                    E[(v_new, v)] += 1
                else:
                    E[(v_new, v)] = 1
    reward = len(T_new)

    return reward, T_new, E


def runICmodel_single_step (G, v_added, P, T=[]):
    ''' Runs independent cascade model.
    Input: G -- networkx graph object
    v_added -- chosen set of vertices
    p -- propagation probability
    Output: T -- resulted influenced set of vertices (including S)

    S -- set of chosen and observed nodes
    T -- set of all observed nodes so far
    '''
    reward = 0
    if type(v_added).__name__ != 'list':
        S = [v_added]
    else:
        S = deepcopy(v_added)

    E = {}

    if type(T).__name__ == 'list' and len(T) > 0:
        T_new = deepcopy(T)
    else:
        T_new = deepcopy(S)

    for v_new in S:
        for v in G[v_new]: # for neighbors of a selected node                
            # w = G[T[i]][v]['weight'] # count the number of edges between two nodes (weight is ther weight of the edge between the two nodes, always 1)
            if random() <= P[v_new][v]['weight']: # if at least one of edges propagate influence
                # print T[i], 'influences', v
                if v not in T_new: # if this node wasn't previously explored
                    # S.append(v) ## dont append to S so that this node is not extended
                    T_new.append(v)
                if (v_new, v) in E:
                    E[(v_new, v)] += 1
                else:
                    E[(v_new, v)] = 1
    reward = len(T_new)

    return reward, T_new, E




# def runICmodel (G, S, P):
#     ''' Runs independent cascade model.
#     Input: G -- networkx graph object
#     S -- initial set of vertices
#     p -- propagation probability
#     Output: T -- resulted influenced set of vertices (including S)
#     '''

#     T = deepcopy(S) # copy already selected nodes
#     E = {}

#     # ugly C++ version
#     i = 0
#     while i < len(T):
#         for v in G[T[i]]: # for neighbors of a selected node
#             if v not in T: # if it wasn't selected yet
#                 w = G[T[i]][v]['weight'] # count the number of edges between two nodes
#                 if random() <= 1 - (1-P[T[i]][v]['weight'])**w: # if at least one of edges propagate influence
#                     # print T[i], 'influences', v
#                     T.append(v)
#                     E[(T[i], v)] = 1
#         i += 1

#     return len(T), T, E

# def runIC (G, S, p = .01):
#     ''' Runs independent cascade model.
#     Input: G -- networkx graph object
#     S -- initial set of vertices
#     p -- propagation probability
#     Output: T -- resulted influenced set of vertices (including S)
#     '''
#     from copy import deepcopy
#     from random import random
#     T = deepcopy(S) # copy already selected nodes
#     E = {}

#     # ugly C++ version
#     i = 0
#     while i < len(T):
#         for v in G[T[i]]: # for neighbors of a selected node
#             if v not in T: # if it wasn't selected yet
#                 w = G[T[i]][v]['weight'] # count the number of edges between two nodes
#                 if random() <= 1 - (1-p)**w: # if at least one of edges propagate influence
#                     # print T[i], 'influences', v
#                     T.append(v)
#                     E[(T[i], v)] = 1
#         i += 1

#     # neat pythonic version
#     # legitimate version with dynamically changing list: http://stackoverflow.com/a/15725492/2069858
#     # for u in T: # T may increase size during iterations
#     #     for v in G[u]: # check whether new node v is influenced by chosen node u
#     #         w = G[u][v]['weight']
#     #         if v not in T and random() < 1 - (1-p)**w:
#     #             T.append(v)

#     return T#len(T), T, E

# def runIC2(G, S, p=.01):
#     ''' Runs independent cascade model (finds levels of propagation).
#     Let A0 be S. A_i is defined as activated nodes at ith step by nodes in A_(i-1).
#     We call A_0, A_1, ..., A_i, ..., A_l levels of propagation.
#     Input: G -- networkx graph object
#     S -- initial set of vertices
#     p -- propagation probability
#     Output: T -- resulted influenced set of vertices (including S)
#     '''
#     from copy import deepcopy
#     import random
#     T = deepcopy(S)
#     Acur = deepcopy(S)
#     Anext = []
#     i = 0
#     while Acur:
#         values = dict()
#         for u in Acur:
#             for v in G[u]:
#                 if v not in T:
#                     w = G[u][v]['weight']
#                     if random.random() < 1 - (1-p)**w:
#                         Anext.append((v, u))
#         Acur = [edge[0] for edge in Anext]
#         print(i, Anext)
#         i += 1
#         T.extend(Acur)
#         Anext = []
#     return T
    
# def avgSize(G,S,p,iterations):
#     avg = 0
#     for i in range(iterations):
#         avg += float(len(runIC(G,S,p)))/iterations
#     return avg
