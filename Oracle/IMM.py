#  IMM
#  https://github.com/snowgy/Influence_Maximization/blob/a7df248c01a788e110164888eaef4fb961ac9158/IMP.py#L58

from Tool.priorityQueue import PriorityQueue as PQ # priority queue
from math import *
import math
import random



def logcnk(n, k):
    res = 0
    for i in range(n-k+1, n+1):
        res += log(i)
    for i in range(1, k+1):
        res -= log(i)
    return res


def generate_rr_ic(node, G, P):
    activity_set = list()
    activity_set.append(node)
    activity_nodes = list()
    activity_nodes.append(node)
    while activity_set:
        new_activity_set = list()
        for seed in activity_set:
            for v in G[seed]:
                if v not in activity_nodes:
                    if random.random() < P[seed][v]['weight']:
                        activity_nodes.append(v)
                        new_activity_set.append(v)
        activity_set = new_activity_set
    return activity_nodes


def node_selection(R, k, n):
    Sk = []
    rr_degree = {}
    node_rr_set = {}
    matched_count = 0
    num_RR = len(R)

    for i in range(num_RR):
        rr = R[i]
        for v in rr:
            if v not in rr_degree.keys():
                rr_degree[v] = 1
            else:
                rr_degree[v] += 1
            if v not in node_rr_set.keys():
                node_rr_set[v] = [i]
            else:
                node_rr_set[v].append(i)

    for i in range(k):
        max_node = max(rr_degree, key=rr_degree.get)
        Sk.append(max_node)
        matched_count += rr_degree[max_node]
        index_set = []
        for node_rr in node_rr_set[max_node]:
            index_set.append(node_rr)
        for j in index_set:
            rr = R[j]
            for rr_node in rr:
                rr_degree[rr_node] -= 1
                node_rr_set[rr_node].remove(j)

    return Sk, matched_count/len(R)


def sampling(G, k, p, epsilon=0.5, l=1):
    R = []
    LB = 1
    epsilon_p = epsilon * sqrt(2)
    n = len(G.nodes())
    lR = 0

    for i in range(1, int(log2(n-1))+1):
        x = n/(2 ** i)
        lamda_p = ((2+2*epsilon_p/3)*(logcnk(n, k) + l*log(n) + log(log2(n)))*n)/pow(epsilon_p, 2)
        theta_i = lamda_p/x

        while lR <= theta_i:
            v = random.choice(range(1, n))
            rr_nodes = generate_rr_ic(G.nodes()[v], G, p)
            R.append(rr_nodes)
            lR += 1

        _, f = node_selection(R, k, n)
        if n * f >= (1+epsilon_p) * x:
            LB = n*f/(1 + epsilon_p)
            break

    alpha = sqrt(l*log(n) + log(2))
    beta = sqrt((1-1/math.e)*(logcnk(n, k)+l*log(n)+log(2)))
    lambda_star = 2*n*pow(((1-1/math.e)*alpha + beta), 2)*pow(epsilon, -2)
    theta = lambda_star/LB

    while lR <= theta_i:
            v = random.choice(range(1, n))
            rr_nodes = generate_rr_ic(G.nodes()[v], G, p)
            R.append(rr_nodes)
            lR += 1

    return R


def imm(G, k, p):
    epsilon = 0.5
    l = 1
    l = l*(1+ log(2)/log(len(G.nodes())))
    R = sampling(G, k, p, epsilon, l)
    Sk, _ = node_selection(R, k, len(G.nodes()))

    return Sk