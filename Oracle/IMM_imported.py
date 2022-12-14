import random
import multiprocessing as mp
import time
import getopt
import sys
import math
import heapq


worker = []

class Worker(mp.Process):
    def __init__(self, inQ, outQ, n, G, p):
        super(Worker, self).__init__(target=self.start)
        self.inQ = inQ
        self.outQ = outQ
        self.R = []
        self.count = 0
        self.node_num = n
        self.G = G
        self.p = p

    def run(self):

        while True:
            theta = self.inQ.get()
            while self.count < theta:
                v = self.G.nodes()[random.randint(0, self.node_num-1)]
                rr = generate_rr_ic(v, self.G, self.p)
                self.R.append(rr)
                self.count += 1
            self.count = 0
            self.outQ.put(self.R)
            self.R = []


def create_worker(num, n, G, p):
    """
        create processes
        :param num: process number
        :param task_num: the number of tasks assigned to each worker
    """
    global worker
    for i in range(num):
        worker.append(Worker(mp.Queue(), mp.Queue(), n, G, p))
        worker[i].start()


def finish_worker():
    """
    关闭所有子进程
    :return:
    """
    for w in worker:
        w.terminate()


def sampling(G, k, p, epsoid=0.5, l=1):
    R = []
    LB = 1
    n = len(G.nodes())
    epsoid_p = epsoid * math.sqrt(2)
    worker_num = 1
    create_worker(worker_num, n, G, p)
    for i in range(1, int(math.log2(n-1))+1):
        s = time.time()
        x = n/(math.pow(2, i))
        lambda_p = ((2+2*epsoid_p/3)*(logcnk(n, k) + l*math.log(n) + math.log(math.log2(n)))*n)/pow(epsoid_p, 2)
        theta = lambda_p/x
        for ii in range(worker_num):
            worker[ii].inQ.put((theta-len(R))/worker_num)
        for w in worker:
            R_list = w.outQ.get()
            R += R_list

        end = time.time()
        # print('time to find rr', end - s)
        start = time.time()
        Si, f = node_selection(R, k, n)
        # print(f)
        end = time.time()
        # print('node selection time', time.time() - start)
        if n*f >= (1+epsoid_p)*x:
            LB = n*f/(1+epsoid_p)
            break

    alpha = math.sqrt(l*math.log(n) + math.log(2))
    beta = math.sqrt((1-1/math.e)*(logcnk(n, k)+l*math.log(n)+math.log(2)))
    lambda_aster = 2*n*pow(((1-1/math.e)*alpha + beta), 2)*pow(epsoid, -2)
    theta = lambda_aster / LB
    length_r = len(R)
    diff = theta - length_r
    _start = time.time()
    if diff > 0:
        for ii in range(worker_num):
            worker[ii].inQ.put(diff/ worker_num)
        for w in worker:
            R_list = w.outQ.get()
            R += R_list
    '''
    
    while length_r <= theta:
        v = random.randint(1, n)
        rr = generate_rr(v)
        R.append(rr)
        length_r += 1
    '''
    _end = time.time()
    finish_worker()
    return R


def node_selection(R, k, n):
    Sk = set()
    rr_degree = {}
    node_rr_set = dict()
    # node_rr_set_copy = dict()
    matched_count = 0
    for j in range(0, len(R)):
        rr = R[j]
        for rr_node in rr:
            if rr_node not in rr_degree.keys():
                rr_degree[rr_node] = 1
            else:
                rr_degree[rr_node] += 1
            if rr_node not in node_rr_set:
                node_rr_set[rr_node] = list()
            node_rr_set[rr_node].append(j)
            # node_rr_set_copy[rr_node].append(j)
    for i in range(k):
        max_point = max(rr_degree, key=rr_degree.get)
        Sk.add(max_point)
        matched_count += len(node_rr_set[max_point])
        index_set = []
        for node_rr in node_rr_set[max_point]:
            index_set.append(node_rr)
        for jj in index_set:
            rr = R[jj]
            for rr_node in rr:
                rr_degree[rr_node] -= 1
                node_rr_set[rr_node].remove(jj)
    return Sk, matched_count/len(R)


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


def imm(G, k, p):
    global worker
    epsoid = 0.5
    worker = []
    l = 1
    n = len(G.nodes())
    l = l*(1+ math.log(2)/math.log(n))
    R = sampling(G, k, p, epsoid, l)
    Sk, z = node_selection(R, k, n)
    Sk = list(Sk)
    return Sk


def logcnk(n, k):
    res = 0
    for i in range(n-k+1, n+1):
        res += math.log(i)
    for i in range(1, k+1):
        res -= math.log(i)
    return res