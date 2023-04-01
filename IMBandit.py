import time
import os
import pickle 
import datetime
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from IMconf import *
from Tool.utilFunc import *
import random

from BanditAlg.CUCB import UCB1Algorithm 
from BanditAlg.CUCB_Attack import UCB1AlgorithmAttack
# from BanditAlg.BanditAlgorithms_MF import MFAlgorithm
# from BanditAlg.BanditAlgorithms_LinUCB import N_LinUCBAlgorithm
from IC.IC import runICmodel_n, runICmodel_single_step
# from IC.runIAC  import weightedEp, runIAC, runIACmodel, randomEp, uniformEp

def clean_to_weakly_connected(P):
    comp_gen = nx.weakly_connected_components(P)
    max_comp = []
    s = 0
    num = 0
    for comp in comp_gen:
        if len(comp)>len(max_comp):
            max_comp  = comp
        # print(len(comp))
        s += len(comp)
        num += 1
    print("forests size",s, num)
    cleaned_G = G.subgraph(max_comp)
    comp_gen = nx.weakly_connected_components(cleaned_G)
    for comp in comp_gen:
        print("final comp",len(comp))
    return cleaned_G


class simulateOnlineData:
    def __init__(self, G, P, oracle, seed_size, iterations, dataset):
        self.G = G
        self.TrueP = P
        self.seed_size = seed_size
        self.oracle = oracle
        self.iterations = iterations
        self.dataset = dataset
        self.startTime = datetime.datetime.now()
        self.BatchCumlateReward = {}
        self.AlgReward = {}
        self.result_oracle = []
        self.result_oracle_rand = []

    def runAlgorithms(self, algorithms):
        self.tim_ = []
        for alg_name, alg in list(algorithms.items()):
            self.AlgReward[alg_name] = []
            self.BatchCumlateReward[alg_name] = []

        self.resultRecord()
        # optS = self.oracle(self.G, self.seed_size, self.TrueP)
        # optrandS = self.oracle(self.G, self.seed_size, self.TruerandP)

        for iter_ in range(self.iterations):
            # optimal_reward_S, live_nodes, live_edges, _ = runICmodel_single_step(G, optS, self.TrueP)
            # optimal_reward_randS, live_nodes, live_edges, _ = runICmodel_single_step(G, optrandS, self.TruerandP)

            # self.result_oracle.append(optimal_reward_S)
            # self.result_oracle_rand.append(optimal_reward_randS)

            # print('oracle', optimal_reward_S)
            
            for alg_name, alg in list(algorithms.items()): 
                S = alg.decide() 
                reward, live_nodes, live_edges, _ = runICmodel_n(G, S, self.TrueP)

                alg.updateParameters(S, live_nodes, live_edges, iter_)

                self.AlgReward[alg_name].append(reward)

            self.resultRecord(iter_)

        self.showResult()

    def resultRecord(self, iter_=None):
        # if initialize
        if iter_ is None:
            self.filenameWriteReward = os.path.join(save_address, 'Reward{}.csv'.format(str(args.exp_num)))
            self.filenameWriteCost = os.path.join(save_address, 'Cost{}.csv'.format(str(args.exp_num)))
            self.filenameTargetRate = os.path.join(save_address, 'Rate{}.csv'.format(str(args.exp_num)))

            if not os.path.exists(save_address):
                os.mkdir(save_address)

            if os.path.exists(self.filenameWriteReward) or os.path.exists(self.filenameWriteCost) or os.path.exists(self.filenameTargetRate):
                raise ValueError ("Save File exists already, please check experiment number")

            with open(self.filenameWriteReward, 'w') as f:
                f.write('Time(Iteration)')
                f.write(',' + ','.join( [str(alg_name) for alg_name in algorithms.keys()]))
                f.write('\n') 

            with open(self.filenameWriteCost, 'w') as f:
                f.write('Time(Iteration)')
                l = []
                for alg_name in algorithms.keys():
                    if 'Attack' in alg_name:
                        l.append(alg_name)
                f.write(',' + ','.join(l))
                f.write('\n') 

            with open(self.filenameTargetRate, 'w') as f:
                f.write('Time(Iteration)')
                l = []
                for alg_name in algorithms.keys():
                    if 'Attack' in alg_name:
                        l.append(alg_name)
                f.write(',' + ','.join(l))
                f.write('\n') 
        
        else:
            # if run in the experiment, save the results
            print("Iteration %d" % iter_, " Elapsed time", datetime.datetime.now() - self.startTime)
            self.tim_.append(iter_)
            for alg_name in algorithms.keys():
                self.BatchCumlateReward[alg_name].append(sum(self.AlgReward[alg_name]))
            with open(self.filenameWriteReward, 'a+') as f:
                f.write(str(iter_))
                f.write(',' + ','.join([str(self.AlgReward[alg_name][-1]) for alg_name in algorithms.keys()]))
                f.write('\n')

            with open(self.filenameWriteCost, 'a+') as f:
                f.write(str(iter_))
                l = []
                for alg_name in algorithms.keys():
                    if 'Attack' in alg_name:
                        l.append(str(algorithms[alg_name].totalCost[-1]))
                f.write(',' + ','.join(l))
                f.write('\n')

            with open(self.filenameTargetRate, 'a+') as f:
                f.write(str(iter_))
                l = []
                for alg_name in algorithms.keys():
                    if 'Attack' in alg_name:
                        l.append(str(algorithms[alg_name].num_targetarm_played[-1]))
                f.write(',' + ','.join(l))
                f.write('\n')

    def showResult(self):
        
        # Reward
        f, axa = plt.subplots(1, sharex=True)
        for alg_name in algorithms.keys():  
            axa.plot(self.tim_, self.AlgReward[alg_name],label = alg_name)
            print('%s: %.2f' % (alg_name, np.mean(self.AlgReward[alg_name])))
        axa.legend(loc='upper left',prop={'size':9})
        axa.set_xlabel("Iteration")
        axa.set_ylabel("Reward")
        axa.set_title("Average Reward")
        plt.savefig('./SimulationResults/IM_seed_size_5/AvgReward' + str(args.exp_num)+'.png')
        plt.show()

        # # plot accumulated reward
        # f, axa = plt.subplots(1, sharex=True)
        # for alg_name in algorithms.keys():  
        #     axa.plot(self.tim_, self.BatchCumlateReward[alg_name],label = alg_name)
        #     print('%s: %.2f' % (alg_name, np.mean(self.BatchCumlateReward[alg_name])))
        # axa.legend(loc='upper left',prop={'size':9})
        # axa.set_xlabel("Iteration")
        # axa.set_ylabel("Reward")
        # axa.set_title("Accumulated Reward")
        # plt.savefig('./SimulationResults/SetCover/AccReward' + str(args.exp_num)+'.png')
        # plt.show()

        # plot cost
        f, axa = plt.subplots(1, sharex=True)
        for alg_name in algorithms.keys():
            if "Attack" in alg_name:
                axa.plot(self.tim_, algorithms[alg_name].cost, label = alg_name)
        axa.legend(loc='upper left',prop={'size':9})
        axa.set_xlabel("Iteration")
        axa.set_ylabel("Cost")
        axa.set_title("Cost")
        plt.savefig('./SimulationResults/IM_seed_size_5/Cost' + str(args.exp_num)+'.png')
        plt.show()

        # plot cumulative cost
        f, axa = plt.subplots(1, sharex=True)
        for alg_name in algorithms.keys():
            if "Attack" in alg_name:
                axa.plot(self.tim_, algorithms[alg_name].totalCost, label = alg_name)
        axa.legend(loc='upper left',prop={'size':9})
        axa.set_xlabel("Iteration")
        axa.set_ylabel("Cost")
        axa.set_title("Total Cost")
        plt.savefig('./SimulationResults/IM_seed_size_5/TotalCost' + str(args.exp_num)+'.png')
        plt.show()

        # plot basearm played
        f, axa = plt.subplots(1, sharex=True)
        for alg_name in algorithms.keys():
            if "Attack" in alg_name:
                axa.plot(self.tim_, algorithms[alg_name].num_basearm_played, label = alg_name)
        axa.legend(loc='upper left',prop={'size':9})
        axa.set_xlabel("Iteration")
        axa.set_ylabel("Percentage")
        axa.set_title("Percentage of basearms in superarm played")
        plt.savefig('./SimulationResults/IM_seed_size_5/BasearmPlayed' + str(args.exp_num)+'.png')
        plt.show()

        # plot superarm played
        f, axa = plt.subplots(1, sharex=True)
        for alg_name in algorithms.keys():
            if "Attack" in alg_name:
                axa.plot(self.tim_, algorithms[alg_name].num_targetarm_played, label = alg_name)
        axa.legend(loc='upper left',prop={'size':9})
        axa.set_xlabel("Iteration")
        axa.set_ylabel("Count")
        axa.set_title("Number of times target arm is played")
        plt.savefig('./SimulationResults/IM_seed_size_5/TargetarmPlayed' + str(args.exp_num)+'.png')
        plt.show()
        
if __name__ == '__main__':
    start = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('exp_num', type=int, default=0)

    args = parser.parse_args()

    G = pickle.load(open(graph_address, 'rb'), encoding='latin1')
    prob = pickle.load(open(prob_address, 'rb'), encoding='latin1')
    parameter = pickle.load(open(param_address, 'rb'), encoding='latin1')
    feature_dic = pickle.load(open(edge_feature_address, 'rb'), encoding='latin1')

    G = clean_to_weakly_connected(G)

    print(f"Number of nodes: {len(G.nodes())}, Number of edges: {len(G.edges())}")

    P = nx.DiGraph()
    # randP = nx.DiGraph()

    np.random.seed(args.exp_num)
    random.seed(args.exp_num)

    for (u,v) in G.edges():
        P.add_edge(u, v, weight=prob[(u,v)])
        # randP.add_edge(u, v, weight=np.random.rand())

    print('nodes:', len(G.nodes()))
    print('edges:', len(G.edges()))
    print('Done with Loading Feature')
    print('Graph build time:', time.time() - start)
    
    n = {}
    n_05 = {}

    # n_rand = {}
    for u in G.nodes():
        for v in G[u]:
            try:
                n[u] += P[u][v]['weight']/len(G[u])
                # n_rand[u] += randP[u][v]['weight']/len(G[u])
            except:
                n[u] = P[u][v]['weight']/len(G[u])
                # n_rand[u] = randP[u][v]['weight']/len(G[u])

    for u in n.keys():
        if n[u] >= 0.5:
            n_05[u] = n[u]

    target_arms = list(dict(sorted(n.items(), key=lambda x: x[1], reverse=True)).keys())[seed_size:2*seed_size]

    target_arms_rand = random.sample(list(n_05.keys()), seed_size)
    # print(target_arms_rand)
    # exit()

    # target_arms_rand = oracle(G, seed_size, P)

    # new_node = G.nodes()[random.sample(range(len(G.nodes())), 1)[0]]
    # target_arms_rand[random.sample(range(seed_size), 1)[0]] = new_node

    # random.seed(1)

    # target_arms_index = random.sample(range(len(G.nodes())), seed_size)


    # cnt = 1
    # target_arms = [G.nodes()[0]]
    # while cnt < seed_size:
    #     target_arms_index = random.sample(range(len(G.nodes())), 1)[0]
    #     f = 0
    #     for t in target_arms:
    #         if G.nodes()[target_arms_index] in G.neighbors(t):
    #             f += 1
    #     if f == 0:
    #         target_arms.append(G.nodes()[target_arms_index])
    #         cnt += 1



    # target_arms = []
    # for i in target_arms_index:
    #     target_arms.append(G.nodes()[i])


    # for v in target_arms:
    #     print("N", G.neighbors(v))
    #     for u in G[v]:
    #         print(v, u, P[v][u]['weight'])

    # print("-------")

    # for v in target_arms_rand:
    #     for u in G[v]:
    #         print(v, u, P[v][u]['weight'])

    # exit()

    simExperiment = simulateOnlineData(G, P, oracle, seed_size, iterations, dataset)

    algorithms = {}
    algorithms['CUCB_Attack_prop=1'] = UCB1AlgorithmAttack(G, P, parameter, seed_size, target_arms, oracle, prop_dist=1)
    algorithms['CUCB_Attack_prop=2'] = UCB1AlgorithmAttack(G, P, parameter, seed_size, target_arms, oracle, prop_dist=2)
    algorithms['CUCB_Attack_prop=3'] = UCB1AlgorithmAttack(G, P, parameter, seed_size, target_arms, oracle, prop_dist=3)
    # algorithms['Randomized CUCB_Attack'] = UCB1AlgorithmAttack(G, P, parameter, seed_size, target_arms_rand, oracle)
    # algorithms['CUCB'] = UCB1Algorithm(G, P, parameter, seed_size, oracle)

    simExperiment.runAlgorithms(algorithms)