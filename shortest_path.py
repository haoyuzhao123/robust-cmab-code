import argparse
import time
import os
import pickle 
import datetime
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
# from conf import *
from Tool.utilFunc import *

from BanditAlg.CLCB import LCB1Algorithm 
from BanditAlg.CLCB_Attack import LCB1AlgorithmAttack
from Oracle.SP import ShortestPath,  TargetPath_Unattackable, TargetPath_Attackable, TargetPath_RandomWalk, TargetPath_RandomWeight

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

def shortest_path_env_step(P, S):
    '''
    P: networkx Digraph
    S: path_list
    return the realizations of path list S based on the value in the true weights graph P
    return as a fictionary with (u,v) as keys and realization as value
    '''
    live_edges = {}
    total_cost = 0
    # print(S)
    for i in range(len(S)):
        if i==0:
            u = S[i]
            continue
        v = S[i]
        realization = np.random.binomial(1, P[u][v]["weight"])
        total_cost += realization
        live_edges[(u,v)] =  realization
        u = v
    return total_cost, live_edges

class simulateOnlineData:
    def __init__(self,  P, oracle, iterations, dataset):
        self.TrueP = P
        self.oracle = oracle
        self.iterations = iterations
        self.dataset = dataset
        self.startTime = datetime.datetime.now()
        self.BatchCumlateReward = {}
        self.AlgReward = {}

    def runAlgorithms(self, algorithms, oracle_params):
        self.tim_ = []
        for alg_name, alg in list(algorithms.items()):
            self.AlgReward[alg_name] = []
            self.BatchCumlateReward[alg_name] = []

        self.resultRecord()

        for iter_ in range(self.iterations):

            
            for alg_name, alg in list(algorithms.items()): 
                Superarm = alg.decide(oracle_params) 
                total_cost, live_edges = shortest_path_env_step(P, Superarm)
                alg.updateParameters(live_edges, iter_)

                self.AlgReward[alg_name].append(total_cost)
                if iter_ % 200 == 0:
                    print(iter_, alg_name, total_cost)

            self.resultRecord(iter_)

        self.showResult()
        attack_algo = 'CLCB_Attack'
        num_targetarm_played = algorithms[attack_algo].num_targetarm_played
        final_length = 500
        percent = num_targetarm_played[-1]/len(num_targetarm_played)
        if len(num_targetarm_played) >final_length:
            percent = (num_targetarm_played[-1]-num_targetarm_played[-final_length-1])/ final_length
        print(percent)
        return percent > 0.95

    def resultRecord(self, iter_=None):
        # if initialize
        if iter_ is None:
            timeRun = self.startTime.strftime('_%m_%d_%H_%M_%S') 
            fileSig = '_iter'+str(self.iterations)+'_'+str(self.oracle.__name__)+'_'+self.dataset
            self.filenameWriteReward = os.path.join(save_address, 'AccReward' + timeRun + fileSig + '.csv')

            with open(self.filenameWriteReward, 'w') as f:
                f.write('Time(Iteration)')
                f.write(',' + ','.join( [str(alg_name) for alg_name in algorithms.keys()]))
                f.write('\n') 
        else:
            # if run in the experiment, save the results
            # print("Iteration %d" % iter_, " Elapsed time", datetime.datetime.now() - self.startTime)
            self.tim_.append(iter_)
            for alg_name in algorithms.keys():
                self.BatchCumlateReward[alg_name].append(sum(self.AlgReward[alg_name][-1:]))
            with open(self.filenameWriteReward, 'a+') as f:
                f.write(str(iter_))
                f.write(',' + ','.join([str(self.BatchCumlateReward[alg_name][-1]) for alg_name in algorithms.keys()]))
                f.write('\n')

    def showResult(self):
        # print('average Shortest Path Weights for oracle:', np.mean(self.result_oracle))
        
        # reward
        # for alg_name in algorithms.keys():
        #     f, axa = plt.subplots(1, sharex=True)
        #     axa.plot(self.tim_, algorithms[alg_name].basearmestimate1, label = alg_name+"[83,86](target)")
        #     axa.plot(self.tim_, algorithms[alg_name].basearmestimate2, label = alg_name+"[86,1937](target)")            
        #     axa.plot(self.tim_, algorithms[alg_name].basearmestimate3, label = alg_name+"[83,85](shortest)")
        #     axa.plot(self.tim_, algorithms[alg_name].basearmestimate4, label = alg_name+"[85,1937](shortest)")
        #     axa.legend(loc='upper left',prop={'size':9})
        #     axa.set_xlabel("Iteration")
        #     axa.set_ylabel("LCB")
        #     axa.set_title("LCB")
        #     plt.savefig('./SimulationResults/basearms'+ alg_name+ str(self.startTime.strftime('_%m_%d_%H_%M'))+'.png')
        #     plt.show()

        f, axa = plt.subplots(1, sharex=True)
        for alg_name in algorithms.keys():  
            axa.plot(self.tim_, self.BatchCumlateReward[alg_name],label = alg_name)
            print('%s: %.2f' % (alg_name, np.mean(self.BatchCumlateReward[alg_name])))
        axa.legend(loc='upper left',prop={'size':9})
        axa.set_xlabel("Iteration")
        axa.set_ylabel("Shortest Path Weights")
        axa.set_title("Average Shortest Path Weights")
        plt.savefig('./SimulationResults/AvgWeights' + str(self.startTime.strftime('_%m_%d_%H_%M'))+'.png')
        plt.show()

        # plot accumulated reward
        f, axa = plt.subplots(1, sharex=True)
        for alg_name in algorithms.keys():  
            result = [sum(self.BatchCumlateReward[alg_name][:i]) for i in range(len(self.tim_))]
            axa.plot(self.tim_, result, label = alg_name)
        axa.legend(loc='upper left',prop={'size':9})
        axa.set_xlabel("Iteration")
        axa.set_ylabel("Shortest Path Weights")
        axa.set_title("Accumulated Shortest Path Weights")
        plt.savefig('./SimulationResults/AcuWeights' + str(self.startTime.strftime('_%m_%d_%H_%M'))+'.png')
        plt.show()
        
        attack_algo = 'CLCB_Attack'
        # plot cost
        f, axa = plt.subplots(1, sharex=True)
        axa.plot(self.tim_, algorithms[attack_algo].cost, label = attack_algo)
        axa.legend(loc='upper left',prop={'size':9})
        axa.set_xlabel("Iteration")
        axa.set_ylabel("Cost")
        axa.set_title("Cost")
        plt.savefig('./SimulationResults/Cost' + str(self.startTime.strftime('_%m_%d_%H_%M'))+'.png')
        plt.show()

        # plot cumulative cost
        f, axa = plt.subplots(1, sharex=True)
        axa.plot(self.tim_, algorithms[attack_algo].totalCost, label = attack_algo)
        axa.legend(loc='upper left',prop={'size':9})
        axa.set_xlabel("Iteration")
        axa.set_ylabel("Cost")
        axa.set_title("Total Cost")
        plt.savefig('./SimulationResults/TotalCost' + str(self.startTime.strftime('_%m_%d_%H_%M'))+'.png')
        plt.show()

        # plot basearm played
        f, axa = plt.subplots(1, sharex=True)
        axa.plot(self.tim_, algorithms[attack_algo].num_basearm_played, label = attack_algo)
        axa.legend(loc='upper left',prop={'size':9})
        axa.set_xlabel("Iteration")
        axa.set_ylabel("Percentage")
        axa.set_title("Percentage of basearms in superarm played")
        plt.savefig('./SimulationResults/BasearmPlayed' + str(self.startTime.strftime('_%m_%d_%H_%M'))+'.png')
        plt.show()

        # plot superarm played
        f, axa = plt.subplots(1, sharex=True)
        axa.plot(self.tim_, algorithms[attack_algo].num_targetarm_played, label = attack_algo)
        axa.legend(loc='upper left',prop={'size':9})
        axa.set_xlabel("Iteration")
        axa.set_ylabel("Count")
        axa.set_title("Number of times target arm is played")
        plt.savefig('./SimulationResults/TargetarmPlayed' + str(self.startTime.strftime('_%m_%d_%H_%M'))+'.png')
        plt.show()

        print(algorithms[attack_algo].getLoss()[0,0])
        print(algorithms[attack_algo].getLoss()[:,0])
        f, axa = plt.subplots(1, sharex=True)
        axa.plot(self.tim_, algorithms[attack_algo].getLoss()[:,0], label = attack_algo)
        axa.plot(self.tim_, algorithms['CLCB'].getLoss()[:,0], label = 'CLCB')
        axa.legend(loc='upper left',prop={'size':9})
        axa.set_xlabel("Iteration")
        axa.set_ylabel("loss")
        axa.set_title("average LCB - mu")
        plt.savefig('./SimulationResults/Loss' + str(self.startTime.strftime('_%m_%d_%H_%M'))+'.png')
        plt.show()

        # for alg_name in algorithms.keys():  
        #     try:
        #         loss = algorithms[alg_name].getLoss()
        #     except:
        #         continue
            
        #     f, ax1 = plt.subplots()
        #     color = 'tab:red'
        #     ax1.set_xlabel("Iteration")
        #     ax1.set_ylabel('Loss of s', color=color)
        #     ax1.plot(self.tim_, loss, color=color, label=alg_name)
        #     ax1.tick_params(axis='y', labelcolor=color)
        #     # ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        #     # ax2.set_ylabel('Loss of Theta and Beta', color='tab:blue')  # we already handled the x-label with ax1
        #     # ax2.plot(self.tim_, loss[:, 1], color='tab:blue', linestyle=':', label = r'$\theta$')
        #     # ax2.plot(self.tim_, loss[:, 2], color='tab:blue', linestyle='--', label = r'$\beta$')
        #     # ax2.tick_params(axis='y', labelcolor='tab:blue')
        #     # ax2.legend(loc='upper left',prop={'size':9})
        #     f.tight_layout()  # otherwise the right y-label is slightly clipped
        #     plt.savefig('./SimulationResults/Loss' + str(self.startTime.strftime('_%m_%d_%H_%M'))+'.png')
        #     plt.show()
            
        #     np.save('./SimulationResults/Loss-{}'.format(alg_name) + str(self.startTime.strftime('_%m_%d_%H_%M'))+'.npy', loss)
            
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--graph", help="the address of graph data", type=str, default='./datasets/Flickr/G_Union.G')
    parser.add_argument("--prob", help="the address of probability", type=str, default='./datasets/Flickr/ProbUnion.dic')
    parser.add_argument("--param", help="the address of parameters", type=str, default='./datasets/Flickr/NodeFeaturesUnion.dic')
    parser.add_argument("--edge_feature", help="the address of edge features", type=str, default='./datasets/Flickr/EdgeFeaturesUnion.dic')
    parser.add_argument("--dataset", help="the address of dataset(Choose from 'default', 'NetHEPT', or 'Flickr' as default)", type=str, default='Flickr-Random')
    parser.add_argument("--iter",  type=int, default=3000)
    parser.add_argument("--save_address",  type=str, default='./SimulationResults')
    parser.add_argument("--seed", help="random seed", type=int, default=0)

    args = parser.parse_args()
    graph_address = args.graph
    prob_address = args.prob
    param_address = args.param
    edge_feature_address = args.edge_feature
    dataset = args.dataset
    iterations = args.iter
    save_address = args.save_address

    oracle = ShortestPath
    np.random.seed(args.seed)

    start = time.time()

    G = pickle.load(open(graph_address, 'rb'), encoding='latin1')
    prob = pickle.load(open(prob_address, 'rb'), encoding='latin1')
    # parameter = pickle.load(open(param_address, 'rb'), encoding='latin1')
    # feature_dic = pickle.load(open(edge_feature_address, 'rb'), encoding='latin1')
    G = clean_to_weakly_connected(G)

    P = nx.DiGraph()
    id_node = {}
    nodes = 0
    for (u,v) in G.edges():
        if u not in id_node:
            nodes += 1
            id_node[u] = nodes
        u_id = id_node[u]
        if v not in id_node:
            nodes += 1
            id_node[v] = nodes
        v_id = id_node[v]
        # print(u_id,v_id,prob[(u,v)])
        P.add_edge(u_id, v_id, weight=prob[(u,v)])

    print('nodes:', len(G.nodes()))
    print('edges:', len(G.edges()))
    print('Done with Loading Feature')
    print('Graph build time:', time.time() - start)
    num_exp = 10
    sum_attackable = 0
    for seed in range(100,100+num_exp):
        np.random.rand(seed)
        simExperiment = simulateOnlineData(P, oracle, iterations, dataset)

        Target, oracle_params = TargetPath_Unattackable(P) #TargetPath_RandomWeight(P)

        # nx.shortest_path(P,oracle_params["start"],oracle_params["end"],weight="weight")
        algorithms = {}
        algorithms['CLCB'] = LCB1Algorithm(P,oracle)
        algorithms['CLCB_Attack'] = LCB1AlgorithmAttack(P, oracle, Target)

        attackable = simExperiment.runAlgorithms(algorithms, oracle_params)
        sum_attackable += attackable
        print("attackable", attackable, sum_attackable/(seed-100+1))
        
