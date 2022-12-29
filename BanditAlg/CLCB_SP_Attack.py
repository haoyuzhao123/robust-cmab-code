from random import choice, random, sample
import numpy as np
import networkx as nx

class ArmBaseStruct(object):
    def __init__(self, armID):
        self.armID = armID
        self.totalReward = 0.0
        self.numPlayed = 0
        self.averageReward  = 0.0
        self.p_max = 1
        self.p_min = 0
       
    def updateParameters(self, reward):
        self.totalReward += reward
        self.numPlayed +=1
        self.averageReward = self.totalReward/float(self.numPlayed)

class LCB1Struct(ArmBaseStruct):    
    def getProb(self, allNumPlayed):
        if self.numPlayed==0:
            return 0
        else:
            p = self.totalReward / float(self.numPlayed) - 0.1*np.sqrt(3*np.log(allNumPlayed) / (2.0 * self.numPlayed))
            if p > self.p_max:
                p = self.p_max
                # print 'p_max'
            if p < self.p_min:
                p = self.p_min
            return p

             
class LCB1AlgorithmAttack:
    def __init__(self, P, oracle, target, feedback = 'edge'):
        # self.G = G
        self.trueP = P
        # self.parameter = parameter  
        self.oracle = oracle
        self.feedback = feedback
        self.arms = {}
        #Initialize P
        self.currentP =nx.DiGraph()
        for (u,v) in P.edges():
            self.arms[(u,v)] = LCB1Struct((u,v))
            self.currentP.add_edge(u,v, weight=0)
        self.list_loss = []
        self.TotalPlayCounter = 0

        path_list = target
        self.opt_path_length = len(path_list) - 1
        self.opt_path = {}
        for i in range(len(path_list)):
            if i==0:
                u = path_list[i]
                continue
            v = path_list[i]
            self.opt_path[(u,v)] = 0
            u = v
       
        self.num_targetarm_played = []
        self.totalCost = []
        self.cost = []
        self.num_basearm_played = []
        
    def decide(self, params):
        self.TotalPlayCounter += 1
        S = self.oracle(self.currentP, params)
        # S = self.oracle(self.G, self.seed_size, self.currentP)
        return S       

    def numTargetPlayed(self, live_edges):
        num_basearm_played = 0
        num_targetarm_played = 0
        for (u,v) in live_edges:
            if (u,v) in self.opt_path:
                num_basearm_played += 1
                self.opt_path[(u,v)] = self.opt_path[(u,v)] + 1
        if num_basearm_played == self.opt_path_length:
            num_targetarm_played = 1
        num_basearm_played = num_basearm_played/self.opt_path_length
        self.num_basearm_played.append(num_basearm_played)
        if len(self.num_targetarm_played) == 0:
            self.num_targetarm_played.append(num_targetarm_played)
        else:
            self.num_targetarm_played.append(self.num_targetarm_played[-1] + num_targetarm_played)

         
    def updateParameters(self, live_edges, iter_): 
        count = 0
        loss_p = 0 
        cost = 0
        for (u, v, weight) in self.trueP.edges(data="weight"):
            if (u,v) in live_edges:
                if (u,v) in self.opt_path:
                    self.arms[(u, v)].updateParameters(reward=live_edges[(u,v)])
                else:
                    self.arms[(u, v)].updateParameters(reward=1)
                    cost += 1 - live_edges[(u,v)] # or just 1

            #update current P
            #print self.TotalPlayCounter
            self.currentP[u][v]['weight'] = self.arms[(u,v)].getProb(self.TotalPlayCounter) 
            estimateP = self.currentP[u][v]['weight']
            trueP = self.trueP[u][v]['weight']
            loss_p += np.abs(estimateP-trueP)
            count += 1
        self.list_loss.append([loss_p/count])
        if len(self.totalCost) == 0:
            self.totalCost = [cost]
        else:
            self.totalCost.append(self.totalCost[-1] + cost)
        self.cost.append(cost)
        self.numTargetPlayed(live_edges)

    def getLoss(self):
        return np.asarray(self.list_loss)

    def getP(self):
        return self.currentP

