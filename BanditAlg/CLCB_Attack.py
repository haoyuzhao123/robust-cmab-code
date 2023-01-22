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
        self.numPlayed += 1
        self.averageReward = self.totalReward/float(self.numPlayed)

class LCB1Struct(ArmBaseStruct):    
    def getProb(self, allNumPlayed):
        if self.numPlayed==0:
            return 0
        else:
            p = self.totalReward / float(self.numPlayed) - 0.01 * np.sqrt(3*np.log(allNumPlayed) / (2.0 * self.numPlayed))
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
        self.currentP = nx.create_empty_copy(P)
        for (u,v) in P.edges():
            self.arms[(u,v)] = LCB1Struct((u,v))
            self.currentP.add_edge(u,v, weight=0)
        self.list_loss = []
        self.TotalPlayCounter = 0

        list = target
        self.target = {}
        self.target_length = len(target)
        for (u,v) in list:
            self.target[(u,v)] = 0
       
        self.num_targetarm_played = []
        self.totalCost = []
        self.cost = []
        self.num_basearm_played = []
        # self.basearmestimate1 = []
        # self.basearmestimate2 = []
        # self.basearmestimate3 = []
        # self.basearmestimate4 = []   
     
    def decide(self, params):
        self.TotalPlayCounter += 1
        S = self.oracle(self.currentP, params)
        return S       

    def numTargetPlayed(self, live_edges):
        num_basearm_played = 0
        num_targetarm_played = 0
        for (u,v) in live_edges:
            if (u,v) in self.target:
                num_basearm_played += 1
                self.target[(u,v)] = self.target[(u,v)] + 1
        if num_basearm_played == self.target_length:
            num_targetarm_played = 1
        num_basearm_played = num_basearm_played/self.target_length
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
                if (u,v) in self.target:
                    self.arms[(u, v)].updateParameters(reward=live_edges[(u,v)])
                else:
                    self.arms[(u, v)].updateParameters(reward=1)
                    cost += 1 - live_edges[(u,v)] # or just 1

                self.currentP[u][v]['weight'] = self.arms[(u,v)].getProb(self.TotalPlayCounter) 
                estimateP = self.currentP[u][v]['weight']
                trueP = self.trueP[u][v]['weight']
                loss_p += np.abs(estimateP-trueP)
                count += 1
            #update current P
            #print self.TotalPlayCounter
            self.currentP[u][v]['weight'] = self.arms[(u,v)].getProb(self.TotalPlayCounter) 
            # estimateP = self.currentP[u][v]['weight']
            # trueP = self.trueP[u][v]['weight']
            # loss_p += np.abs(estimateP-trueP)
            # count += 1
        self.list_loss.append([loss_p/count])
        if len(self.totalCost) == 0:
            self.totalCost = [cost]
        else:
            self.totalCost.append(self.totalCost[-1] + cost)
        self.cost.append(cost)
        # self.basearmestimate1.append(self.currentP[83][86]['weight'])
        # self.basearmestimate2.append(self.currentP[86][1937]['weight'])
        # self.basearmestimate3.append(self.currentP[83][85]['weight'])
        # self.basearmestimate4.append(self.currentP[85][1937]['weight'])
        self.numTargetPlayed(live_edges)

    def getLoss(self):
        return np.asarray(self.list_loss)

    def getP(self):
        return self.currentP

