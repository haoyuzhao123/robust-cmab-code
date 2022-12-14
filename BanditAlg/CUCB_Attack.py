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
       
    def updateParameters(self, reward):
        self.totalReward += reward
        self.numPlayed +=1
        self.averageReward = self.totalReward/float(self.numPlayed)

class UCB1Struct(ArmBaseStruct):    
    def getProb(self, allNumPlayed):
        if self.numPlayed==0:
            return 0
        else:
            p = self.totalReward / float(self.numPlayed) + 0.1*np.sqrt(3*np.log(allNumPlayed) / (2.0 * self.numPlayed))
            if p > self.p_max:
                p = self.p_max
                # print 'p_max'
            return p

             
class UCB1AlgorithmAttack:
    def __init__(self, G, P, parameter, seed_size, oracle, feedback = 'edge'):
        self.G = G
        self.trueP = P
        self.parameter = parameter  
        self.seed_size = seed_size
        self.oracle = oracle
        self.feedback = feedback
        self.arms = {}
        #Initialize P
        self.currentP =nx.DiGraph()
        for (u,v) in self.G.edges():
            self.arms[(u,v)] = UCB1Struct((u,v))
            self.currentP.add_edge(u,v, weight=0)
        self.list_loss = []
        self.TotalPlayCounter = 0
        S_Star_nodes = sample(range(len(G.nodes())), self.seed_size)
        self.S_star = []
        for i in S_Star_nodes:
            self.S_star.append(G.nodes()[i])
        self.num_targetarm_played = []
        self.totalCost = []
        self.cost = []
        self.num_basearm_played = []
        
    def decide(self):
        self.TotalPlayCounter +=1
        S = self.oracle(self.G, self.seed_size, self.currentP)
        return S       

    def numTargetPlayed(self, S):
        num_basearm_played = 0
        num_targetarm_played = 0
        for u in S:
            if u in self.S_star:
                num_basearm_played += 1
        if num_basearm_played == self.seed_size:
            num_targetarm_played = 1
        num_basearm_played = num_basearm_played*100/self.seed_size
        self.num_basearm_played.append(num_basearm_played)
        if len(self.num_targetarm_played) == 0:
            self.num_targetarm_played.append(num_targetarm_played)
        else:
            self.num_targetarm_played.append(self.num_targetarm_played[-1] + num_targetarm_played)

         
    def updateParameters(self, S, live_nodes, live_edges, iter_): 
        count = 0
        loss_p = 0 
        loss_out = 0
        loss_in = 0
        cost = 0
        for u in live_nodes:
            for (u, v) in self.G.edges(u):
                if (u,v) in live_edges:
                    if (u in self.S_star and u in S) or (v in self.S_star and v in S):
                        self.arms[(u, v)].updateParameters(reward=live_edges[(u,v)])
                    else:
                        self.arms[(u, v)].updateParameters(reward=0)
                        cost += live_edges[(u,v)]
                else:
                    self.arms[(u, v)].updateParameters(reward=0)
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
        self.numTargetPlayed(S)

    def getLoss(self):
        return np.asarray(self.list_loss)

    def getP(self):
        return self.currentP

