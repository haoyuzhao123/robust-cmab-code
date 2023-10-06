from math import *
import numpy as np
import random


class CascadeUCB1_Attack():

    def __init__(self, dataset, num_arms, seed_size, target_arms):
        self.dataset = dataset
        self.T = {}
        self.w_hat = {}
        self.U = {}
        self.t = 1
        self.num_arms = num_arms
        self.seed_size = seed_size
        self.target_arms = target_arms
        self.totalCost = []
        self.cost = []
        self.num_targetarm_played = []
        self.num_basearm_played = []

        for i in range(self.num_arms):
            self.T[i] = 0
            self.w_hat[i] = 0

    def decide(self):
        for i in range(self.num_arms):
            if self.T[i] == 0:
                self.U[i] = float('inf')
            else:
                self.U[i] = self.w_hat[i] + np.sqrt(1.5*np.log(self.t)/self.T[i])
                self.U[i] = min(1, max(self.U[i], 0))
    
        best_arms = list(dict(sorted(self.U.items(), key=lambda x: x[1], reverse=True)).keys())[:self.seed_size]
        
        self.click_prob = 1
        for i in best_arms:
            self.click_prob *= (1 - self.dataset.w[i])

        # print(self.U, best_arms, self.target_arms)

        return best_arms

    
    def numTargetPlayed(self, S):
        num_basearm_played = 0
        num_targetarm_played = 0
        for u in S:
            if u in self.target_arms:
                num_basearm_played += 1
        if num_basearm_played == self.seed_size:
            num_targetarm_played = 1
        num_basearm_played = num_basearm_played*100/self.seed_size
        self.num_basearm_played.append(num_basearm_played)
        if len(self.num_targetarm_played) == 0:
            self.num_targetarm_played.append(num_targetarm_played)
        else:
            self.num_targetarm_played.append(self.num_targetarm_played[-1] + num_targetarm_played)


    def updateParameters(self, C, best_arms):
        if C == -1:
            C = self.num_arms

        cost = 0
        for i in range(self.seed_size):
            if i <= C:
                arm = best_arms[i]
                
                r = self.T[arm]*self.w_hat[arm]
                if i == C:
                    r += 1
                if arm not in self.target_arms:
                    cost += r
                    r = 0
                self.w_hat[arm] = r/(self.T[arm]+1)
                self.T[arm] += 1
            
            else:
                break

        if len(self.totalCost) == 0:
            self.totalCost = [cost]
        else:
            self.totalCost.append(self.totalCost[-1] + cost)
        self.cost.append(cost)

        self.t += 1
        self.numTargetPlayed(best_arms)