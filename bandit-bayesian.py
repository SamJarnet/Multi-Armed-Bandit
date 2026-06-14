import random
import numpy as np

class Bandit:
    def __init__(self):
        self.probs = [0.1, 0.3, 0.8, 0.5]

    def pull(self, arm):
        num = random.random()
        if num < self.probs[arm]:
            return 1
        else:
            return 0
        
    
    def learn(self, eps):
        Q = [0.0,0.0,0.0,0.0] # estimated values of arms
        N = [0,0,0,0] # num of pulls 
        alpha, beta = [1, 1, 1, 1], [1, 1, 1, 1]
        P = [0.0, 0.0, 0.0, 0.0]
        for i in range(0, eps):
            for j in range(0, 4):
                P[j] = np.random.beta(alpha[j], beta[j])

            C = np.argmax(P)

            reward = self.pull(C)
            if reward == 1:
                alpha[C] += 1
            else:
                beta[C] += 1
            N[C] += 1
            Q[C] = Q[C] + (reward-Q[C]) / N[C]
        return Q


test = Bandit()
monte_carlo = [0.0, 0.0, 0.0, 0.0]
n = 10
for i in range(0, n):
    result = test.learn(100000)
    for j in range(0, 4):
        monte_carlo[j] += result[j]/n

print(monte_carlo)
