import random
import numpy as np

class Bandit:
    def __init__(self):
        self.probs = [0.1, 0.3, 0.8, 0.5]

    def pull(self, arm):
        num = random.random()
        if num < self.probs[arm]:
            # for i in range(len(self.probs)):
            #     self.probs[i] += random.uniform(-0.01, 0.01)
            #     self.probs[i] = min(1.0, max(0.0, self.probs[i]))
            return 1
        else:
            # for i in range(len(self.probs)):
            #     self.probs[i] += random.uniform(-0.01, 0.01)
            #     self.probs[i] = min(1.0, max(0.0, self.probs[i]))
            return 0
        
    def track(self, eps):
        total_reward = 0
        for i in range(0, eps):
            arm = random.randint(0, 3)
            reward = self.pull(arm)
            total_reward += reward
        average_reward = total_reward/eps
        return total_reward, average_reward
    
    def learn(self, eps):
        Q = [0.0,0.0,0.0,0.0] # estimated values of arms
        N = [0,0,0,0] # num of pulls 
        for i in range(0, 4):
                    reward = self.pull(i)
                    Q[i] += reward
                    N[i] += 1


        for t in range(0, eps):
            ucb = [0.0,0.0,0.0,0.0] 
            for i in range(0, 4):
                if N[i] == 0:
                    ucb[i] = float('inf')
                else:
                    bonus = np.sqrt(2 * np.log(t + 1) / N[i])
                    ucb[i] = Q[i] / N[i] + bonus
            C = ucb.index(max(ucb))

            reward = self.pull(C)
            N[C] += 1
            Q[C] = Q[C] + (reward-Q[C]) / N[C]
        return Q


test = Bandit()
monte_carlo = [0.0, 0.0, 0.0, 0.0]
n = 100
for i in range(0, n):
    result = test.learn(10000)
    for j in range(0, 4):
        monte_carlo[j] += result[j]/n

print(monte_carlo)
