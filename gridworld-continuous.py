import random
import numpy as np


class Bandit:
    def __init__(self):
        self.grid_shape = np.array([5, 5]) # now 5 by 5 continuous instead of graph
        self.start = np.array([0.0, 0.0])
        self.goal = np.array([4.9, 4.9]) # 10 by 5 with bins
        self.pos = np.array([0.0, 0.0])
        self.actions = {
            0: np.array([0, 0.1]),
            1: np.array([0.1, 0]),
            2: np.array([0, -0.1]),
            3: np.array([-0.1, 0])
        }
        


    def binning(self, pos):
        bin_count = 50
        bins_x, bins_y = self.grid_shape[0]/bin_count, self.grid_shape[1]/bin_count

        return(int(pos[0]/bins_x), int(pos[1]/bins_y))

    def reset(self):
        self.pos = self.start.copy()

    def fix_pos(self, pos):
        return (pos[0] * 50 + pos[1])
    
    def step(self, action):
        reward = -1
        done = False
        move = self.actions.get(action, np.array([0 , 0]))
        future = self.pos + move
        binned = self.binning(self.pos)
        
        if 0.0 < future[0] < self.grid_shape[0] and 0.0 < future[1] < self.grid_shape[1]:
            self.pos = future
            if np.linalg.norm(self.pos - self.goal) < 0.05:
                reward = 100
                done = True
        return self.fix_pos(binned), reward, done


    
    
    def learn(self, eps):
        V = np.zeros(2500)
        gamma = 0.9

        for i in range(0, eps):
            self.reset()
            done = False
            V_new = np.zeros(2500)

            for s in range(2500):
                row = s // 50
                col = s % 50
                if (row == 49 and col == 49):
                    continue
                    
                actions = []
                for action in [0, 1, 2, 3]:
                    bin_width = 0.1
                    self.pos = np.array([row * bin_width, col * bin_width])
                    next_pos, reward, done = self.step(action)
                    value = reward + gamma * V[next_pos]
                    actions.append(value)
                V_new[s] = max(actions)
                
            V = V_new.copy()
        return V
    

test = Bandit()
np.set_printoptions(threshold=np.inf, linewidth=300, precision=2, suppress=True)
V = test.learn(100)
V_grid = V.reshape(50, 50)
print(V_grid)