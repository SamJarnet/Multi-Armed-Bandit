import random
import numpy as np


class Bandit:
    def __init__(self):
        self.grid = np.array([['S', '.','.','.','.'], ['.', 'X','.','X','.'], ['.', '.','.','X','.'], ['.', 'X','.','.','.'], ['.', '.','.','.','G']])
        self.start = np.array([0, 0])
        self.goal = np.array([4, 4])
        self.pos = self.start.copy()
        self.actions = {
            0: np.array([0, 1]),
            1: np.array([1, 0]),
            2: np.array([0, -1]),
            3: np.array([-1, 0])
        }

    def reset(self):
        self.pos = self.start.copy()

    def fix_pos(self, pos):
        return pos[0] * 5 + pos[1]
    
    def step(self, action):
        reward = -1
        done = False
        move = self.actions.get(action, np.array([0 , 0]))
        future = self.pos + move

        if 0 <= future[0] < 5 and 0 <= future[1] < 5:
            future_index = self.grid[future[0], future[1]]

            if future_index != 'X':
                self.pos = future
    
                if future_index == 'G':
                    reward = 100
                    done = True

        return self.fix_pos(self.pos), reward, done

   

    def learn(self, eps):
        Q = np.zeros((25,4))
        alpha = 0.1
        gamma = 0.9
        epsilon = 0.05
        for i in range(0, eps):
            self.reset()
            done = False
            while not done:
                previous = self.fix_pos(self.pos)
                if random.random() < epsilon:
                    action = random.randint(0,3)
                else:
                    action = np.argmax(Q[previous])
                pos, reward, done = self.step(action)
                max_a = np.argmax(Q[pos])
                td_target = reward + gamma * Q[pos][max_a]
                Q[previous][action] +=  alpha * (td_target - Q[previous][action])
            epsilon = max(0.05, epsilon * 0.9995)
        return Q

test = Bandit()
Q = test.learn(100000)
policy = np.zeros((25,))
for s in range(25):
    policy[s] = np.argmax(Q[s])
policy.resize(5,5)
print(policy)

