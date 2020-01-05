import numpy as np

ACTIONS = [0, 1, 2, 3]
ACTIONS_NAME = ['up', 'right', 'down', 'left']
ACTIONS_MOVEMENT = [[-1, 0], [0, 1], [1, 0], [0, -1]]
GRID_SIZE = 3


class DynamicProgrammingExample:

    def __init__(self):
        self.policy = self.get_policy()
        self.V = np.zeros([GRID_SIZE, GRID_SIZE])

    def get_policy(self):
        policy = np.zeros([GRID_SIZE, GRID_SIZE, len(ACTIONS)])
        policy.fill(1.0 / len(ACTIONS))
        return policy

    def get_next_state(self, s, a):
        s_next = (
            min(max(s[0] + ACTIONS_MOVEMENT[a][0], 0), GRID_SIZE-1),
            min(max(s[1] + ACTIONS_MOVEMENT[a][1], 0), GRID_SIZE-1)
        )
        return s_next
    
    def get_reward(self, s, a):
        s_next = self.get_next_state(s, a)
        r = 0
        if s_next == (0, GRID_SIZE-1):
            r = -0.1  # trap
        elif s_next == (GRID_SIZE-1, 0):
            r = -0.1  # trap
        elif s_next == (GRID_SIZE-1, GRID_SIZE-1):
            r = 1  # goal
        return r

    def policy_iteration(self, K=10, gamma=1):
        self.policy_evaluation(K=K, gamma=gamma)
        self.policy_improvement()

    def policy_evaluation(self, K=10, gamma=1):
        for k in range(K):
            V_next = np.zeros([GRID_SIZE, GRID_SIZE])
            for s in [(i, j) for i in range(GRID_SIZE) for j in range(GRID_SIZE)]:
                v = 0
                for a in ACTIONS:
                    s_next = self.get_next_state(s, a)
                    v += self.policy[s[0]][s[1]][a] * (self.get_reward(s, a) + gamma * self.V[s_next])
                V_next[s] = v
            self.V = V_next
            with np.printoptions(precision=3, suppress=True):
                print('k={}, V={}'.format(k+1, self.V))
        return self.V

    def policy_improvement(self):
        for s in [(i, j) for i in range(GRID_SIZE) for j in range(GRID_SIZE)]:
            list_v = [np.round(self.V[self.get_next_state(s, a)], 3) for a in ACTIONS]
            list_a_max = [a for a, v in enumerate(list_v) if v == max(list_v)] 

            self.policy[s[0]][s[1]] = [0] * len(ACTIONS)
            for a in list_a_max:
                self.policy[s[0]][s[1]][a] = 1.0 / len(list_a_max)
        with np.printoptions(precision=3, suppress=True):
            print('pi={}'.format(self.policy))
        return self.policy

    def value_iteration(self, K=10, gamma=1):
        for k in range(K):
            V_next = np.zeros([GRID_SIZE, GRID_SIZE])
            for s in [(i, j) for i in range(GRID_SIZE) for j in range(GRID_SIZE)]:
                list_r = [self.get_reward(s, a) for a in ACTIONS]
                v = 0
                for a in ACTIONS:
                    s_next = self.get_next_state(s, a)
                    v += gamma * self.V[s_next]
                V_next[s] = max(list_r) + v
            self.V = V_next
            with np.printoptions(precision=3, suppress=True):
                print('k={}, V={}'.format(k+1, self.V))
        return self.V

if __name__ == "__main__":
    dp = DynamicProgrammingExample()
    dp.policy_iteration(K=8)
    dp.value_iteration(K=8, gamma=0.9)