import gym
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt


class LRUAgent:

    def act(self, state):
        # print(state)
        # if filesize is positive return 1 (cache prediction), else return 0 (remove from cache)
        if state[0][6] > 0:
            return 0
        else:
            return 0
        # return np.argmax(act_values[0])


env = gym.make('gym_cache:Cache-v0')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = LRUAgent()

accesses = 20000
# accesses = 3090162

name = "LRU_acc{}".format(accesses)
inepi_rew = []
sum_rew = 0

state = env.reset()
state = np.reshape(state, [1, state_size])
for time in range(accesses):
    # env.render()

    action = agent.act(state)
    next_state, reward, done, _ = env.step(action)
    sum_rew += reward

    next_state = np.reshape(next_state, [1, state_size])
    state = next_state

    if not time % 1000:
        inepi_rew.append(sum_rew)
        sum_rew = 0

print(inepi_rew)

r1 = pd.DataFrame.from_dict(inepi_rew)
ax = r1.plot()
ax.grid(True)
ax.figure.savefig('plots/' + name + '.png')
