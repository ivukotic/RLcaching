# emulates LRU cleanup method
# it always performs the same action (0)
# reward is not important to this one
# it is important to look at the cache hit rate produced by the environment.

import gym
import numpy as np
import pandas as pd


class LRUAgent:

    def act(self, state):
        # print(state)
        return 0
        # return np.argmax(act_values[0])


env = gym.make('gym_cache:Cache-v0')
env.set_actor_name('LRU')

state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = LRUAgent()

max_accesses = 60000000

res = {}
inepi_rew = []
sum_rew = 0
time = 0
done = False
bsize = 1000

state = env.reset()
state = np.reshape(state, [1, state_size])
while time < max_accesses and not done:
    # env.render()

    action = agent.act(state)
    next_state, reward, done, _ = env.step(action)
    sum_rew += reward

    next_state = np.reshape(next_state, [1, state_size])
    state = next_state

    if not time % bsize:
        inepi_rew.append(sum_rew/bsize)
        sum_rew = 0

    if not time % 10000:
        print(time)

    time += 1

if not done:
    env.save_monitoring_data()
    env.close()

r1 = pd.DataFrame.from_dict(inepi_rew)
ax = r1.plot(title='Average reward per {} accesses'.format(bsize))
ax.grid(True)

name = "LRU_reward_acc_{}".format(time)
ax.figure.savefig('plots/' + name + '.png')
