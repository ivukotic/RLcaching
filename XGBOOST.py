import random
import gym
import numpy as np
import pandas as pd
from collections import deque
import xgboost as xgb

# won't be needed for actual actor.
# here just in case I want to check its performance
from sklearn.model_selection import train_test_split
# X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)

# have to transform input data into its own representation


class XGBagent:
    def __init__(self, batch_size=1000, window_size=50000):
        self.accesses = 0
        self.batch_size = batch_size
        self.window_size = window_size
        self.param = {
            'eta': 0.2,  # learning rate
            'max_depth': 6,
            'objective': 'multi:softprob',
            'num_class': 2
        }
        self.memory = deque(maxlen=2000)
        self.model = None

    def _build_model(self):

        D_train = xgb.DMatrix(X_train, label=Y_train)
        self.model = xgb.train(self.param, D_train, steps)

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward))

    def act(self, state):
        if self.accesses < self.batch_size:
            return random.randrange(2)  # TODO check this gives 0 or 1
        if not self.accesses % self.batch_size:
            self._build_model()
        pred = model.predict(D_test)
        return pred  # returns action

    def load(self, name):
        self.model = keras.models.load_model(name)

    def save(self, name):
        self.model.save(name)

    def print(self):
        pass  # TODO interesting to see what it learned.


env = gym.make('gym_cache:Cache-v0')
env.set_actor_name('XGB')

state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = XGBagent(state_size, action_size)
# agent.load("./save/cache-ddqn_" + name)
accesses = 20000
# accesses = 1000000
# accesses = 3090162


name = "acc{}".format(accesses)
epi_rew = []
inepi_rew = []
in_act = [0, 0]

for e in range(EPISODES):

    epi_rew.append(0)
    inepi_rew.append([])
    sum_rew = 0

    state = env.reset()
    state = np.reshape(state, [1, state_size])
    for time in range(accesses):
        # env.render()

        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        sum_rew += reward
        epi_rew[e] += reward
        in_act[action] += 1

        next_state = np.reshape(next_state, [1, state_size])
        agent.memorize(state, action, reward, next_state, done)
        state = next_state

        if not time % updata_steps:
            agent.update_target_model()
            print("episode: {}/{}, access: {}, score: {}, e: {:.2}, actions: {}".format(e,
                                                                                        EPISODES, time, sum_rew, agent.epsilon, in_act))
            inepi_rew[e].append(sum_rew)
            sum_rew = 0
            in_act = [0, 0]

            if len(agent.memory) > replay_batch_size:
                agent.replay(replay_batch_size)

#     if e % 10 == 0:
    agent.save("./save/cache-ddqn_" + name)


print(epi_rew)

env.close()

# r1 = pd.DataFrame.from_dict(inepi_rew[0])
# ax = r1.plot()
# ax.grid(True)
# ax.figure.savefig('results/plots/' + name + '.png')
