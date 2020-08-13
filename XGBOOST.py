import random
import gym
import numpy as np
import pandas as pd
from collections import deque
import xgboost as xgb

# won't be needed for actual actor.
# here just in case I want to check its performance
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score
# X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)

# have to transform input data into its own representation


class XGBagent:
    def __init__(self, batch_size=1000, window_size=50000):
        self.accesses = 0
        self.batch_size = batch_size  # retrain after adding each batch
        self.window_size = window_size  # keep this many last accesses
        self.param = {
            'eta': 0.2,  # learning rate
            'max_depth': 6,
            'objective': 'binary:logistic',
            'nthread': 2
        }
        self.m_state = deque(maxlen=self.window_size)
        self.m_label = deque(maxlen=self.window_size)
        self.model = xgb.Booster(self.param)

    def _retrain_model(self):
        da = np.asarray(self.m_state)
        la = np.asarray(self.m_label)
        # print(da)
        # print(la)
        print('retraining...', da.shape[0])
        D_train = xgb.DMatrix(da, label=la)
        self.model = xgb.train(self.param, D_train, num_boost_round=10)

    def memorize(self, state, action, reward):
        # TODO based on action and reward find out if the file was in the cache
        # this is how reward gets calculated
        # reward = self.weight
        # if (self.found_in_cache and action == 0) or (not self.found_in_cache and action == 1):
        #     reward = -reward
        # print(state, action, reward)
        if reward < 0:
            if action == 0:
                was_cached = True
            else:
                was_cached = False
        else:
            if action == 0:
                was_cached = False
            else:
                was_cached = True
        self.m_state.append(state)
        self.m_label.append(was_cached)
        self.accesses += 1

        if not self.accesses % self.batch_size:
            self._retrain_model()

    def act(self, state):
        if self.accesses < self.batch_size:
            return random.randrange(2)
        # else:
            # return random.randrange(2)
        state = np.reshape(state, [1, 8])
        dtest = xgb.DMatrix(state)
        pred = self.model.predict(dtest)
        # print(state.shape, state, 'prediction:', pred)
        return pred[0]  # returns action

    def load(self, name):
        self.model.load_model(name+'.raw.txt')

    def save(self, name):
        self.model.dump_model(name+'.raw.txt', name+'.featmap.txt')

    def print(self):
        pass  # TODO interesting to see what it learned.


env = gym.make('gym_cache:Cache-v0')
env.set_actor_name('XGB')

state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = XGBagent()
# agent.load("./save/cache-XGB_" + name)
accesses = 20000
# accesses = 1000000
# accesses = 3090162


name = "acc{}".format(accesses)
m_rew = []  # to monitor
m_act = []

state = env.reset()

for time in range(accesses):
    # env.render()

    action = agent.act(state)
    next_state, reward, done, _ = env.step(action)
    if done:
        break
    agent.memorize(state, action, reward)
    m_rew.append(reward)
    m_act.append(action)

    # next_state = np.reshape(next_state, [1, state_size])
    state = next_state

    if not time % 100:
        m_df = pd.DataFrame({"reward": m_rew, "actions": m_act})
        print("access: {}, reward: {:.2f}, actions: {:.2f}".format(
            time, m_df.reward.mean(), m_df.actions.mean()))

#     if e % 10 == 0:
agent.save("/save/cache-XGB_" + name)

env.close()

# r1 = pd.DataFrame.from_dict(inepi_rew[0])
# ax = r1.plot()
# ax.grid(True)
# ax.figure.savefig('results/plots/' + name + '.png')
