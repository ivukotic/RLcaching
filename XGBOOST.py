import random
import gym
import numpy as np
import pandas as pd
from collections import deque
import xgboost as xgb
from pathlib import Path
import matplotlib.pyplot as plt

# won't be needed for actual actor.
# here just in case I want to check its performance
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score

# have to transform input data into its own representation

model_folder = Path('save')
plots_folder = Path('results/plots')
feature_file = Path('data') / 'featuremap.txt'


class XGBagent:
    def __init__(self, batch_size=1000, window_size=50000):
        self.accesses = 0
        self.start_accesses = 0
        self.batch_size = batch_size  # retrain after adding each batch
        self.window_size = window_size  # keep this many last accesses
        # objectives: binary:logistic - returns probability of true
        self.param = {
            'eta': 0.2,  # learning rate
            'max_depth': 6,
            'objective': 'binary:hinge',
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
        state = np.reshape(state, [1, 8])
        dtest = xgb.DMatrix(state)
        pred = self.model.predict(dtest)
        # print(state.shape, state, 'prediction:', pred)
        if pred[0] > 0.5:
            return 1
        else:
            return 0

    def load(self):
        model_file = model_folder / \
            'XGB_{}.raw.txt'.format(self.start_accesses)
        self.model.load_model(model_file)

    def save(self):
        model_file = model_folder / 'XGB_{}.raw.txt'.format(self.accesses)
        model_bin = model_folder / 'XGB_{}.bin'.format(self.accesses)
        self.model.dump_model(model_file, feature_file,
                              with_stats=True, dump_format='json')
        self.model.save_model(model_bin)

    def plot(self):
        fig, ax = plt.subplots()
        # weight, gain, or cover
        # importance_type='weight'
        xgb.plot_importance(self.model, ax=ax, fmap=feature_file)
        plt.savefig(plots_folder /
                    'XGB_importance_{}.png'.format(self.accesses))

        fig, ax = plt.subplots(constrained_layout=True, figsize=(150, 100))
        xgb.plot_tree(self.model, ax=ax, fmap=feature_file, rankdir='LR')
        plt.savefig(plots_folder / 'XGB_tree_{}.png'.format(self.accesses))

    def print(self):
        pass  # TODO interesting to see what it learned.


env = gym.make('gym_cache:Cache-v0')

env.set_actor_name('XGB_20000')

accesses = 20000
# accesses = 100000
# accesses = 3090162

state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = XGBagent()

# agent.load()


name = "acc{}".format(accesses)
m_rew = []  # to monitor
m_act = []
s_rew = 0  # to print
s_act = 0
print_step = int(accesses/200)


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
    s_rew += reward
    s_act += action

    # next_state = np.reshape(next_state, [1, state_size])
    state = next_state

    if not time % print_step:
        print("access: {}, avg. reward: {:.2f}, actions: {:.2f}".format(
            time, s_rew/print_step, s_act/print_step))
        s_rew = 0
        s_act = 0

agent.save()
agent.plot()
env.close()


m_df = pd.DataFrame({"reward": m_rew, "actions": m_act})
# m_df['smoothed reward'] = m_df.reward.rolling(window=print_step).mean()

fig, ax = plt.subplots(3, 1, sharex=True, gridspec_kw={'hspace': 0.15})
ax[0].plot(m_df["reward"], label='reward')
ax[1].plot(m_df["reward"].rolling(window=print_step).mean(),
           label='rolling mean reward')
ax[2].plot(m_df["actions"].rolling(window=print_step).mean(),
           label='rolling mean action')
ax[0].grid(True)
ax[1].grid(True)
ax[2].grid(True)
ax[0].legend()
ax[1].legend()
ax[2].legend()
plt.savefig(plots_folder / 'rewards_{}.png'.format(name))
