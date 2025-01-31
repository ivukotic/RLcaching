import random
import gym
import numpy as np
import pandas as pd
from collections import deque
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K

import tensorflow as tf


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    """Huber loss for Q Learning
    References: https://en.wikipedia.org/wiki/Huber_loss
                https://www.tensorflow.org/api_docs/python/tf/losses/huber_loss
    """

    def _huber_loss(self, y_true, y_pred, clip_delta=1.0):
        error = y_true - y_pred
        cond = K.abs(error) <= clip_delta

        squared_loss = 0.5 * K.square(error)
        quadratic_loss = 0.5 * \
            K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)

        return K.mean(tf.where(cond, squared_loss, quadratic_loss))

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss=self._huber_loss,
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                # a = self.model.predict(next_state)[0]
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * np.amax(t)
                # target[0][action] = reward + self.gamma * t[np.argmax(a)]
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model = keras.models.load_model(name)

    def save(self, name):
        self.model.save(name)


env = gym.make('gym_cache:Cache-v0')
env.set_actor_name('DDQN')

state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)
# agent.load("./save/cache-ddqn_" + name)
EPISODES = 1
replay_batch_size = 64
updata_steps = 128

accesses = 20000
# accesses = 1000000
# accesses = 3090162


name = "E{}_b{}_acc{}".format(EPISODES, replay_batch_size, accesses)
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
