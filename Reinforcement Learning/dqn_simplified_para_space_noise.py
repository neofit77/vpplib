# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 17:43:44 2019

@author: Patrick Lehnen
"""
import random
import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
from tqdm import tqdm
import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Dense, GaussianNoise
from keras_layer_normalization import LayerNormalization
from keras.optimizers import SGD

"""
This implementation uses epsilon greedy + parameter space noise as 
exploration strategy, as I found parameter space noise perfomance to
dependent from weight initialization!
""" 

PRINT_EVERY_X_ITER = 1
EPISODES = 5000
EP_LEN = 480
BATCH_SIZE = 240
WEIGHTS_PATH = "dqn.h5"

class crl():
    
    def __init__(self):
        #environment variables
        self.state_size = 4
        self.state_dim = (self.state_size,)
        self.actions = 1
        self.action_dim = 1
        self.epsilon = 0.5
        self.epsilon_decay_rate = 0.97
        self.epsilon_min = 0.01
        self.gamma = 0.99
        
        #network variables
        self.nodes = 64
        self.layers = 3
        self.learning_rate = 0.0001
        self.tau = 0.01
        self.loss = self._huber_loss
        self.target_std = 0.2
        self.std = self.target_std
        self.std_var = K.variable(value = self.std)
        self.actor_perturbed = self.network_perturbed()
        self.actor_unperturbed = self.network_unperturbed()
        self.actor_target = self.network_unperturbed()
        self.memory = deque(maxlen=20000)
        
        #helper
        self.SAVE_HIGHSCORE = True
        self.high_score = 0
        
    def load_weights(self, name):
        if name == None: return print("No weights loaded")
        try: self.actor_target.load_weights(name)
        except: print("Loading weights caused an error!")
        self.best_weights = self.actor_target.get_weights()
        self.actor_perturbed.set_weights(self.best_weights)
        self.actor_unperturbed.set_weights(self.best_weights)
        
    def _huber_loss(self, y_true, y_pred, clip_delta=1.0):
        ### source: keon.io ###
        error = y_true - y_pred
        cond  = K.abs(error) <= clip_delta

        squared_loss = 0.5 * K.square(error)
        quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)
        
        return K.mean(tf.where(cond, squared_loss, quadratic_loss))
                
    def network(self):
        inp = Input((self.state_dim))
        x = Dense(self.nodes, activation='relu')(inp)
        x = GaussianNoise(self.std_var)(x, training = self.apply_noise)
        x = LayerNormalization()(x)
        for _ in range(self.layers - 1):
            x = Dense(self.nodes, activation='relu')(x)
            x = GaussianNoise(self.std_var)(x, training = True)
            x = LayerNormalization()(x)
        out = (Dense(self.actions, activation='linear')(x))
        M = Model(inp, out)
        M.compile(optimizer = SGD(self.learning_rate, momentum = 0.9),loss = self.loss)
        return M
    
    def train(self, batch):
        states = np.stack(batch[:,0])
        actions = np.stack(batch[:,1])
        rewards = np.stack(batch[:,2])
        next_states = np.stack(batch[:,3])
        dones = np.array(np.stack(batch[:,4]), dtype = "bool")
        targets = self.actor_unperturbed.predict(states)
        t = self.actor_target.predict(next_states)
        if len(targets[dones]) > 0: targets[dones, actions[dones]] = rewards[dones]
        targets[~dones, actions[~dones]] = rewards[~dones] + self.gamma * np.amax(t[~dones])
        self.actor_unperturbed.fit(states, targets, verbose = 0)
        weights = self.actor_unperturbed.get_weights()
        self.actor_perturbed.set_weights(weights)
        self.update_std(np.array(states))  
        
    def update_std(self, states):
        au = self.actor_unperturbed.predict(states)
        ap = self.actor_perturbed.predict(states)
        self.std_log = np.sqrt(np.mean(np.square(au - ap)))
        self.calc_adaptive_noise(self.std_log)
        
    def calc_adaptive_noise(self, std):
        if std > self.target_std: self.std /= 1.01
        else: self.std *= 1.01
        self.change_std(self.std)
        
    def change_std(self, std):
        K.set_value(self.std_var, std)
    
    def soft_update_actor_target(self):
        weights, target_weights = self.actor_unperturbed.get_weights(), self.actor_target.get_weights()        
        for i, weight in enumerate(weights):
            target_weights[i] = weight * self.tau + target_weights[i] * (1 - self.tau) 
        self.actor_target.set_weights(target_weights)
        
    def epsilon_greedy(self, action):
        if np.random.random() < self.epsilon:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay_rate
            return np.random.randint(self.actions)
        else:
            return action
    
#    def plot_test(self, LOGFILE = False):
#        test_env = gym.make("CartPole-v1")      
#        state = test_env.reset()
#        log, soc = [], []
#        cum_r = 0
#        for i in range(671):
#            action = self.actor_target.predict(np.expand_dims(state, axis = 0))
#            a = np.argmax(action)
#            state, r, done, _ = test_env.step(a) 
#            #log.append([action, state[0], state[1], state[2], r])
#            #soc.append(state[0])
#            cum_r += r
#        tqdm.write(f" Current weights achieve a score of {cum_r}")
#        if cum_r > self.high_score and self.SAVE_HIGHSCORE:
#            self.high_score = cum_r
#            self.actor_target.save_weights(f"high_score_dqn_{cum_r}.h5")
#        #pd.DataFrame(soc).plot()
#        #plt.show()
#        #plt.close()
#        if LOGFILE:
#            xls = pd.DataFrame(log)
#            xls.to_excel("results_log.xls")

if __name__ == "__main__":
    actor = crl()
    actor.load_weights(WEIGHTS_PATH)
    env = gym.make("CartPole-v1")
    cumul_r = 0
    for ep in tqdm(range(EPISODES)):
        done = False
        ep_r = 0
        state = env.reset()
        while not done:
            prior_state = state      
            action_output = actor.actor_perturbed.predict(np.expand_dims(state, axis = 0))
            action = actor.epsilon_greedy(np.argmax(action_output))
            state, reward, done, _ = env.step(action)      
            cumul_r += reward
            ep_r += reward
            actor.memory.append([prior_state, action , reward, state, done])
            batch = np.array(random.sample(actor.memory, min(BATCH_SIZE, len(actor.memory))))
            actor.train(batch)
            actor.soft_update_actor_target()
        tqdm.write(f"--------------------------\n Episode: {ep+1}/{EPISODES} \n Epsilon: {actor.epsilon} \n Cumulative Reward: {cumul_r} \n Episodic Reward: {ep_r}\n Current Std: {actor.std}")
#        if not (ep+1) % PRINT_EVERY_X_ITER:
#            actor.plot_test()