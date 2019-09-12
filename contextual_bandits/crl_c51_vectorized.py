# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 17:43:44 2019

@author: Patrick Lehnen
"""
import random
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
from tqdm import tqdm
import VPPGym as ems_env
import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Dense, GaussianNoise
from keras_layer_normalization import LayerNormalization
from keras.optimizers import SGD

PRINT_EVERY_X_ITER = 1
EPISODES = 5000
EP_LEN = 480
BATCH_SIZE = 480
WEIGHTS_PATH = None

"""
This implementation uses epsilon greedy + parameter space noise as 
exploration strategy, as I found parameter space noise perfomance to
dependent from weight initialization!

Beware: if r_max + r_min equals 0, any 0 reward from the environment 
will not be regarded as (m_u - bj) equals to zero as well!
""" 

class crl():
    
    def __init__(self):        
        #C51
        self.atoms = 51 
        self.r_max = 2
        self.r_min = -1
        self.delta_r = (self.r_max - self.r_min) / float(self.atoms - 1)
        self.z = [self.r_min + i * self.delta_r for i in range(self.atoms)]
        self.epsilon = 0.5
        self.epsilon_decay_rate = 0.9995
        self.epsilon_min = 0.00
        
        #environment variables
        self.state_size = 3
        self.state_dim = (self.state_size,)
        self.actions = 3
        
        #network variables
        self.nodes = 12
        self.layers = 2
        self.learning_rate = 0.0001
        self.tau = 0.01
        self.target_std = 0.3
        self.std = self.target_std
        self.std_var = K.variable(value = self.std)
        self.actor_perturbed = self.network_perturbed()
        self.actor_unperturbed = self.network_unperturbed()
        self.actor_target = self.network_unperturbed()
        self.memory = deque(maxlen=20000)
        
        #helper
        self.SAVE_HIGHSCORE = False
        self.high_score = 0
              
    def load_weights(self, name):
        if name == None: return print("No weights loaded")
        try: self.actor_target.load_weights(name)
        except: print("Loading weights caused an error!")
        self.best_weights = self.actor_target.get_weights()
        self.actor_perturbed.set_weights(self.best_weights)
        self.actor_unperturbed.set_weights(self.best_weights)
                
    def network_perturbed(self):
        out = []
        inp = Input((self.state_dim))
        x = Dense(self.nodes, activation='relu')(inp)
        x = GaussianNoise(self.std_var)(x, training = True)
        x = LayerNormalization()(x)
        for _ in range(self.layers - 1):
            x = Dense(self.nodes, activation='relu')(x)
            x = GaussianNoise(self.std_var)(x, training = True)
            x = LayerNormalization()(x)
        for i in range(self.actions):
            out.append(Dense(self.atoms, activation='softmax')(x))
        M = Model(inp, out)
        M.compile(optimizer = SGD(self.learning_rate, momentum = 0.9), loss = "categorical_crossentropy")
        return M

    def network_unperturbed(self):
        out = []
        inp = Input((self.state_dim))
        x = Dense(self.nodes, activation='relu')(inp)
        x = LayerNormalization()(x)
        for _ in range(self.layers - 1):
            x = Dense(self.nodes, activation='relu')(x)
            x = LayerNormalization()(x)
        for i in range(self.actions):
            out.append(Dense(self.atoms, activation='softmax')(x))
        M = Model(inp, out)
        M.compile(optimizer = SGD(self.learning_rate, momentum = 0.9), loss = "categorical_crossentropy")
        return M
    
    def calc_action(self, state):
        ### source: flyyufelix ###
        z = self.actor_perturbed.predict(state)
        z_concat = np.vstack(z)
        q = np.sum(np.multiply(z_concat, np.array(self.z)), axis=1) 
        action_idx = np.argmax(q)        
        return action_idx
    
    def calc_unperturbed_action(self, state):
        ### source: flyyufelix ###
        z = self.actor_unperturbed.predict(state)
        z_concat = np.vstack(z)
        q = np.sum(np.multiply(z_concat, np.array(self.z)), axis=1) 
        action_idx = np.argmax(q)       
        return action_idx    
    
    def train(self, batch):
        states = np.stack(batch[:,0])
        actions = np.stack(batch[:,1])
        rewards = np.stack(batch[:,2])
        m_prob = np.zeros((self.actions, batch.shape[0], self.atoms))
        Tz = np.clip(rewards, self.r_min, self.r_max)
        bj = (Tz - self.r_min) / self.delta_r 
        m_l, m_u = np.floor(bj).astype("int32"), np.ceil(bj).astype("int32")
        index = np.arange(0, batch.shape[0]).astype("int32")
        where_equal = np.equal(m_l, m_u)
        if any(where_equal):
            m_prob[actions[where_equal], index[where_equal], m_l[where_equal]] = 1
        if any(~where_equal):
            m_prob[actions[~where_equal], index[~where_equal], m_l[~where_equal]] = (m_u - bj)[~where_equal] 
            m_prob[actions[~where_equal], index[~where_equal], m_u[~where_equal]] = (bj - m_l)[~where_equal]
        m_prob = [m_prob[0], m_prob[1], m_prob[2]]
        self.actor_unperturbed.train_on_batch(states, m_prob)
        weights = self.actor_unperturbed.get_weights()
        self.actor_perturbed.set_weights(weights)
        self.update_std(np.array(states))  
    
    def update_std(self, states):
        au = self.calc_action_list(self.actor_unperturbed.predict(states))
        ap = self.calc_action_list(self.actor_perturbed.predict(states))
        self.std_log = np.sqrt(np.mean(np.square(au - ap)))
        self.calc_adaptive_noise(self.std_log)
        
    def calc_action_list(self, z):
        z_concat = np.vstack(z)
        q = np.sum(np.multiply(z_concat, np.array(self.z)), axis=1) 
        q = q.reshape((min(len(self.memory), BATCH_SIZE), self.actions), order='F')
        return np.argmax(q, axis=1)
    
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
        
    def plot_test(self, LOGFILE = False):
        test_env = ems_env.ems(EP_LEN)      
        state = test_env.reset()
        test_env.time = 20000
        log, soc = [], []
        cum_r = 0
        for i in range(960):
            action = self.calc_unperturbed_action(np.expand_dims(state, axis = 0))
            state, r, done, _ = test_env.step(action) 
            log.append([action, state[0], state[1], state[2], r])
            soc.append(state[0])
            cum_r += r
        tqdm.write(f" Current weights achieve a score of {cum_r}")
        if cum_r > self.high_score and self.SAVE_HIGHSCORE:
            self.high_score = cum_r
            self.actor_target.save_weights(f"high_score_weights_{cum_r}.h5")
        pd.DataFrame(soc).plot()
        pd.DataFrame(np.squeeze(self.actor_target.predict(np.expand_dims(state, axis = 0)))).T.plot(kind = "bar", subplots = True)
        plt.show()
        plt.close()
        if LOGFILE:
            xls = pd.DataFrame(log)
            xls.to_excel("results_log_ddpg.xls")

if __name__ == "__main__":
    actor = crl()
    actor.load_weights(WEIGHTS_PATH)
    env = ems_env.ems(EP_LEN)
    cumul_r = 0
    for ep in tqdm(range(EPISODES)):
        done = False
        ep_r = 0
        state = env.reset()
        while not done:
            prior_state = state      
            action = actor.epsilon_greedy(actor.calc_action(np.expand_dims(state, axis = 0)))
            state, r, done, _ = env.step(action)        
            cumul_r += r
            ep_r += r
            actor.memory.append([prior_state, action, r]) 
            batch = np.array(random.sample(actor.memory, min(BATCH_SIZE, len(actor.memory))))    
            actor.train(batch)
            actor.soft_update_actor_target()
        tqdm.write(f"\n--------------------------\n Episode: {ep+1}/{EPISODES} \n Epsilon: {np.round(actor.epsilon, 2)} \n Cumulative Reward: {cumul_r} \n Episodic Reward: {ep_r}\n Current Std: {actor.std}")
        if not (ep+1) % PRINT_EVERY_X_ITER:
            actor.plot_test()