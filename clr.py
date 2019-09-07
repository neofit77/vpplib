# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 17:43:44 2019

@author: patri
"""
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
from tqdm import tqdm
import VPPGym as ems_env
import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Dense, Lambda, GaussianNoise
from keras_layer_normalization import LayerNormalization
from keras.optimizers import SGD

PRINT_EVERY_X_ITER = 5
EPISODES = 5000
EP_LEN = 480
BATCH_SIZE = 480
WEIGHTS_PATH = None

class crl():
    
    def __init__(self):
        #environment variables
        self.state_size = 3
        self.state_dim = (self.state_size,)
        self.actions = 3
        self.action_dim = 1
        
        #network variables
        self.nodes = 12
        self.layers = 2
        self.learning_rate = 0.0001
        self.tau = 0.01 #0.0005
        self.loss = self._huber_loss
        self.target_std = 0.3 #0.1
        self.std = self.target_std
        self.std_var = K.variable(value = self.std)
        self.actor_perturbed = self.network_perturbed()
        self.actor_unperturbed = self.network_unperturbed()
        self.actor_target = self.network_unperturbed()
        self.memory = deque(maxlen=20000)
        
        #helper
        self.high_score = 0
        
    def load_weights(self, name):
        if name == None: return print("No weights loaded")
        try: self.actor_target.load_weights(f"name")
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
                
    def network_perturbed(self):
        inp = Input((self.state_dim))
        for _ in range(self.layers):
            x = Dense(self.nodes, activation='relu')(inp)
            x = GaussianNoise(self.std_var)(x, training = True)
            x = LayerNormalization()(x)
        out = Dense(self.actions, activation='linear')(x)
        #out = GaussianNoise(self.std_var)(out, training = True)
        #out = Lambda(lambda i: i * self.action_dim)(out)
        M = Model(inp, out)
        M.compile(optimizer = SGD(self.learning_rate, momentum = 0.9),loss=self.loss)
        return M

    def network_unperturbed(self):
        inp = Input((self.state_dim))
        for _ in range(self.layers):
            x = Dense(self.nodes, activation='relu')(inp)
            x = LayerNormalization()(x)
        out = Dense(self.actions, activation='linear')(x)
        #out = Lambda(lambda i: i * self.action_dim)(out)
        M = Model(inp, out)
        M.compile(optimizer = SGD(self.learning_rate, momentum = 0.9),loss=self.loss)
        return M
    
    def train(self, batch):
        states = np.stack(batch[:,0])
        actions = np.stack(batch[:,1])
        rewards = np.stack(batch[:,2])
        targets = self.actor_target.predict(states)
        for target, action, reward in zip(targets, actions, rewards):
            target[action] = reward
        self.actor_unperturbed.fit(np.squeeze(states), np.squeeze(targets), verbose = 0)
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
        
    def update_actor_target(self): 
        #NOT IN USE
        weights = self.actor_perturbed.get_weights()
        self.actor_target.set_weights(weights)
    
    def soft_update_actor_target(self):
        weights, target_weights = self.actor_unperturbed.get_weights(), self.actor_target.get_weights()        
        for i, weight in enumerate(weights):
            target_weights[i] = weight * self.tau + target_weights[i] * (1 - self.tau) 
        self.actor_target.set_weights(target_weights)
    
    def plot_test(self, LOGFILE = False):
        test_env = ems_env.ems(EP_LEN)      
        state = test_env.reset()
        test_env.time = 20000
        log, soc = [], []
        cum_r = 0
        for i in range(960):
            action = self.actor_target.predict(np.expand_dims(state, axis = 0))
            a = np.argmax(action)
            state, r, done, _ = test_env.step(a) 
            log.append([action, state[0], state[1], state[2], r])
            soc.append(state[0])
            cum_r += r
        tqdm.write(f" Current weights achieve a score of {cum_r}")
        if cum_r > self.high_score:
            self.high_score = cum_r
            self.actor_target.save_weights(f"high_score_weights_{cum_r}.h5")
        pd.DataFrame(soc).plot()
        plt.show()
        plt.close()
        if LOGFILE:
            xls = pd.DataFrame(log)
            xls.to_excel("results_log_ddpg.xls")

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
        action = actor.actor_perturbed.predict(np.expand_dims(state, axis = 0))
        a = np.argmax(action)
        state, r, done, _ = env.step(a)        
        cumul_r += r
        ep_r += r
        actor.memory.append([prior_state, a , r])        
        if len(actor.memory) > BATCH_SIZE:
            batch = np.array(random.sample(actor.memory, BATCH_SIZE))    
            actor.train(batch)
            actor.soft_update_actor_target()
    tqdm.write(f"--------------------------\n Episode: {ep+1}/{EPISODES} \n Cumulative Reward: {cumul_r} \n Episodic Reward: {ep_r}\n Current Std: {actor.std}")
    if not (ep+1) % PRINT_EVERY_X_ITER:
        actor.plot_test()
    
    
