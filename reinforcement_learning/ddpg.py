# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 17:43:44 2019

@author: Patrick Lehnen
"""
import random
import math
import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
from tqdm import tqdm
import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Dense, GaussianNoise, concatenate
from keras_layer_normalization import LayerNormalization
from keras.optimizers import SGD, Adam

PRINT_EVERY_X_ITER = 10
EPISODES = 5000
EP_LEN = 480
BATCH_SIZE = 96
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
#        self.atoms = 51 
#        self.r_max = 5
#        self.r_min = -1.5
#        self.delta_r = (self.r_max - self.r_min) / float(self.atoms - 1)
#        self.z = [self.r_min + i * self.delta_r for i in range(self.atoms)]
        self.epsilon = 0.0
        self.epsilon_decay_rate = 0.999
        self.epsilon_min = 0.00
        self.gamma = 0.99
        
        #environment variables
        self.state_size = 2
        self.state_dim = (self.state_size,)
        self.actions = 1
        
        #network variables
        self.nodes = 12
        self.layers = 2
        self.learning_rate = 0.01
        self.tau = 0.01
        self.target_std = 0.2
        self.std = self.target_std
        self.std_var = K.variable(value = self.std)
        self.actor_perturbed = self.network_perturbed()
        self.actor_unperturbed = self.network_unperturbed()
        self.actor_target = self.network_unperturbed()
        self.critic = self.network_critic()
        self.critic_target = self.network_critic() 
        self.memory = deque(maxlen=20000)
        
        #helper
        self.SAVE_HIGHSCORE = False
        self.high_score = 0
        self.loss = []
        self.calculate_gradients = K.function(self.critic.input, K.gradients(self.critic.output, self.critic.input[1]))
        self.action_gradients = K.placeholder([None, self.actions])
        self.parameter_gradients = tf.gradients(
                                    self.actor_unperturbed.output, 
                                    self.actor_unperturbed.trainable_weights, 
                                    -self.action_gradients)
        self.gradients = zip(self.parameter_gradients, self.actor_unperturbed.trainable_weights)
        self.optimize = K.function([self.actor_unperturbed.input, self.action_gradients], outputs = [], updates = [tf.train.AdamOptimizer(self.learning_rate).apply_gradients(self.gradients)])
        
    def load_weights(self, name):
        if name == None: return print("No weights loaded")
        try: self.actor_target.load_weights(name)
        except: print("Loading weights caused an error!")
        self.best_weights = self.actor_target.get_weights()
        self.actor_perturbed.set_weights(self.best_weights)
        self.actor_unperturbed.set_weights(self.best_weights)
    
    def network_critic(self):
        state = Input((self.state_dim))
        action = Input((self.actions,))
        x = Dense(self.nodes, activation='relu')(state)
        x = concatenate([x, action])
        x = LayerNormalization()(x)
        for _ in range(self.layers - 1):
            x = Dense(self.nodes, activation='relu')(x)
            x = LayerNormalization()(x)
        out = Dense(1, activation = 'linear')(x)
        M = Model([state, action], out)
        M.compile(optimizer = Adam(self.learning_rate), loss = "MSE")
        return M

###! implement with training bool as K.variable!            
    def network_perturbed(self):
        inp = Input((self.state_dim))
        x = Dense(self.nodes, activation = 'relu')(inp)
        x = GaussianNoise(self.std_var)(x, training = True)
        x = LayerNormalization()(x)
        for _ in range(self.layers - 1):
            x = Dense(self.nodes, activation = 'relu')(x)
            x = GaussianNoise(self.std_var)(x, training = True)
            x = LayerNormalization()(x)
        out = Dense(self.actions, activation = 'tanh')(x)
        M = Model(inp, out)
        return M

    def network_unperturbed(self):
        inp = Input((self.state_dim))
        x = Dense(self.nodes, activation = 'relu')(inp)
        x = LayerNormalization()(x)
        for _ in range(self.layers - 1):
            x = Dense(self.nodes, activation = 'relu')(x)
            x = LayerNormalization()(x)
        out = Dense(self.actions, activation = 'tanh')(x)
        M = Model(inp, out)
        return M
    
    def train(self, batch):
        ###! impl actor target ###
        states = np.stack(batch[:,0])
        actions = np.stack(batch[:,1])
        rewards = np.stack(batch[:,2])
        next_states = np.stack(batch[:,3])
        dones = np.array(np.stack(batch[:,4]), dtype = "bool")
        new_actions = self.actor.predict(states)
        targets
        #targets = self.critic.predict([states, actions])
        t = self.critic_target.predict([next_states, new_actions])
        if len(targets[dones]) > 0: targets[dones] = np.expand_dims(rewards[dones], axis = 1)
        targets[~dones] = np.expand_dims(rewards[~dones] + self.gamma * np.amax(t[~dones]), axis = 1)
        hist = self.critic.fit([states, actions], targets, verbose = 0)
        
        self.loss.append(hist.history["loss"])           
        grads = self.calculate_gradients([states, actions])
        assert any(np.isnan(grads[0][0])) == False
        self.optimize([states, grads])  
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
    
    def soft_update_critic_target(self):
        weights, target_weights = self.critic.get_weights(), self.critic_target.get_weights()        
        for i, weight in enumerate(weights):
            target_weights[i] = weight * self.tau + target_weights[i] * (1 - self.tau) 
        self.critic_target.set_weights(target_weights)
        
    def epsilon_greedy(self, action):
        if np.random.random() < self.epsilon:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay_rate
            return np.random.random(self.actions) * 2 - 1
        else:
            return action
        
    def plot_test(self, LOGFILE = False):
        test_env = ems_env.ems(EP_LEN)      
        state = test_env.reset()
        test_env.time = 20000
        log, soc = [], []
        cum_r = 0
        for i in range(960):
            
            action = agent.actor_perturbed.predict(np.expand_dims(state, axis = 0))[0]
            state, r, done, _ = test_env.step(action) 
            log.append([action, state[0], state[1], state[2], r])
            soc.append(state[0])
            cum_r += r
        tqdm.write(f" Current weights achieve a score of {cum_r}")
        if cum_r > self.high_score and self.SAVE_HIGHSCORE:
            self.high_score = cum_r
            self.actor_unperturbed.save_weights(f"high_score_weights_{cum_r}.h5")
        pd.DataFrame(soc).plot()
        pd.DataFrame(self.loss).plot(title = "critic loss")
        plt.show()
        plt.close()
        if LOGFILE:
            xls = pd.DataFrame(log)
            xls.to_excel("results_log_ddpg.xls")

if __name__ == "__main__":
    agent = crl() 
    
    #DEBUG FUNCTION
    self = agent    
    
    agent.load_weights(WEIGHTS_PATH)
    env = gym.make("MountainCarContinuous-v0")
#    env = gym.make("Pendulum-v0")
    cumul_r = 0
    for ep in tqdm(range(EPISODES)):
        done = False
        ep_r = 0
        state = env.reset()
        while not done:
            env.render()
            prior_state = state      
            action = agent.epsilon_greedy(
                    agent.actor_perturbed.predict(np.expand_dims(state, axis = 0))[0])#agent.epsilon_greedy(agent.calc_action(np.expand_dims(state, axis = 0)))
            state, reward, done, _ = env.step(action)        
            cumul_r += reward
            ep_r += reward
            agent.memory.append([prior_state, action, reward, state, done])
            batch = np.array(random.sample(agent.memory, min(BATCH_SIZE, len(agent.memory))))    
            agent.train(batch)
            agent.soft_update_actor_target()
            agent.soft_update_critic_target()
        tqdm.write(f"\n--------------------------\n Episode: {ep+1}/{EPISODES} \n Epsilon: {np.round(agent.epsilon, 2)} \n Cumulative Reward: {cumul_r} \n Episodic Reward: {ep_r}\n Current Std: {agent.std}")
#        if not (ep+1) % PRINT_EVERY_X_ITER:
#            agent.plot_test()