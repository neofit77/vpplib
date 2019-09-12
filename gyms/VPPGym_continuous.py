# -*- coding: utf-8 -*-
"""
Created on Thu May  2 12:13:59 2019

@author: patri
"""


import numpy as np
import pandas as pd
from gym import spaces
import gym
from collections import deque
from model.VPPPhotovoltaic import VPPPhotovoltaic
from model.VPPEnergyStorage import VPPEnergyStorage
from model.VPPHousehold import VPPHousehold

class ems(gym.Env):
               
    """
        Info
        ----
        The ems-class is the training environment for the DDQN-Agent. 
        
        Parameters
        ----------
        Under __init__ 
        ...
        	
        Attributes
        ----------
        
        ...
        
        Notes
        -----
        The offset variable takes into account that there are nearly no days with more PV-production than electricity demand. 
        ...
        
        References
        ----------
        
        ...
        
        Returns
        -------
        
        ...
        
    """

    def __init__(self, EP_LEN):
        super(ems, self).__init__()
        self.LOG_EVENTS = False
        self.EP_LEN = EP_LEN
        self.obs = 3
        self.offset = 24 * 4 * 7 * 13 
        self.observation_space = spaces.Box(low = -5, high = 5, shape = (self.obs,), dtype=np.float32)
        self.action_space = spaces.Box(low = -1, high = 1, shape = (2,), dtype = np.float32)#spaces.Continuous(3)
        self.time = 0 + self.offset
        self.residual = 0
        self.max_pv = 0
        self.r = 0
        self.log = deque(maxlen=self.EP_LEN)
        self.pv = self.prepareTimeSeriesPV()
        self.loadprofile = VPPHousehold(15, None, None)
        self.day_ind = self.day_sin()
        self.max_lp = max(self.loadprofile.data)
        
    def prepareTimeSeriesPV(self):
        """
        Info
        ----
        This method initializes the PV-Lib module with its necessary parameters.
        
        Parameters
        ----------
        
        ...
        	
        Attributes
        ----------
        
        ...
        
        Notes
        -----
        
        ...
        
        References
        ----------
        
        ...
        
        Returns
        -------
        A DataFrame Object with the 
        ...
        
        """
        latitude = 50.941357
        longitude = 6.958307
        name = 'Cologne'
        
        weather_data = pd.read_csv("./Input_House/PV/2017_irradiation_15min.csv")
        weather_data.set_index("index", inplace = True)
        
        pv = VPPPhotovoltaic(timebase=1, identifier=name, latitude=latitude, longitude=longitude, modules_per_string=5, strings_per_inverter=1)
        pv.prepareTimeSeries(weather_data)
        return pv
        
    def reset(self):
        """
        Info
        ----
        This method resets all necessary parameters for a new episode of the training.
        
        Parameters
        ----------
        
        ...
        	
        Attributes
        ----------
        
        ...
        
        Notes
        -----
        
        ...
        
        References
        ----------
        
        ...
        
        Returns
        -------
        The starting state is an array of zeros, as the first observation for the agent.
        ...
        
        """
        self.rand_start = int(np.random.rand()*25000)+self.offset
        state = np.array(np.zeros(self.obs))
        self.time = self.rand_start
        self.residual = 0       
        self.el_storage = VPPEnergyStorage(15, 15, 0.9, 0.9, 15, 1)
        self.el_storage.prepareTimeSeries()
        self.soc = []
        return state
    
    def day_sin(self):
        day_sin = []
        for i in range(24*4*366):
            sin_day = np.sin(2*np.pi*i/(24*4))
            cos_day = np.cos(2*np.pi*i/(24*4))
            day_sin.append([sin_day, cos_day])
        return day_sin
    
    def size_of_charge(self, action_size):
        #This function ensures that an action of 0.8 is the maximum charge multiplier. Higher multipliers need significantly higher input values in the neural network, thus reducing learning efficiency. 
        if action_size > 0.8: action_size = 0.8
        return abs(self.residual)*1.25*action_size
    
    def step(self, action):
    #Actions:
    # 0 = nichts
    # 1 = Laden
    # 2 = Entladen   
        res_bool, is_valid_action, info = 0, False, 0
        #selected_action = np.argmax(action)
        charge_size_1 = self.size_of_charge(action[0])
        charge_size_2 = self.size_of_charge(action[1])
        #Action 0: Nichts
        #if selected_action == 0: r, is_valid_action = 0, True
        if charge_size_1 < 0 and charge_size_2 < 0: r, is_valid_action = 0, True
        #Action 1: Laden
        elif self.residual < 0 and charge_size_1 > 0 and charge_size_2 < 0:
            r = charge_size_1*3
            is_valid_action = self.el_storage.charge(charge_size_1, 15, self.time)
        #Action 2: Entladen
        elif self.residual > 0 and charge_size_2 > 0 and charge_size_1 < 0:
            r = charge_size_2*3
            is_valid_action = self.el_storage.discharge(charge_size_2, 15, self.time)

        
        #Fehler- und Rewardüberprüfung
        if not is_valid_action: r = -1
        #Bereite den nächsten state vor
        lp = self.loadprofile.valueForTimestamp(self.time)
        pv = self.pv.valueForTimestamp(self.time)*10
        self.residual = lp - pv
        if self.pv.valueForTimestamp(self.time) > self.max_pv: self.max_pv = self.pv.valueForTimestamp(self.time)

        res_bool = self.residual > 0     
        state = np.array([self.el_storage.stateOfCharge/self.el_storage.capacity, self.residual/self.max_lp, res_bool])
        
        #state = np.array([self.el_storage.stateOfCharge/self.el_storage.capacity, self.loadprofile.data[self.time]/self.max_lp, self.pv.data[self.time]/self.max_pv, res_bool])
        self.time += 1
        self.soc.append(self.el_storage.stateOfCharge)
        done = self.time >= self.rand_start + self.EP_LEN
        #needs rework
        if done:
            info = pd.DataFrame(self.soc)
        return state, r, done, info