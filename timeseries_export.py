# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 15:50:38 2019

@author: pyosch
"""

import pandas as pd
import matplotlib.pyplot as plt

from model.VPPBEV import VPPBEV
from model.VPPPhotovoltaic import VPPPhotovoltaic
from model.VPPHeatPump import VPPHeatPump

start = '2017-06-01 00:00:00'
end = '2017-06-07 23:45:00'
time_freq = "15 min"
scenario = 1

df_timeseries = pd.DataFrame(index = pd.date_range(start=start, end=end, 
                                         freq=time_freq, name ='Time'))

#%% baseload

#input data
baseload = pd.read_csv("./Input_House/Base_Szenario/df_S_15min.csv")
baseload.set_index(baseload.Time, inplace = True)
baseload.drop(labels="Time", axis=1, inplace = True)

df_timeseries["baseload"] = pd.DataFrame(baseload['0'].loc[start:end]/1000)

df_timeseries.baseload.plot(figsize=(16,9), label="baseload")

#%% PV
latitude = 50.941357
longitude = 6.958307
name = 'Cologne'

weather_data = pd.read_csv("./Input_House/PV/2017_irradiation_15min.csv")
weather_data.set_index("index", inplace = True)

pv = VPPPhotovoltaic(timebase=1, identifier=name, latitude=latitude, longitude=longitude, environment = None, userProfile = None,
                     start = start, end = end,
                     module_lib = 'SandiaMod', module = 'Canadian_Solar_CS5P_220M___2009_', 
                     inverter_lib = 'cecinverter', inverter = 'ABB__MICRO_0_25_I_OUTD_US_208_208V__CEC_2014_',
                     surface_tilt = 20, surface_azimuth = 200,
                     modules_per_string = 1, strings_per_inverter = 1)

df_timeseries["pv"] = pv.prepareTimeSeries(weather_data=weather_data)

df_timeseries.pv.plot(figsize=(16,9), label="PV")

#%% BEV
bev = VPPBEV(timebase=15/60, identifier='bev_1', 
             start = start, end = end, time_freq = time_freq, 
             battery_max = 16, battery_min = 0, battery_usage = 1, 
             charging_power = 11, chargeEfficiency = 0.98, 
             environment=None, userProfile=None)

def get_at_home(bev):
    
    weekend_trip_start = ['08:00:00', '08:15:00', '08:30:00', '08:45:00', 
                          '09:00:00', '09:15:00', '09:30:00', '09:45:00',
                          '10:00:00', '10:15:00', '10:30:00', '10:45:00', 
                          '11:00:00', '11:15:00', '11:30:00', '11:45:00', 
                          '12:00:00', '12:15:00', '12:30:00', '12:45:00', 
                          '13:00:00']
    
    weekend_trip_end = ['17:00:00', '17:15:00', '17:30:00', '17:45:00', 
                        '18:00:00', '18:15:00', '18:30:00', '18:45:00', 
                        '19:00:00', '19:15:00', '19:30:00', '19:45:00', 
                        '20:00:00', '20:15:00', '20:30:00', '20:45:00', 
                        '21:00:00', '21:15:00', '21:30:00', '21:45:00', 
                        '22:00:00', '22:15:00', '22:30:00', '22:45:00', 
                        '23:00:00']
    
    work_start = ['07:00:00', '07:15:00', '07:30:00', '07:45:00', 
                  '08:00:00', '08:15:00', '08:30:00', '08:45:00', 
                  '09:00:00']
    
    work_end = ['16:00:00', '16:15:00', '16:30:00', '16:45:00', 
                '17:00:00', '17:15:00', '17:30:00', '17:45:00', 
                '18:00:00', '18:15:00', '18:30:00', '18:45:00', 
                '19:00:00', '19:15:00', '19:30:00', '19:45:00', 
                '20:00:00', '20:15:00', '20:30:00', '20:45:00', 
                '21:00:00', '21:15:00', '21:30:00', '21:45:00', 
                '22:00:00']
    
    bev.timeseries = pd.DataFrame(index = pd.date_range(start=bev.start, end=bev.end, 
                                         freq=bev.time_freq, name ='Time'))
    bev.split_time() 
    bev.set_weekday()
    bev.set_at_home(work_start, work_end, weekend_trip_start, weekend_trip_end)
    
    return bev.at_home
    
df_timeseries["at_home"] = get_at_home(bev)

df_timeseries.at_home.plot(figsize=(16,9), label="at home")


#%% heat_pump

hp = VPPHeatPump(identifier = "House 1", timebase = 1, heatpump_type = "Air", 
                 heat_sys_temp = 60, environment = None, userProfile = None, 
                 heatpump_power = 10.6, full_load_hours = 2100, heat_demand_year = None,
                 building_type = 'DE_HEF33', start = start,
                 end = end, year = '2017')

if scenario == 1:
    
    df_timeseries["heat_pump"] = hp.prepareTimeSeries().el_demand
    df_timeseries.heat_pump.plot(figsize=(16,9), label="heat pump")
    
elif scenario == 2:
    
    df_timeseries["heat_demand"] = hp.get_heat_demand()
    df_timeseries["cop"] = hp.get_cop()
    df_timeseries.cop.interpolate(inplace = True)
    
    df_timeseries.heat_demand.plot(figsize=(16,9), label="heat demand")
    df_timeseries.cop.plot(figsize=(16,9), label="cop")
    
else:
    print("Heat pump scenario ", scenario, " not defined")
    
    

    
plt.legend()