# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 15:33:53 2019

@author: patri, pyosch
"""
from vpplib.user_profile import UserProfile
from vpplib.environment import Environment
from vpplib.thermal_energy_storage import ThermalEnergyStorage
from vpplib.combined_heat_and_power import CombinedHeatAndPower
import matplotlib.pyplot as plt

figsize = (10,6)

start = '2015-01-01 00:00:00'
end = '2015-12-31 23:45:00'
year = '2015'
periods = None
time_freq = "15 min"

#Values for user_profile
yearly_heat_demand = 2500 # kWh
building_type = 'DE_HEF33'
t_0 = 40 # °C

#Values for Thermal Storage
target_temperature = 60 # °C
hysteresis = 5 # °K
mass_of_storage = 500 # kg

#Values for chp
el_power = 4 #kw
th_power = 6 #kW
overall_efficiency = 0.8
rampUpTime = 1/15 #timesteps
rampDownTime = 1/15 #timesteps
min_runtime = 1 #timesteps
min_stop_time = 2 #timesteps
timebase = 15


environment = Environment(timebase=timebase, start=start, end=end, year=year,
                          time_freq=time_freq)

user_profile = UserProfile(identifier=None,
                           latitude = None,
                           longitude = None,
                           yearly_heat_demand=yearly_heat_demand,
                           building_type = building_type,
                           comfort_factor = None,
                           t_0=t_0)

def test_get_heat_demand(user_profile):
    
    user_profile.get_heat_demand()
    user_profile.heat_demand.plot()
    plt.show()
    
test_get_heat_demand(user_profile)

tes = ThermalEnergyStorage(environment=environment,
                           user_profile=user_profile,
                           mass=mass_of_storage,
                           hysteresis=hysteresis,
                           target_temperature=target_temperature)

chp = CombinedHeatAndPower(unit="kW", identifier='chp1',
                           environment=environment,
                           user_profile=user_profile,
                           el_power=el_power,
                           th_power=th_power,
                           overall_efficiency=overall_efficiency,
                           ramp_up_time=rampUpTime,
                           ramp_down_time=rampDownTime,
                           min_runtime=min_runtime,
                           min_stop_time=min_stop_time)


for i in tes.user_profile.heat_demand.index:
    tes.operate_storage(i, chp)


tes.timeseries.plot(figsize=figsize, title="Yearly Temperature of Storage")
plt.show()
tes.timeseries.iloc[10000:10960].plot(figsize=figsize, title="10-Day View")
plt.show()
tes.timeseries.iloc[10000:10096].plot(figsize=figsize, title="Daily View")
plt.show()
chp.timeseries.el_demand.plot(figsize=figsize, title="Yearly Electrical Loadshape")
plt.show()
chp.timeseries.el_demand.iloc[10000:10960].plot(figsize=figsize, title="10-Day View")
plt.show()
chp.timeseries.el_demand.iloc[10000:10096].plot(figsize=figsize, title="Daily View")
plt.show()