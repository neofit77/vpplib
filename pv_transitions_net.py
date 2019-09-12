import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from tqdm import tqdm

import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Dense, GaussianNoise
from keras_layer_normalization import LayerNormalization
from keras.optimizers import SGD
import math
###!IMPLEMENT FORECASTS###
atoms = 11 

def network():
    inp = Input((4,))
    x = Dense(12, activation='relu')(inp)
    x = LayerNormalization()(x)
    for _ in range(2 - 1):
        x = Dense(12, activation='relu')(x)
        x = LayerNormalization()(x)
    out = (Dense(atoms, activation='softmax')(x))
    M = Model(inp, out)
    M.compile(optimizer = SGD(0.0001, momentum = 0.9), loss = "categorical_crossentropy")
    return M

predictor = network()
data = pd.read_csv("./Input_House/PV/Solar_Data-2011.csv", delimiter = ";")#, skiprows = 0)
day_sin = []
for i, rad in enumerate(data["Generation"]):
    sin_day = np.sin(2*np.pi*i/(24*4))
    cos_day = np.cos(2*np.pi*i/(24*4))
    sin_year = np.sin(2*np.pi*i/(24*4*365))
    cos_year = np.cos(2*np.pi*i/(24*4*365))
    day_sin.append([sin_day, cos_day, sin_year, cos_year])

df = pd.DataFrame(day_sin)
df.plot(subplots = True)#

r_max = 0.9
r_min = -0.1
delta_r = (r_max - r_min) / float(atoms - 1)
z = [r_min + i * delta_r for i in range(atoms)]


m_prob = np.zeros((35040, atoms))
for i, reward in enumerate(data["Generation"]):
    Tz = min(r_max, max(r_min, reward))
    bj = (Tz - r_min) / delta_r 
    m_l, m_u = math.floor(bj), math.ceil(bj)
    m_prob[i][int(m_l)] += (m_u - bj)
    m_prob[i][int(m_u)] += (bj - m_l)
predictor.fit(df, m_prob, epochs = 150, verbose = 1)

for j in range(96):
    a = predictor.predict(df.iloc[(96*90+j):(96*90+j+1)])
    d = pd.DataFrame(a)
    d.T.plot.bar(subplots = True, title = int(j/4))#
plt.close()
