import pysindy as ps
from LorenzClass import LorenzSystem as LS
import numpy as np
from pysindy.utils import lorenz, lorenz_control, enzyme
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import LorenzFunctions

def Noisy_Data_Generation(x):
    rmse = mean_squared_error(x,
                            np.zeros(x.shape),
                            squared = False)
    x_noisy = x + np.random.normal(0, 
                                    rmse/20.0,
                                    x.shape)
    return x_noisy

#Define Lorenz System
LorenzSys = LS(sigma=10.0, rho = 28.0, beta = 8.0/3.0)
initial_conditions = [1 , 0, 0];
t_span = (0,10)
dt = 0.002
t_train = np.arange(t_span[0],t_span[-1],dt)

#Solve Lorenz System with LorenzClass
LorenzSys.set_initial_conditions(initial_conditions)
LorenzSys.set_time_span(t_span)
output = LorenzSys.solve_system(dt)
x_train = np.stack((output.y[0], output.y[1], output.y[2]), axis=-1)
x_noisy = Noisy_Data_Generation(x_train)
#LorenzSys.plot_solution(x_train)

#Generate SINDy Model using LorenzClass Ouput to Estimate Lorenz System
features_name = ["x", "y", "z"]
opt = ps.STLSQ(threshold = 0.5) #Sequential Treshold Least Square Algorithm -> Threshold Defined Randomly
model = ps.SINDy(feature_names = features_name, optimizer = opt)
model.fit(x_noisy, t = dt)
model.print()