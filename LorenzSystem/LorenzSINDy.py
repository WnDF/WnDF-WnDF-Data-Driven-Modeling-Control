import pysindy as ps
from LorenzClass import LorenzSystem as LS
import numpy as np
from pysindy.utils import lorenz, lorenz_control, enzyme
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def Noisy_Data_Generation(x):
    rmse = mean_squared_error(x,
                            np.zeros(x.shape),
                            squared = False)
    x_noisy = x + np.random.normal(0, 
                                    rmse/10.0,
                                    x.shape)
    return x_noisy

#Define Lorenz System
LorenzSys = LS(sigma=10.0, rho = 28.0, beta = 8.0/3.0)
initial_conditions = [1 , 0, 0];
t_span = (0,25)
dt = 0.002

#Solve Lorenz System with LorenzClass
LorenzSys.set_initial_conditions(initial_conditions)
LorenzSys.set_time_span(t_span)
output = LorenzSys.solve_system(dt)
x_train = np.stack((output.y[0], output.y[1], output.y[2]), axis=-1)
#LorenzSys.plot_solution(x_train)

#Generate SINDy Model using LorenzClass Ouput to Estimate Lorenz System
features_name = ["x", "y", "z"]
opt = ps.STLSQ(threshold = 0.1) #Sequential Treshold Least Square Algorithm -> Threshold Defined Randomly
model = ps.SINDy(feature_names = features_name, optimizer = opt)
model.fit(x_train, t = dt)
model.print()

#Model Output
# (x)' = -9.982 x + 9.982 y
# (y)' = 27.645 x + -0.926 y + -0.989 x z
# (z)' = -2.658 z + 0.996 x y

#Effect of Hyperparameter Lambda
x_noisy = Noisy_Data_Generation(x_train)

#Train Model with Noisy Data for different HyperParameters
opt_noisy = ps.STLSQ(threshold = 0.7)
model_noisy = ps.SINDy(feature_names = features_name, optimizer = opt_noisy)
model_noisy.fit(x_noisy, t = dt)
model_noisy.print()

#Direct model training with SINy is not robust. Anathoer SINDy technics should be consider.