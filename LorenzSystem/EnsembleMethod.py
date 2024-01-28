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
initial_conditions = [8, 7, 15];
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
feature_names = ["x", "y", "z"]
opt = ps.STLSQ(threshold = 0.5) #Sequential Treshold Least Square Algorithm -> Threshold Defined Randomly
model = ps.SINDy(feature_names = feature_names, optimizer = opt)
model.fit(x_noisy, t = dt)
model.print()

#System Output
# (x)' = -9.343 x + 9.450 y
# (y)' = 25.054 x + -0.932 x z
# (z)' = -0.866 1 + -2.595 z + 0.987 x y
# With noisy data, SINDy quite differs from original system.

model.fit(x_noisy, t = dt, ensemble = True)
mean_ensemble = np.mean(model.coef_list, axis = 0)
std_ensemble = np.std(model.coef_list, axis = 0)

model.fit(x_noisy, t=dt, library_ensemble = True)
mean_library_ensemble = np.mean(model.coef_list, axis = 0)
std_library_ensemble = np.std(model.coef_list, axis = 0)

LorenzFunctions.plot_ensemble_results(model, mean_ensemble, 
                                      std_ensemble, mean_library_ensemble, 
                                      std_library_ensemble)

# Ensemble method increase the roboustness of SINDy. It takes all of the trainging data and subsample it for generating different 
# multiple models. Library_ensamble is quite similar method to increase roboutness. It is only subsample candidate future library 
# (x, y, z for lorenz). It takes one of the feature in feature library and eliminate others and just fit the model on extracted future.
# This is sometimes problematic due to elimination of other futures. Results can seen from plot_ensemble_results.