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
                                    rmse/50.0,
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

#LorenzFunctions.plot_data_and_derivative(x_noisy, dt, ps.FiniteDifference()._differentiate)

#First figure shows us normal x data and second figure its derivative. You can see from second figure, noise amplifies due to integrator.
#We can solve this with better differentiator.

#LorenzFunctions.plot_data_and_derivative(x_noisy, dt, ps.SmoothedFiniteDifference()._differentiate)

#This results are much smoother than the first differentiator.

n_trajectories = 30
x0s = (np.random.rand(n_trajectories,3)-0.5)*20
x_train_multi = []

for i in range(n_trajectories):
    x_train_temp = solve_ivp(LorenzSys.lorenz_equations, t_span, x0s[i], t_eval = t_train, **LorenzFunctions.integrator_keywords).y.T
    x_train_temp_noisy = Noisy_Data_Generation(x_train_temp)
    x_train_multi.append(x_train_temp_noisy)

features_name = ["x", "y", "z"]
model = ps.SINDy(feature_names=features_name, optimizer=ps.STLSQ(threshold=0.1))
model.fit(x_train_multi, t=dt, multiple_trajectories=True)
model.print()