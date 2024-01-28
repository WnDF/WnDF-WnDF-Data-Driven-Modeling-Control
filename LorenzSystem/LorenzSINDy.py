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
                                    rmse/10.0,
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
t_test_span = (0,15)
dt = 0.002
t_test = np.arange(t_span[0],t_span[-1],dt)
x0_test = np.array([8,7,15]);
x_test = solve_ivp(LorenzSys.lorenz_equations, t_test_span, x0_test, t_eval = t_test, **LorenzFunctions.integrator_keywords).y.T

#Train Model with Noisy Data for different HyperParameters
threshold_scan = np.linspace(0, 1.0, 10)
coefs = []
x_noisy = Noisy_Data_Generation(x_train)
for i, threshold in enumerate(threshold_scan):
    sparse_regression_optimizer = ps.STLSQ(threshold=threshold)
    model = ps.SINDy(feature_names=features_name, 
                     optimizer=sparse_regression_optimizer)
    model.fit(x_noisy, t=dt, quiet=True)
    coefs.append(model.coefficients())
    
LorenzFunctions.plot_pareto(coefs, sparse_regression_optimizer, model, 
            threshold_scan, x_test, t_test)

#You can see the RMSE with respect to given threshold lambda. Lamba can choose with respect to 
#RMSE value as seen in chart. However, we can also see the direct model training with SINy is not robust with noisy data. 
#Anathoer SINDy technics should be consider.