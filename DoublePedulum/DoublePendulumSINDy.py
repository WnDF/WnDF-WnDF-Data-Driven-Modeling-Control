import numpy as np
import pysindy as ps
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Define parameters
dt = 0.001
T = 10
t_train = np.arange(0, T, dt)
x0_train = [np.pi + 0.3, np.pi - 0.5, 0, 0]
t_train_span = (t_train[0], t_train[-1])

# Generate training data using pysindy.utils.double_pendulum
integrator_keywords = {
    'rtol': 1e-12,
    'method': 'LSODA',
    'atol': 1e-12
}
x_train = solve_ivp(
    ps.utils.double_pendulum, t_train_span, x0_train, t_eval=t_train, **integrator_keywords
).y.T

feature_names = ["Theta", "Theta*", "Omega", "Omega*"]
opt = ps.STLSQ(threshold = 0.27)
model = ps.SINDy(feature_names = feature_names, optimizer = opt, differentiation_method = ps.SmoothedFiniteDifference()._differentiate)
print("Training model...")
model.fit(x_train, t = dt, ensemble=True, quiet=True)
model.print()
print("")
out = model.simulate(x_train[0, :], t_train, integrator = "odeint")

fig1 = plt.figure(figsize = (35,12))
fig1ax1 = fig1.add_subplot(2,2,1)
fig1ax2 = fig1.add_subplot(2,2,2)
fig1ax3 = fig1.add_subplot(2,2,3)
fig1ax4 = fig1.add_subplot(2,2,4)

fig1ax1.plot(t_train, x_train[:,0], color = 'red', label = 'Act.')
fig1ax1.plot(t_train, out[:,0], color = 'blue', label = 'Pred.')
fig1ax1.set_title("Theta", fontweight="bold")
fig1ax1.legend(loc = 'upper right')

fig1ax2.plot(t_train, x_train[:,1], color = 'red', label = 'Act.')
fig1ax2.plot(t_train, out[:,1], color = 'blue', label = 'Pred.')
fig1ax2.set_title("Theta*", fontweight="bold")
fig1ax2.legend(loc = 'upper right')

fig1ax3.plot(t_train, x_train[:,2], color = 'red', label = 'Act.')
fig1ax3.plot(t_train, out[:,2], color = 'blue', label = 'Pred.')
fig1ax3.set_title("Omega", fontweight="bold")
fig1ax3.legend(loc = 'upper right')

fig1ax4.plot(t_train, x_train[:,3], color = 'red', label = 'Act.')
fig1ax4.plot(t_train, out[:,3], color = 'blue', label = 'Pred.')
fig1ax4.set_title("Omega*", fontweight="bold")
fig1ax4.legend(loc = 'upper right')

plt.show()