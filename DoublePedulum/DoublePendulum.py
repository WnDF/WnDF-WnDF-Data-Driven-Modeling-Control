import numpy as np
from scipy.integrate import solve_ivp
import pysindy as ps

# Define the double pendulum ODE
def double_pendulum_ode(t, y, m1, m2, a1, a2, L1, L2, I1, I2, g, k1, k2):
    return [
        y[2],
        y[3],
        (
            L1 * a2 * 2 * g * m2 * 2 * np.sin(y[0])
            - 2 * L1 * a2 * 3 * y[3] ** 2 * m2 * 2 * np.sin(y[0] - y[1])
            + 2 * I2 * L1 * g * m2 * np.sin(y[0])
            + L1 * a2 * 2 * g * m2 * 2 * np.sin(y[0] - 2 * y[1])
            + 2 * I2 * a1 * g * m1 * np.sin(y[0])
            - (L1 * a2 * y[2] * m2) ** 2 * np.sin(2 * (y[0] - y[1]))
            - 2 * I2 * L1 * a2 * y[3] ** 2 * m2 * np.sin(y[0] - y[1])
            + 2 * a1 * a2 ** 2 * g * m1 * m2 * np.sin(y[0])
        )
        / (
            2 * I1 * I2
            + (L1 * a2 * m2) ** 2
            + 2 * I2 * L1 ** 2 * m2
            + 2 * I2 * a1 ** 2 * m1
            + 2 * I1 * a2 ** 2 * m2
            - (L1 * a2 * m2) ** 2 * np.cos(2 * (y[0] - y[1]))
            + 2 * (a1 * a2) ** 2 * m1 * m2
        ),
        (
            a2
            * m2
            * (
                2 * I1 * g * np.sin(y[1])
                + 2 * L1 ** 3 * y[2] ** 2 * m2 * np.sin(y[0] - y[1])
                + 2 * L1 ** 2 * g * m2 * np.sin(y[1])
                + 2 * I1 * L1 * y[2] ** 2 * np.sin(y[0] - y[1])
                + 2 * a1 ** 2 * g * m1 * np.sin(y[1])
                + L1 ** 2 * a2 * y[3] ** 2 * m2 * np.sin(2 * (y[0] - y[1]))
                + 2 * L1 * a1 ** 2 * y[2] ** 2 * m1 * np.sin(y[0] - y[1])
                - 2 * L1 ** 2 * g * m2 * np.cos(y[0] - y[1]) * np.sin(y[0])
                - 2 * L1 * a1 * g * m1 * np.cos(y[0] - y[1]) * np.sin(y[0])
            )
        )
        / (
            2
            * (
                I1 * I2
                + (L1 * a2 * m2) ** 2
                + I2 * L1 ** 2 * m2
                + I2 * a1 ** 2 * m1
                + I1 * a2 ** 2 * m2
                - (L1 * a2 * m2) ** 2 * np.cos(y[0] - y[1]) ** 2
                + a1 * 2 * a2 * 2 * m1 * m2
            )
        ),
    ]

# Simulate the double pendulum
t_span = (0, 10)
dt = 0.001
y0 = [np.pi - 0.6, np.pi - 0.4, 0, 0]
t_train = np.arange(t_span[0],t_span[-1],dt)
y_train = solve_ivp(
    double_pendulum_ode,
    t_span,
    y0,
    args=(0.2704, 0.2056, 0.191, 0.1621, 0.2667, 0.2667, 0.003, 0.0011, 9.81, 0, 0),
    t_eval=t_train,
).y.T

# Use PySINDy to discover the system dynamics
feature_names = ["sin(Q1)", "cos(Q1)", "sin(Q2)", "cos(Q2)", "sin(Q1 - Q2)", "cos(Q1 - Q2)", "Q1'", "Q2'", "Q1''", "Q2''"]
optimizer = ps.SINDyPI(threshold=0.002, max_iter=20000)
model = ps.SINDy(feature_names=feature_names, 
                 optimizer=optimizer,
                 differentiation_method=ps.FiniteDifference(drop_endpoints=True),
                 )
model.fit(y_train, t=t_train)
# Print the discovered equations
model.print()