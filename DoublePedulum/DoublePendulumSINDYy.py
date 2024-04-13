import numpy as np
import pysindy as ps
from scipy.integrate import solve_ivp

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

# Initialize custom SINDy library
x_library_functions = [
    lambda theta1: theta1,
    lambda theta2: theta2,
    lambda omega1: omega1,
    lambda omega2: omega2,
    lambda theta1: np.sin(theta1),
    lambda theta2: np.sin(theta2),
    lambda omega1: np.cos(omega1),
    lambda omega2: np.cos(omega2),
    lambda theta1, theta2, omega2: omega2**2*(np.sin(theta1-theta2)),
    lambda theta1, theta2, omega1: omega1**2*(np.sin(theta1-theta2)),
    lambda theta1, theta2: np.cos(theta1-theta2),
    # lambda theta1, omega1: omega1*np.cos(theta1),
    # lambda theta1, omega1: omega1*np.sin(theta1),
    # lambda theta2, omega2: omega2*np.cos(theta2),
    # lambda theta2, omega2: omega2*np.sin(theta2),
]
x_dot_library_functions = [
    lambda theta1: theta1,
    lambda theta2: theta2,
    lambda theta1: theta1 ** 2,
    lambda theta2: theta2 ** 2,
    lambda omega1: omega1,
    lambda omega2: omega2,
    lambda omega2: omega2 ** 2,
    lambda omega2: omega2 ** 2,
]

# Combine the library functions and their names
library_function_names = [
    r'$\theta_{1}$', 
    r'$\theta_{2}$', 
    r'$\omega_{1}$', 
    r'$\omega_{2}$', 
    r'$\sin(\theta_{1})$', 
    r'$\sin(\theta_{2})$', 
    r'$\cos(\theta_{1})$', 
    r'$\cos(\theta_{2})$', 
    r'$\omega_{2}^{2}\sin(\theta_{1}-\theta_{2})$', 
    r'$\omega_{1}^{2}\sin(\theta_{1}-\theta_{2})$', 
    r'$\cos(\theta_{1}-\theta_{2})$',
    # r'$\omega_{1}\cos(\theta_{1})$', 
    # r'$\omega_{1}\sin(\theta_{1})$',
    # r'$\omega_{2}\cos(\theta_{2})$', 
    # r'$\omega_{2}\sin(\theta_{2})$',
    r'$\theta_{1}$', 
    r'$\theta_{2}$',
    r'$\theta_{1}^{2}$', 
    r'$\theta_{2}^{2}$', 
    r'$\omega_{1}$', 
    r'$\omega_{2}$',
    r'$\omega_{1}^{2}$', 
    r'$\omega_{2}^{2}$', 
]

# Initialize SINDy library
sindy_library = ps.SINDyPILibrary(
    library_functions=x_library_functions,
    x_dot_library_functions=x_dot_library_functions,
    t=t_train,
    function_names=library_function_names,
    include_bias=True,
)

# Use SINDy-PI optimizer
sindy_opt = ps.SINDyPI(
    threshold=1e-6,
    tol=1e-8,
    max_iter=20,
)

# Initialize SINDy model
model = ps.SINDy(
    optimizer=sindy_opt,
    feature_library=sindy_library,
    differentiation_method=ps.FiniteDifference(drop_endpoints=True),
    feature_names = [r'$\theta_{1}$', r'$\theta_{2}$', r'$\omega_{1}$', r'$\omega_{2}$',]
)

# Fit the model to training data
model.fit(x_train, t=t_train)

# Print the discovered equations
model.print()
