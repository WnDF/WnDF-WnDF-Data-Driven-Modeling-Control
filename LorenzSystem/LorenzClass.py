import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

class LorenzSystem:
    def __init__(self, sigma = 1, rho = 1, beta = 1):
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.initial_conditions = None
        self.t_span = None

    def lorenz_equations(self, t, u):
        x, y, z = u
        dxdt = self.sigma * (y - x)
        dydt = x * (self.rho - z) - y
        dzdt = x * y - self.beta * z
        return [dxdt, dydt, dzdt]

    def set_initial_conditions(self, initial_conditions):
        self.initial_conditions = initial_conditions

    def set_time_span(self, t_span):
        self.t_span = t_span

    def solve_system(self, dt=0.01):
        if self.initial_conditions is None or self.t_span is None:
            raise ValueError("Initial conditions and time span must be set before solving the system.")

        t_eval = np.arange(self.t_span[0], self.t_span[1], dt)
        solution = solve_ivp(self.lorenz_equations, self.t_span, self.initial_conditions, args=(), t_eval=t_eval)
        return solution

    def plot_solution(self, solution):
        x_points = solution.y[0]
        y_points = solution.y[1]
        z_points = solution.y[2]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(x_points, y_points, z_points, lw=0.5)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Lorenz Attractor')
        plt.show()
    
    def get_final_equations(self):
        equations = [
            f"dx/dt = {self.sigma}(y - x)",
            f"dy/dt = x({self.rho} - z) - y",
            f"dz/dt = xy - {self.beta}z"
        ]
        return equations