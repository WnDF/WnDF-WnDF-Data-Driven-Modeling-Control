import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
from scipy.integrate import solve_ivp
from sklearn.metrics import mean_squared_error
import matplotlib.cm as cm
import warnings

integrator_keywords = {}
integrator_keywords['rtol'] = 1e-12
integrator_keywords['method'] = 'LSODA'
integrator_keywords['atol'] = 1e-12

warnings.filterwarnings("ignore")

class LorenzSystem:
    def __init__(self, sigma = 10, rho = 28, beta = 8/3, noisy = False):
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.noisy = noisy

    def lorenz_equations(self, t, y):
        x, y, z = y
        dx_dt = self.sigma * (y - x)
        dy_dt = x * (self.rho - z) - y
        dz_dt = x * y - self.beta * z
        return [dx_dt, dy_dt, dz_dt]

    def solve(self, y0, t_span, t_eval):
        sol = solve_ivp(self.lorenz_equations, t_span, y0, t_eval=t_eval, **integrator_keywords)
        return sol

    def noisy_data_generation(self, x):
        rmse = mean_squared_error(x, np.zeros(x.shape), squared=False)
        x_noisy = x + np.random.normal(0, rmse / 50.0, x.shape)
        return x_noisy

    def simulate(self, y0, t_span, t_eval):
        sol = self.solve(y0, t_span, t_eval)
        t = sol.t
        y = sol.y

        x, y, z = y[0], y[1], y[2]

        if self.noisy:
            x = self.noisy_data_generation(x)
            y = self.noisy_data_generation(y)
            z = self.noisy_data_generation(z)

        df = pd.DataFrame({'Time': t, 'X': x, 'Y': y, 'Z': z})
        return df

    def plot_outputs(self, df):
        fig = plt.figure(figsize=(12, 10))  # Increase plot size
        ax = fig.add_subplot(111, projection='3d')
        
        # Define the custom colormap from light purple to dark purple
        cmap = cm.get_cmap('Purples_r')
        colors = cmap(np.linspace(0.1, 0.7, len(df['X'])))  # Adjust the range to include tones of purple
        
        for i in range(len(df['X'])-1):
            ax.plot(df['X'][i:i+2], df['Y'][i:i+2], df['Z'][i:i+2], color=colors[i], linewidth=0.8)
        
        ax.set_title('Lorenz Attractor')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.grid(True, color='k', linestyle='-', linewidth=0.5)  # Change grid color and style
        ax.set_facecolor('w')  # Change background color to white
        ax.xaxis.pane.set_edgecolor('black')  # Change x-axis color to black
        ax.yaxis.pane.set_edgecolor('black')  # Change y-axis color to black
        ax.zaxis.pane.set_edgecolor('black')  # Change z-axis color to black
        ax.view_init(elev=20, azim=45)  # Adjusting the viewing angle
        plt.show()

# Example usage:
if __name__ == "__main__":
    lorenz_system = LorenzSystem(sigma=10, rho=28, beta=8/3, noisy=False)

    y0 = [1,2,3]
    t_stop = 100
    t_span = (0, t_stop)  # Simulate for 100 seconds
    t_eval = np.arange(0, t_stop, 0.01)

    df = lorenz_system.simulate(y0, t_span, t_eval)
    print(df)

    lorenz_system.plot_outputs(df)
