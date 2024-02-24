import numpy as np
from scipy.integrate import solve_ivp
import pandas as pd
import matplotlib.pyplot as plt

class DoublePendulumSS:
    def __init__(self, m1=0.2704, m2=0.2056, cg1=0.191, cg2=0.1621, L1=0.2667, L2=0.2667, I1=0.003, I2=0.0011, g=9.81):
        self.m1 = m1  # mass of pendulum 1
        self.m2 = m2  # mass of pendulum 2
        self.L1 = L1  # length of pendulum 1
        self.L2 = L2  # length of pendulum 2
        self.cg1 = cg1 if cg1 is not None else L1 / 2  # center of gravity position of pendulum 1
        self.cg2 = cg2 if cg2 is not None else L2 / 2  # center of gravity position of pendulum 2
        self.g = g    # acceleration due to gravity
        self.I1 = I1  # inertia of pendulum 1
        self.I2 = I2  # inertia of pendulum 2

    def equations_of_motion(self, t, y):
        theta1, theta2, omega1, omega2 = y[:4]

        theta1_dot = omega1
        theta2_dot = omega2

        omega1_dot = (
                self.L1 * self.cg2 ** 2 * self.g * self.m2 ** 2 * np.sin(theta1)
                - 2 * self.L1 * self.cg2 ** 3 * omega2 ** 2 * self.m2 ** 2 * np.sin(theta1 - theta2)
                + 2 * self.I2 * self.L1 * self.g * self.m2 * np.sin(theta1)
                + self.L1 * self.cg2 ** 2 * self.g * self.m2 ** 2 * np.sin(theta1 - 2 * theta2)
                + 2 * self.I2 * self.cg1 * self.g * self.m1 * np.sin(theta1)
                - (self.L1 * self.cg2 * omega1 * self.m2) ** 2 * np.sin(2 * (theta1 - theta2))
                - 2 * self.I2 * self.L1 * self.cg2 * omega2 ** 2 * self.m2 * np.sin(theta1 - theta2)
                + 2 * self.cg1 * self.cg2 ** 2 * self.g * self.m1 * self.m2 * np.sin(theta1)
                )/(
                2 * self.I1 * self.I2
                + (self.L1 * self.cg2 * self.m2) ** 2
                + 2 * self.I2 * self.L1 ** 2 * self.m2
                + 2 * self.I2 * self.cg1 ** 2 * self.m1
                + 2 * self.I1 * self.cg2 ** 2 * self.m2
                - (self.L1 * self.cg2 * self.m2) ** 2 * np.cos(2 * (theta1 - theta2))
                + 2 * (self.cg1 * self.cg2) ** 2 * self.m1 * self.m2
                )

        omega2_dot =(
                self.cg2
                * self.m2
                * (
                    2 * self.I1 * self.g * np.sin(theta2)
                    + 2 * self.L1 ** 3 * omega1 ** 2 * self.m2 * np.sin(theta1 - theta2)
                    + 2 * self.L1 ** 2 * self.g * self.m2 * np.sin(theta2)
                    + 2 * self.I1 * self.L1 * omega1 ** 2 * np.sin(theta1 - theta2)
                    + 2 * self.cg1 ** 2 * self.g * self.m1 * np.sin(theta2)
                    + self.L1 ** 2 * self.cg2 * omega2 ** 2 * self.m2 * np.sin(2 * (theta1 - theta2))
                    + 2 * self.L1 * self.cg1 ** 2 * omega1 ** 2 * self.m1 * np.sin(theta1 - theta2)
                    - 2 * self.L1 ** 2 * self.g * self.m2 * np.cos(theta1 - theta2) * np.sin(theta1)
                    - 2 * self.L1 * self.cg1 * self.g * self.m1 * np.cos(theta1 - theta2) * np.sin(theta1)
                )
                )/(
                2
                * (
                    self.I1 * self.I2
                    + (self.L1 * self.cg2 * self.m2) ** 2
                    + self.I2 * self.L1 ** 2 * self.m2
                    + self.I2 * self.cg1 ** 2 * self.m1
                    + self.I1 * self.cg2 ** 2 * self.m2
                    - (self.L1 * self.cg2 * self.m2) ** 2 * np.cos(theta1 - theta2) ** 2
                    + self.cg1 ** 2 * self.cg2 ** 2 * self.m1 * self.m2
                )
            )

        return [theta1_dot, theta2_dot, omega1_dot, omega2_dot]

    def solve(self, y0, t_span, t_eval=None):
        sol = solve_ivp(self.equations_of_motion, t_span, y0, t_eval=t_eval)
        return sol

    def simulate(self, y0, t_span, t_eval=None):
        sol = self.solve(y0, t_span, t_eval)
        t = sol.t
        y = sol.y

        theta1, theta2 = y[0], y[1]
        omega1, omega2 = y[2], y[3]

        theta1_dot, theta2_dot = self.equations_of_motion(t, y)[:2]
        omega1_dot, omega2_dot = self.equations_of_motion(t, y)[2:]

        df = pd.DataFrame({'Time': t, 'Theta1': theta1, 'Theta2': theta2,
                           'Omega1': omega1, 'Omega2': omega2,
                           'Theta1_dot': theta1_dot, 'Theta2_dot': theta2_dot,
                           'Omega1_dot': omega1_dot, 'Omega2_dot': omega2_dot})
        return df

    def plot_outputs(self, df):
        plt.figure(figsize=(10, 6))
        plt.plot(df['Time'], df['Theta1'], label='Theta1')
        plt.plot(df['Time'], df['Theta2'], label='Theta2')
        #plt.plot(df['Time'], df['Omega1'], label='Omega1')
        #plt.plot(df['Time'], df['Omega2'], label='Omega2')
        plt.title('Double Pendulum Motion')
        plt.xlabel('Time')
        plt.ylabel('Angle/Velocity')
        plt.legend()
        plt.grid(True)
        plt.show()

# Example usage:
if __name__ == "__main__":
    double_pendulum = DoublePendulumSS(m1=0.2704, m2=0.2056, cg1=0.191, cg2=0.1621, L1=0.2667, L2=0.2667, I1=0.003, I2=0.0011, g=9.81)  # Set parameters

    y0 = [np.pi+0.3, np.pi-0.5, 0, 0]
    t_span = (0, 3)
    t_eval = np.arange(0, 3, 0.001)

    df = double_pendulum.simulate(y0, t_span, t_eval)
    print(df)

    double_pendulum.plot_outputs(df)