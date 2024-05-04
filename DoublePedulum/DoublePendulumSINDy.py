import numpy as np
import pysindy as ps
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd
import DoublePendulum as DP

class DoublePendulumnSINDy:
    def __init__(self):

        #Common variables
        self.feature_names = ["Theta1", "Theta2", "Omega1", "Omega2"]
        self.dt = 0.001

        #Data generation for train and test
        self.traindata_clean = self.DataGeneration(t_stop = 10, dt = self.dt, is_noisy = False, iteration = 1, SEED = 52)
        self.testdata_clean = self.DataGeneration(t_stop = 10, dt = self.dt, is_noisy = False, iteration = 1, SEED = 53)
        self.traindata_noisy = self.DataGeneration(t_stop = 10, dt = self.dt, is_noisy = True, iteration = 1, SEED = 52)
        self.testdata_noisy = self.DataGeneration(t_stop = 10, dt = self.dt, is_noisy = True, iteration = 1, SEED = 53)

        self.SaveSimulationData(data = pd.concat([self.traindata_clean,self.testdata_clean], axis = 1), 
                                PATH = f'./DoublePedulum/Dataset/SINDyDataset/DPSINDyDatasetClean.csv') 
        self.SaveSimulationData(data = pd.concat([self.traindata_noisy, self.testdata_noisy], axis = 1), 
                                PATH = f'./DoublePedulum/Dataset/SINDyDataset/DPSINDyDatasetNoisy.csv')

        self.model_clean_SINDy = []
        self.model_noisy_SINDy = []
        self.model_clean_SINDyPI = []
        self.model_noisy__SINDyPI = []

        self.test_sim_clean_SINDy = np.array(0)
        self.test_sim_noisy_SINDy = np.array(0)
        self.test_sim_clean_SINDyPI = np.array(0)
        self.test_sim_noisy_SINDyPI = np.array(0)

    def DataGeneration(self, SEED, is_noisy, t_stop = 10, dt = 0.001, iteration = 1,
                        m1 = 0.2704, m2 = 0.2056, cg1 = 0.191, cg2 = 0.1621, 
                        L1 = 0.2667, L2 = 0.2667, I1 = 0.003, I2 = 0.0011, g = 9.81):
        
        print("-> DataGeneration Node Runnig...\n")
        print("Generating simulation data...")

        np.random.seed(SEED)
        df = pd.DataFrame()
        for i in range(iteration):
            #Calls LorenzSystem file to generate data as pandas dataframe format 
            dp_system = DP.DoublePendulumSS(noisy = is_noisy, m1 = m1, m2 = m2, cg1 = cg1, cg2 = cg2, L1 = L1, L2 = L2, I1 = I1, I2 = I2, g = g)
            inits = [180.0*np.random.rand(), 180.0*np.random.rand(), 0, 0]
            # initial states
            state = np.radians(inits)
            t_span = (0, t_stop)
            t_eval = np.arange(0, t_stop, dt)

            df = dp_system.simulate(state, t_span, t_eval)
        return df
    
    def SINYDyModel(self, threshold_clean, threshold_noisy):
        print("-> ModelTrainEval Node Runnig...\n")
        train_clean = self.traindata_clean[["Theta1", "Theta2", "Omega1", "Omega2"]].loc[0:self.traindata_clean.shape[0]-2].reset_index(drop = True).values 
        train_noisy = self.traindata_noisy[["Theta1", "Theta2", "Omega1", "Omega2"]].loc[0:self.traindata_noisy.shape[0]-2].reset_index(drop = True).values
        test_clean = self.testdata_clean[["Theta1", "Theta2", "Omega1", "Omega2"]].loc[0:self.testdata_clean.shape[0]-2].reset_index(drop = True).values
        test_noisy = self.testdata_noisy[["Theta1", "Theta2", "Omega1", "Omega2"]].loc[0:self.testdata_noisy.shape[0]-2].reset_index(drop = True).values
        test_timespan = self.testdata_clean[["Time"]].loc[0:self.testdata_clean.shape[0]-2].reset_index(drop = True).values

        opt_clean = ps.STLSQ(threshold = threshold_clean)
        opt_noisy = ps.STLSQ(threshold = threshold_noisy)

        self.model_clean_SINDy = ps.SINDy(feature_names = self.feature_names, 
                                          optimizer = opt_clean)
        self.model_noisy_SINDy = ps.SINDy(feature_names = self.feature_names, 
                                          optimizer = opt_noisy, 
                                          differentiation_method = ps.SmoothedFiniteDifference()._differentiate)
        
        print("Tranining Model...")
        self.model_clean_SINDy.fit(train_clean, t = self.dt, quiet = True)
        self.model_noisy_SINDy.fit(train_noisy, t = self.dt, ensemble = True, quiet = True)

        print("Model with Clean Data ---->")
        self.model_clean_SINDy.print()
        print("")
        print("Model with Noisy Data ---->")
        self.model_noisy_SINDy.print()
        print("")

        print("Simulating Models...")
        self.test_sim_clean_SINDy = self.model_clean_SINDy.simulate(test_clean[0, :], test_timespan.flatten(), integrator = "odeint")
        self.test_sim_noisy_SINDy = self.model_noisy_SINDy.simulate(test_noisy[0, :], test_timespan.flatten(), integrator = "odeint")

        print("-> ModelTrainEval Node Executed...")
    
    def SaveSimulationData(self, data = pd.DataFrame(), PATH = str()):
        data.to_csv(PATH, encoding='utf-8')

    def SINDyPlots(self):
        print("-> SINDyPlots Node Runnig...\n")
        test_timespan = self.testdata_clean[['Time']].loc[0:self.testdata_clean.shape[0]-2].reset_index(drop = True).values
        test_out_clean = self.testdata_clean[["Theta1", "Theta2", "Omega1", "Omega2"]].loc[1:self.testdata_clean.shape[0]-1].reset_index(drop = True).values
        test_out_noisy = self.testdata_noisy[["Theta1", "Theta2", "Omega1", "Omega2"]].loc[1:self.testdata_noisy.shape[0]-1].reset_index(drop = True).values

        figsize = (30, 20)
        title_font = 24
        xlabel_font = 20
        ylabel_font = 20
        ticks_font = 16
        legend_font = 18

        fig1 = plt.figure(figsize = figsize)
        fig1ax1 = fig1.add_subplot(2,1,1)
        fig1ax2 = fig1.add_subplot(2,1,2)

        fig1ax1.plot(test_timespan, test_out_clean[:,0], color = 'red', label = 'Act.')
        fig1ax1.plot(test_timespan, self.test_sim_clean_SINDy[:,0], color = 'blue', label = 'Pred.')
        fig1ax1.set_title("First Link Angle - " r'$\theta_{1}$' " - (Clean Data)", fontweight="bold", fontsize = title_font)
        fig1ax1.set_xlabel('Time (sec)', fontsize = xlabel_font)
        fig1ax1.set_ylabel(r'$\theta_{1}$'u'\xb0' " (deg)", fontsize = ylabel_font)
        fig1ax1.tick_params(labelsize = ticks_font)
        fig1ax1.legend(loc = 'upper right', fontsize = legend_font)
        fig1ax1.grid(linestyle='--')

        fig1ax2.plot(test_timespan, test_out_clean[:,1], color = 'red', label = 'Act.')
        fig1ax2.plot(test_timespan, self.test_sim_clean_SINDy[:,1], color = 'blue', label = 'Pred.')
        fig1ax2.set_title("Second Link Angle - " r'$\theta_{2}$' " - (Clean Data)", fontweight="bold", fontsize = title_font)
        fig1ax2.set_xlabel('Time (sec)', fontsize = xlabel_font)
        fig1ax2.set_ylabel(r'$\theta_{2}$'u'\xb0' " (deg)", fontsize = ylabel_font)
        fig1ax2.tick_params(labelsize = ticks_font)
        fig1ax2.legend(loc = 'upper right', fontsize = legend_font)
        fig1ax2.grid(linestyle='--')

        fig1.savefig(f"./DoublePedulum/Figures/SINDyFigures/ActualVsPredictedLinkAngleTheta(Clean Data).png")

        fig2 = plt.figure(figsize = figsize)
        fig2ax1 = fig2.add_subplot(2,1,1)
        fig2ax2 = fig2.add_subplot(2,1,2)

        fig2ax1.plot(test_timespan, test_out_clean[:,2], color = 'red', label = 'Act.')
        fig2ax1.plot(test_timespan, self.test_sim_clean_SINDy[:,2], color = 'blue', label = 'Pred.')
        fig2ax1.set_title("First Link Angular Speed - " r'$\omega_{1}$' " - (Clean Data)", fontweight="bold", fontsize = title_font)
        fig2ax1.set_xlabel('Time (sec)', fontsize = xlabel_font)
        fig2ax1.set_ylabel(r'$\omega_{1}$'" (deg/s)", fontsize = ylabel_font)
        fig2ax1.tick_params(labelsize = ticks_font)
        fig2ax1.legend(loc = 'upper right', fontsize = legend_font)
        fig2ax1.grid(linestyle='--')

        fig2ax2.plot(test_timespan, test_out_clean[:,3], color = 'red', label = 'Act.')
        fig2ax2.plot(test_timespan, self.test_sim_clean_SINDy[:,3], color = 'blue', label = 'Pred.')
        fig2ax2.set_title("First Link Angular Speed - " r'$\omega_{2}$' " - (Clean Data)", fontweight="bold", fontsize = title_font)
        fig2ax2.set_xlabel('Time (sec)', fontsize = xlabel_font)
        fig2ax2.set_ylabel(r'$\omega_{2}$'" (deg/s)", fontsize = ylabel_font)
        fig2ax2.tick_params(labelsize = ticks_font)
        fig2ax2.legend(loc = 'upper right', fontsize = legend_font)
        fig2ax2.grid(linestyle='--')

        fig2.savefig(f"./DoublePedulum/Figures/SINDyFigures/ActualVsPredictedLinkAngularSpeedOmega(Clean Data).png")

        fig3 = plt.figure(figsize = figsize)
        fig3ax1 = fig3.add_subplot(2,1,1)
        fig3ax2 = fig3.add_subplot(2,1,2)

        fig3ax1.plot(test_timespan, test_out_noisy[:,0], color = 'red', label = 'Act.')
        fig3ax1.plot(test_timespan, self.test_sim_noisy_SINDy[:,0], color = 'blue', label = 'Pred.')
        fig3ax1.set_title("First Link Angle - " r'$\theta_{1}$' " - (Noisy Data)", fontweight="bold", fontsize = title_font)
        fig3ax1.set_xlabel('Time (sec)', fontsize = xlabel_font)
        fig3ax1.set_ylabel(r'$\theta_{1}$'" (deg)", fontsize = ylabel_font)
        fig3ax1.tick_params(labelsize = ticks_font)
        fig3ax1.legend(loc = 'upper right', fontsize = legend_font)
        fig3ax1.grid(linestyle='--')

        fig3ax2.plot(test_timespan, test_out_noisy[:,1], color = 'red', label = 'Act.')
        fig3ax2.plot(test_timespan, self.test_sim_noisy_SINDy[:,1], color = 'blue', label = 'Pred.')
        fig3ax2.set_title("Second Link Angle - " r'$\theta_{2}$' " - (Noisy Data)", fontweight="bold", fontsize = title_font)
        fig3ax2.set_xlabel('Time (sec)', fontsize = xlabel_font)
        fig3ax2.set_ylabel(r'$\theta_{2}$'" (deg)", fontsize = ylabel_font)
        fig3ax2.tick_params(labelsize = ticks_font)
        fig3ax2.legend(loc = 'upper right', fontsize = legend_font)
        fig3ax2.grid(linestyle='--')

        fig3.savefig(f"./DoublePedulum/Figures/SINDyFigures/ActualVsPredictedLinkAngleTheta(Noisy Data).png")

        fig4 = plt.figure(figsize = figsize)
        fig4ax1 = fig4.add_subplot(2,1,1)
        fig4ax2 = fig4.add_subplot(2,1,2)

        fig4ax1.plot(test_timespan, test_out_noisy[:,2], color = 'red', label = 'Act.')
        fig4ax1.plot(test_timespan, self.test_sim_noisy_SINDy[:,2], color = 'blue', label = 'Pred.')
        fig4ax1.set_title("First Link Angular Speed - " r'$\omega_{1}$' " - (Noisy Data)", fontweight="bold", fontsize = title_font)
        fig4ax1.set_xlabel('Time (sec)', fontsize = xlabel_font)
        fig4ax1.set_ylabel(r'$\omega_{1}$'" (deg/s)", fontsize = ylabel_font)
        fig4ax1.tick_params(labelsize = ticks_font)
        fig4ax1.legend(loc = 'upper right', fontsize = legend_font)
        fig4ax1.grid(linestyle='--')

        fig4ax2.plot(test_timespan, test_out_noisy[:,3], color = 'red', label = 'Act.')
        fig4ax2.plot(test_timespan, self.test_sim_noisy_SINDy[:,3], color = 'blue', label = 'Pred.')
        fig4ax2.set_title("Second Link Angular Speed - " r'$\omega_{2}$' " - (Noisy Data)", fontweight="bold", fontsize = title_font)
        fig4ax2.set_xlabel('Time (sec)', fontsize = xlabel_font)
        fig4ax2.set_ylabel(r'$\omega_{2}$'" (deg/s)", fontsize = ylabel_font)
        fig4ax2.tick_params(labelsize = ticks_font)
        fig4ax2.legend(loc = 'upper right', fontsize = legend_font)
        fig4ax2.grid(linestyle='--')

        fig4.savefig(f"./DoublePedulum/Figures/SINDyFigures/ActualVsPredictedLinkAngularSpeedOmega(Noisy Data).png")

        print("-> SINDyPlots Node Executed...\n")

if __name__ == "__main__":
    DPSINDy = DoublePendulumnSINDy()
    DPSINDy.SINYDyModel(threshold_clean = 0.32, threshold_noisy = 0.45)
    DPSINDy.SINDyPlots()