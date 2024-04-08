import pysindy as ps
import numpy as np
import matplotlib.pyplot as plt
import LorenzSystem as LS
import pandas as pd
from sklearn.metrics import mean_squared_error
import warnings
from scipy.integrate.odepack import ODEintWarning
warnings.simplefilter("ignore", category=UserWarning)
warnings.simplefilter("ignore", category=FutureWarning)
warnings.simplefilter("ignore", category=ODEintWarning)

class LorenzSINDy():
    def __init__(self):

        #Common variables
        self.feature_names = ['x', 'y', 'z']
        self.dt = 0.002

        #Data generation for train and test
        self.x_traindata_clean = self.DataGeneration(t_stop = 10, dt = self.dt, noisy = False, iteration = 10, SEED = 20)
        self.x_testdata_clean = self.DataGeneration(t_stop = 15, dt = self.dt, noisy = False, iteration = 10, SEED = 21)
        self.x_traindata_noisy = self.DataGeneration(t_stop = 10, dt = self.dt, noisy = True, iteration = 10, SEED = 22)
        self.x_testdata_noisy = self.DataGeneration(t_stop = 15, dt = self.dt, noisy = True, iteration = 10, SEED = 23)

    def DataGeneration(self, SEED, noisy, t_stop = 10, dt = 0.002, iteration = 1):
        np.random.seed(SEED)
        df = pd.DataFrame()
        for i in range(iteration):
            #Calls LorenzSystem file to generate data as pandas dataframe format 
            lorenz_system = LS.LorenzSystem(sigma=10, rho=28, beta=8/3, noisy = noisy)
            t_span = (0, t_stop)
            t_eval = np.arange(0, t_stop, dt)
            x0 = ((np.random.rand(1, 3) - 0.5) * 20)[0]
            output = lorenz_system.simulate(x0, t_span, t_eval)
            df = pd.concat([df, output], axis=0)
        return df
        
    def HyperparameterEffect(self):
        #Dataframe to matrix type conversion
        x_train = self.DataGeneration(t_stop = 10, dt = self.dt, noisy = False, iteration = 1, SEED = 50)[['X', 'Y', 'Z']].values

        #Effect of sparse hyperparameter lambda (threshold)
        for threshold in [0, 0.1, 1]:
            print(f"Model output for threshold: {threshold}...")
            sparse_regression_optimizer = ps.STLSQ(threshold = threshold)
            model = ps.SINDy(feature_names = self.feature_names, optimizer = sparse_regression_optimizer)
            model.fit(x_train, t = self.dt)
            model.print()
            print("")

        #Hyperparameter lambda (threshold) effect to RMSE with Noisy Data
        threshold_scan = np.linspace(0, 1.0, 10)
        coefs = []
        x_train_noisy = self.DataGeneration(t_stop = 10, dt = self.dt, noisy = True, iteration = 1, SEED = 52)[['X', 'Y', 'Z']].values
        
        for i, threshold in enumerate(threshold_scan):
            sparse_regression_optimizer = ps.STLSQ(threshold = threshold)
            model = ps.SINDy(feature_names = self.feature_names, 
                            optimizer = sparse_regression_optimizer)
            model.fit(x_train_noisy, t = self.dt, quiet = True)
            coefs.append(model.coefficients())
        
        #Plots RMSE over changing lambda
        data = self.DataGeneration(t_stop = 15, dt = self.dt, noisy = False, iteration = 1, SEED = 51)
        x_test_clean = data[['X', 'Y', 'Z']].values
        t_test = data[['Time']].values.flatten()

        self.HyperparameterPlot(coefs, sparse_regression_optimizer, model, 
                    threshold_scan, x_test_clean, t_test)
        
    def DifferentiatorEffect(self):
        x_train_noisy = self.DataGeneration(t_stop = 10, dt = self.dt, noisy = True, iteration = 1, SEED = 20)[['X', 'Y', 'Z']].values
        self.DifferentiatorPlot(x_train_noisy, self.dt, ps.FiniteDifference()._differentiate, diff_name = "Finite Difference")
        self.DifferentiatorPlot(x_train_noisy, self.dt, ps.SmoothedFiniteDifference()._differentiate, diff_name = "SmoothedFiniteDifference") 

    def EnsembleEffect(self):
        pass
    
    def SaveSimulationData(self, data = pd.DataFrame(), PATH = str()):
        data.to_csv(PATH, encoding='utf-8')
    
    def HyperparameterPlot(self, coefs, opt, model, threshold_scan, x_test, t_test):                                                                    
        dt = t_test[1] - t_test[0]
        mse = np.zeros(len(threshold_scan))
        mse_sim = np.zeros(len(threshold_scan))
        for i in range(len(threshold_scan)):
            opt.coef_ = coefs[i]
            mse[i] = model.score(x_test, t=dt, metric = mean_squared_error)
            x_test_sim = model.simulate(x_test[0, :], t_test, integrator="odeint")
            if np.any(x_test_sim > 1e4):
                x_test_sim = 1e4
            mse_sim[i] = np.sum((x_test - x_test_sim) ** 2)
        plt.figure(figsize = (8,5))
        plt.title(r"RMSE Over Changing $\lambda$", fontsize = 15)
        plt.semilogy(threshold_scan, mse, "bo")
        plt.semilogy(threshold_scan, mse, "b")
        plt.ylabel(r"$\dot{X}$ RMSE", fontsize = 12)
        plt.xlabel(r"$\lambda$", fontsize = 12)
        plt.xticks(fontsize = 10)
        plt.yticks(fontsize = 10)
        plt.grid(True)
        plt.savefig(f"./LorenzSystem/Figures/SINDyFigures/HyperparameterEffect.png")
        plt.show()

    def DifferentiatorPlot(self, x, dt, deriv, diff_name):
        plt.figure(figsize=(30, 8))
        for i in range(3):
            plt.subplot(1, 3, i + 1)
            plt.plot(x[:, i], label = self.feature_names[i])
            plt.grid(True)
            plt.title(f"{diff_name} - {self.feature_names[i]} Value ", fontsize = 20)
            plt.xlabel("t", fontsize=24)
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            plt.legend(fontsize=18)
        plt.savefig(f"./LorenzSystem/Figures/SINDyFigures/{diff_name} - xyz.png")
        x_dot = deriv(x, t=dt)
        plt.figure(figsize=(30, 8))
        for i in range(3):
            plt.subplot(1, 3, i + 1)
            plt.plot(x_dot[:, i], label=r"$\dot{" + self.feature_names[i] + "}$")
            plt.grid(True)
            plt.title(f"{diff_name} - $\\dot{{{self.feature_names[i]}}}$ Value", fontsize = 20)
            plt.xlabel("t", fontsize=24)
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            plt.legend(fontsize=18)
        plt.savefig(f"./LorenzSystem/Figures/SINDyFigures/{diff_name} - xdotydotzdot.png")
    
if __name__ == "__main__":
    LorenzSys = LorenzSINDy()
    LorenzSys.HyperparameterEffect()
    #LorenzSys.DifferentiatorEffect()