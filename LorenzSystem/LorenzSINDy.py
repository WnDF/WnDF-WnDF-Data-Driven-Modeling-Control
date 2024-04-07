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
        self.x_traindata_clean = self.DataGeneration(t_stop = 10, dt = self.dt, noisy = False, iteration = 10, SEED = 50)
        self.x_testdata_clean = self.DataGeneration(t_stop = 15, dt = self.dt, noisy = False, iteration = 10, SEED = 51)
        self.x_traindata_noisy = self.DataGeneration(t_stop = 10, dt = self.dt, noisy = True, iteration = 10, SEED = 52)
        self.x_testdata_noisy = self.DataGeneration(t_stop = 15, dt = self.dt, noisy = True, iteration = 10, SEED = 53)

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
        x_test_clean = self.DataGeneration(t_stop = 15, dt = self.dt, noisy = False, iteration = 1, SEED = 51)[['X', 'Y', 'Z']].values
        t_test = self.DataGeneration(t_stop = 15, dt = self.dt, noisy = False, iteration = 1, SEED = 51)[['Time']].values.flatten()

        self.HyperparameterPlot(coefs, sparse_regression_optimizer, model, 
                    threshold_scan, x_test_clean, t_test)
        
    def DifferentiatorEffect(self):
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
    
if __name__ == "__main__":
    LorenzSys = LorenzSINDy()
    LorenzSys.HyperparameterEffect()