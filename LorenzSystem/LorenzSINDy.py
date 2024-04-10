import pysindy as ps
import numpy as np
import matplotlib.pyplot as plt
import LorenzSystem as LS
import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy.integrate import solve_ivp
import warnings
from scipy.integrate.odepack import ODEintWarning
warnings.simplefilter("ignore", category=UserWarning)
warnings.simplefilter("ignore", category=FutureWarning)
warnings.simplefilter("ignore", category=ODEintWarning)

integrator_keywords = {}
integrator_keywords['rtol'] = 1e-12
integrator_keywords['method'] = 'LSODA'
integrator_keywords['atol'] = 1e-12

class LorenzSINDy():
    def __init__(self):

        #Common variables
        self.feature_names = ['x', 'y', 'z']
        self.dt = 0.002

        #Data generation for train and test
        self.x_traindata_clean = self.DataGeneration(t_stop = 10, dt = self.dt, noisy = False, iteration = 1, SEED = 24)
        self.x_testdata_clean = self.DataGeneration(t_stop = 10, dt = self.dt, noisy = False, iteration = 1, SEED = 23)
        self.x_traindata_noisy = self.DataGeneration(t_stop = 10, dt = self.dt, noisy = True, iteration = 1, SEED = 24)
        self.x_testdata_noisy = self.DataGeneration(t_stop = 10, dt = self.dt, noisy = True, iteration = 1, SEED = 23)

        self.model_clean = []
        self.model_noisy = []
        self.init_X_noisy = list()
        self.init_X_clean = list()

        self.x_test_sim_clean = np.array(0)
        self.x_test_sim_noisy_finitedif = np.array(0)
        self.x_test_sim_noisy_smoothdif = np.array(0)
        self.x_test_sim_noisy_smoothdif_ensemble = np.array(0)
        self.x_test_sim_noisy_smoothdif_ensemble_weakform = np.array(0)

    def DataGeneration(self, SEED, noisy, t_stop = 10, dt = 0.002, iteration = 1):
        print("-> DataGeneration Node Runnig...\n")
        print("Generating simulation data...")
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
        
        print("Simulation data generated...\n")
        print("-> DataGeneration Node Executed...")
        return df
        
    def HyperparameterEffect(self):
        print("-> HyperparameterEffect Node Runnig...\n")
        #Dataframe to matrix type conversion
        x_train = self.x_traindata_clean[['X', 'Y', 'Z']].values
        x_train_noisy = self.x_traindata_noisy[['X', 'Y', 'Z']].values

        #Effect of sparse hyperparameter lambda (threshold)
        for threshold in [0, 0.33, 0.7, 1]:
            print(f"Model with clean data output for threshold: {threshold}...")
            sparse_regression_optimizer = ps.STLSQ(threshold = threshold)
            model = ps.SINDy(feature_names = self.feature_names, optimizer = sparse_regression_optimizer)
            model.fit(x_train, t = self.dt)
            model.print()
            print("")

        #Hyperparameter lambda (threshold) effect to RMSE with Noisy Data
        threshold_scan = np.linspace(0, 1.0, 10)
        coefs = []
        
        for i, threshold in enumerate(threshold_scan):
            sparse_regression_optimizer = ps.STLSQ(threshold = threshold)
            model = ps.SINDy(feature_names = self.feature_names, 
                            optimizer = sparse_regression_optimizer)
            print(f"Noisy Model for threshold: {threshold}")
            model.fit(x_train_noisy, t = self.dt)
            model.print()
            print("")
            coefs.append(model.coefficients())
        
        #Plots RMSE over changing lambda

        print("Plotting Hyperparameter Effect plot...\n")
        x_test_clean = self.x_testdata_clean[['X', 'Y', 'Z']].values
        t_test = self.x_testdata_clean[['Time']].values.flatten()

        self.HyperparameterPlot(coefs, sparse_regression_optimizer, model, 
                    threshold_scan, x_test_clean, t_test)
        
        print("-> HyperparameterEffect Node Executed...")
        
    def DifferentiatorEffect(self):
        print("-> DifferentiatorEffect Node Runnig...\n")

        x_train_noisy = self.x_traindata_noisy[['X', 'Y', 'Z']].loc[0:self.x_traindata_noisy.shape[0]-2].reset_index(drop = True).values
        x_test_noisy = self.x_testdata_noisy[['X', 'Y', 'Z']].loc[0:self.x_testdata_noisy.shape[0]-2].reset_index(drop = True).values
        test_timespan = self.x_testdata_clean[['Time']].loc[0:self.x_testdata_clean.shape[0]-2].reset_index(drop = True).values

        opt = ps.STLSQ(threshold = 0.3)
        model = ps.SINDy(feature_names = self.feature_names, optimizer = opt, differentiation_method = ps.FiniteDifference()._differentiate)
        print("Training model...")
        print("Model with Finite Difference")
        model.fit(x_train_noisy, t = self.dt)
        model.print()
        print("")
        self.x_test_sim_noisy_finitedif = model.simulate(x_test_noisy[0, :], test_timespan.flatten(), integrator = "odeint")

        print("Plotting Differentiator Effect plot - Finite Difference...")
        self.DifferentiatorPlot(x_train_noisy, self.dt, ps.FiniteDifference()._differentiate, diff_name = "FiniteDifference")
        print("Plotting Differentiator Effect plot - Smoothed Finite Difference...\n")
        self.DifferentiatorPlot(x_train_noisy, self.dt, ps.SmoothedFiniteDifference()._differentiate, diff_name = "SmoothedFiniteDifference")

        print("-> DifferentiatorEffect Node Executed...") 

    def EnsembleEffect(self):
        print("-> EnsembleEffect Node Runnig...\n")
        x_train_noisy = self.x_traindata_noisy[['X', 'Y', 'Z']].loc[0:self.x_traindata_noisy.shape[0]-2].reset_index(drop = True).values
        x_test_noisy = self.x_testdata_noisy[['X', 'Y', 'Z']].loc[0:self.x_testdata_noisy.shape[0]-2].reset_index(drop = True).values
        test_timespan = self.x_testdata_clean[['Time']].loc[0:self.x_testdata_clean.shape[0]-2].reset_index(drop = True).values

        opt = ps.STLSQ(threshold = 0.3)
        model = ps.SINDy(feature_names = self.feature_names, optimizer = opt,
                                differentiation_method = ps.SmoothedFiniteDifference()._differentiate)
        print("Training model...")
        print("Model with SmoothedFiniteDifference & Without Ensembling")
        model.fit(x_train_noisy, t = self.dt)
        model.print()
        print("")
        self.x_test_sim_noisy_smoothdif = model.simulate(x_test_noisy[0, :], test_timespan.flatten(), integrator = "odeint")

        print("Training model...")
        model.fit(x_train_noisy, t = self.dt, ensemble=True, quiet=True)
        print("Model with SmoothedFiniteDifference & Ensembling")
        model.print()
        print("")
        self.x_test_sim_noisy_smoothdif_ensemble = model.simulate(x_test_noisy[0, :], test_timespan.flatten(), integrator = "odeint")

        ensemble_coefs = model.coef_list
        mean_ensemble = np.mean(ensemble_coefs, axis=0)
        std_ensemble = np.std(ensemble_coefs, axis=0)
        print("Plotting Ensembling Effect plot...")
        self.plot_ensemble_results(model, mean_ensemble, std_ensemble)

        print("-> EnsembleEffect Node Executed...")
    
    def ModelTrainEval(self, threshold_clean, threshold_noisy):
        print("-> ModelTrainEval Node Runnig...\n")
        x_train_clean = self.x_traindata_clean[['X', 'Y', 'Z']].loc[0:self.x_traindata_clean.shape[0]-2].reset_index(drop = True).values 
        x_train_noisy = self.x_traindata_noisy[['X', 'Y', 'Z']].loc[0:self.x_traindata_noisy.shape[0]-2].reset_index(drop = True).values
        x_test_clean = self.x_testdata_clean[['X', 'Y', 'Z']].loc[0:self.x_testdata_clean.shape[0]-2].reset_index(drop = True).values
        x_test_noisy = self.x_testdata_noisy[['X', 'Y', 'Z']].loc[0:self.x_testdata_noisy.shape[0]-2].reset_index(drop = True).values
        test_timespan = self.x_testdata_clean[['Time']].loc[0:self.x_testdata_clean.shape[0]-2].reset_index(drop = True).values

        self.init_x_clean = x_test_clean[0,:]
        print(f"Initial points for test with clean data ---->\n x = {self.init_x_clean[0]}\n y = {self.init_x_clean[1]}\n z = {self.init_x_clean[2]}\n")

        self.init_x_noisy = x_test_noisy[0,:]
        print(f"Initial points for test with noisy data ---->\n x = {self.init_x_noisy[0]}\n y = {self.init_x_noisy[1]}\n z = {self.init_x_noisy[2]}\n")

        library_functions = [lambda x: x, lambda x, y: x * y, lambda x: x ** 2]
        library_function_names = [lambda x: x, lambda x, y: x + y, lambda x: x + x]

        ode_lib = ps.WeakPDELibrary(
            library_functions = library_functions,
            function_names = library_function_names,
            spatiotemporal_grid = test_timespan,
            include_bias = True
        )

        opt_clean = ps.STLSQ(threshold = threshold_clean)
        opt_noisy = ps.STLSQ(threshold = threshold_noisy)

        self.model_clean = ps.SINDy(feature_names = self.feature_names, optimizer = opt_clean)
        self.model_noisy = ps.SINDy(feature_names = self.feature_names, feature_library = ode_lib, optimizer = opt_noisy,
                                differentiation_method = ps.SmoothedFiniteDifference()._differentiate)
        
        print("Tranining Model...")
        self.model_clean.fit(x_train_clean, t = self.dt, quiet = True)
        self.model_noisy.fit(x_train_noisy, t = self.dt, ensemble = True, quiet = True)

        print("Model with Clean Data ---->")
        self.model_clean.print()
        print("")
        print("Model with Noisy Data ---->")
        self.model_noisy.print()
        print("")

        print("Simulating Models...")
        self.x_test_sim_clean = self.model_clean.simulate(x_test_clean[0, :], test_timespan.flatten(), integrator = "odeint")
        
        def lorenz_equations(t, y):
            x, y, z = y
            dx_dt = -10.00 * x + 10.001 * y
            dy_dt = 27.995 * x -1.001*y -1.000*x*z
            dz_dt = -2.66*z + 1.000*x*y
            return [dx_dt, dy_dt, dz_dt]

        self.x_test_sim_noisy_smoothdif_ensemble_weakform = solve_ivp(
            lorenz_equations, (test_timespan[0], test_timespan[-1]), x_test_noisy[0, :], t_eval = test_timespan.flatten(), **integrator_keywords
        ).y.T
        
        #self.x_test_sim_noisy = self.model_noisy.simulate(x_test_noisy[0, :], test_timespan.flatten(), integrator = "odeint")
        print("-> ModelTrainEval Node Executed...")

    def ModelPlots(self):
        print("-> ModelPlots Node Runnig...\n")
        test_timespan = self.x_testdata_clean[['Time']].loc[0:self.x_testdata_clean.shape[0]-2].reset_index(drop = True).values
        x_test_out_clean = self.x_testdata_clean[['X', 'Y', 'Z']].loc[1:self.x_testdata_clean.shape[0]-1].reset_index(drop = True).values
        x_test_out_noisy = self.x_testdata_noisy[['X', 'Y', 'Z']].loc[1:self.x_testdata_noisy.shape[0]-1].reset_index(drop = True).values

        figsize = (40, 12)
        title_font = 24
        xlabel_font = 20
        ylabel_font = 20
        ticks_font = 16
        legend_font = 18

        fig1 = plt.figure(figsize = figsize)
        fig1ax1 = fig1.add_subplot(1,3,1)
        fig1ax2 = fig1.add_subplot(1,3,2)
        fig1ax3 = fig1.add_subplot(1,3,3)
        fig1ax1.plot(test_timespan, x_test_out_clean[:,0], color = 'red', label = 'Act.')
        fig1ax1.plot(test_timespan, self.x_test_sim_clean[:,0], color = 'blue', label = 'Pred.')
        fig1ax1.set_title("X", fontweight="bold", fontsize = title_font)
        fig1ax1.set_xlabel('Time (sec)', fontsize = xlabel_font)
        fig1ax1.set_ylabel("x", fontsize = ylabel_font)
        fig1ax1.tick_params(labelsize = ticks_font)
        fig1ax1.legend(loc = 'upper right', fontsize = legend_font)
        fig1ax1.grid(linestyle='--')

        fig1ax2.plot(test_timespan, x_test_out_clean[:,1], color = 'red', label = 'Act.')
        fig1ax2.plot(test_timespan, self.x_test_sim_clean[:,1], color = 'blue', label = 'Pred.')
        fig1ax2.set_title("Y", fontweight="bold", fontsize = title_font)
        fig1ax2.set_xlabel('Time (sec)', fontsize = xlabel_font)
        fig1ax2.set_ylabel("y", fontsize = ylabel_font)
        fig1ax2.tick_params(labelsize = ticks_font)
        fig1ax2.legend(loc = 'upper right', fontsize = legend_font)
        fig1ax2.grid(linestyle='--')

        fig1ax3.plot(test_timespan, x_test_out_clean[:,2], color = 'red', label = 'Act.')
        fig1ax3.plot(test_timespan, self.x_test_sim_clean[:,2], color = 'blue', label = 'Pred.')
        fig1ax3.set_title("Z", fontweight="bold", fontsize = title_font)
        fig1ax3.set_xlabel('Time (sec)', fontsize = xlabel_font)
        fig1ax3.set_ylabel("z", fontsize = ylabel_font)
        fig1ax3.tick_params(labelsize = ticks_font)
        fig1ax3.legend(loc = 'upper right', fontsize = legend_font)
        fig1ax3.grid(linestyle='--')
        fig1.savefig(f"./LorenzSystem/Figures/SINDyFigures/Actual vs Predicted (Clean Data).png")

        fig2 = plt.figure(figsize = figsize)
        fig2ax1 = fig2.add_subplot(1,3,1)
        fig2ax2 = fig2.add_subplot(1,3,2)
        fig2ax3 = fig2.add_subplot(1,3,3)
        fig2ax1.plot(test_timespan, x_test_out_noisy[:,0], color = 'red', label = 'Act.')
        fig2ax1.plot(test_timespan, self.x_test_sim_noisy_finitedif[:,0], color = 'blue', label = 'Pred.')
        fig2ax1.set_title("X", fontweight="bold", fontsize = title_font)
        fig2ax1.set_xlabel('Time (sec)', fontsize = xlabel_font)
        fig2ax1.set_ylabel("x", fontsize = ylabel_font)
        fig2ax1.tick_params(labelsize = ticks_font)
        fig2ax1.legend(loc = 'upper right', fontsize = legend_font)
        fig2ax1.grid(linestyle='--')

        fig2ax2.plot(test_timespan, x_test_out_noisy[:,1], color = 'red', label = 'Act.')
        fig2ax2.plot(test_timespan, self.x_test_sim_noisy_finitedif[:,1], color = 'blue', label = 'Pred.')
        fig2ax2.set_title("Y", fontweight="bold", fontsize = title_font)
        fig2ax2.set_xlabel('Time (sec)', fontsize = xlabel_font)
        fig2ax2.set_ylabel("y", fontsize = ylabel_font)
        fig2ax2.tick_params(labelsize = ticks_font)
        fig2ax2.legend(loc = 'upper right', fontsize = legend_font)
        fig2ax2.grid(linestyle='--')

        fig2ax3.plot(test_timespan, x_test_out_noisy[:,2], color = 'red', label = 'Act.')
        fig2ax3.plot(test_timespan, self.x_test_sim_noisy_finitedif[:,2], color = 'blue', label = 'Pred.')
        fig2ax3.set_title("Z", fontweight="bold", fontsize = title_font)
        fig2ax3.set_xlabel('Time (sec)', fontsize = xlabel_font)
        fig2ax3.set_ylabel("z", fontsize = ylabel_font)
        fig2ax3.tick_params(labelsize = ticks_font)
        fig2ax3.legend(loc = 'upper right', fontsize = legend_font)
        fig2ax3.grid(linestyle='--')
        fig2.savefig(f"./LorenzSystem/Figures/SINDyFigures/Actual vs Predicted with Finite Difference (Noisy Data).png")

        fig3 = plt.figure(figsize = figsize)
        fig3ax1 = fig3.add_subplot(1,3,1)
        fig3ax2 = fig3.add_subplot(1,3,2)
        fig3ax3 = fig3.add_subplot(1,3,3)
        fig3ax1.plot(test_timespan, x_test_out_noisy[:,0], color = 'red', label = 'Act.')
        fig3ax1.plot(test_timespan, self.x_test_sim_noisy_smoothdif[:,0], color = 'blue', label = 'Pred.')
        fig3ax1.set_title("X", fontweight="bold", fontsize = title_font)
        fig3ax1.set_xlabel('Time (sec)', fontsize = xlabel_font)
        fig3ax1.set_ylabel("x", fontsize = ylabel_font)
        fig3ax1.tick_params(labelsize = ticks_font)
        fig3ax1.legend(loc = 'upper right', fontsize = legend_font)
        fig3ax1.grid(linestyle='--')

        fig3ax2.plot(test_timespan, x_test_out_noisy[:,1], color = 'red', label = 'Act.')
        fig3ax2.plot(test_timespan, self.x_test_sim_noisy_smoothdif[:,1], color = 'blue', label = 'Pred.')
        fig3ax2.set_title("Y", fontweight="bold", fontsize = title_font)
        fig3ax2.set_xlabel('Time (sec)', fontsize = xlabel_font)
        fig3ax2.set_ylabel("y", fontsize = ylabel_font)
        fig3ax2.tick_params(labelsize = ticks_font)
        fig3ax2.legend(loc = 'upper right', fontsize = legend_font)
        fig3ax2.grid(linestyle='--')

        fig3ax3.plot(test_timespan, x_test_out_noisy[:,2], color = 'red', label = 'Act.')
        fig3ax3.plot(test_timespan, self.x_test_sim_noisy_smoothdif[:,2], color = 'blue', label = 'Pred.')
        fig3ax3.set_title("Z", fontweight="bold", fontsize = title_font)
        fig3ax3.set_xlabel('Time (sec)', fontsize = xlabel_font)
        fig3ax3.set_ylabel("z", fontsize = ylabel_font)
        fig3ax2.tick_params(labelsize = ticks_font)
        fig3ax3.legend(loc = 'upper right', fontsize = legend_font)
        fig3ax3.grid(linestyle='--')
        fig3.savefig(f"./LorenzSystem/Figures/SINDyFigures/Actual vs Predicted with Smoothed Finite Difference (Noisy Data).png")

        fig4 = plt.figure(figsize = figsize)
        fig4ax1 = fig4.add_subplot(1,3,1)
        fig4ax2 = fig4.add_subplot(1,3,2)
        fig4ax3 = fig4.add_subplot(1,3,3)
        fig4ax1.plot(test_timespan, x_test_out_noisy[:,0], color = 'red', label = 'Act.')
        fig4ax1.plot(test_timespan, self.x_test_sim_noisy_smoothdif_ensemble[:,0], color = 'blue', label = 'Pred.')
        fig4ax1.set_title("X", fontweight="bold", fontsize = title_font)
        fig4ax1.set_xlabel('Time (sec)', fontsize = xlabel_font)
        fig4ax1.set_ylabel("x", fontsize = ylabel_font)
        fig4ax1.tick_params(labelsize = ticks_font)
        fig4ax1.legend(loc = 'upper right', fontsize = legend_font)
        fig4ax1.grid(linestyle='--')

        fig4ax2.plot(test_timespan, x_test_out_noisy[:,1], color = 'red', label = 'Act.')
        fig4ax2.plot(test_timespan, self.x_test_sim_noisy_smoothdif_ensemble[:,1], color = 'blue', label = 'Pred.')
        fig4ax2.set_title("Y", fontweight="bold", fontsize = title_font)
        fig4ax2.set_xlabel('Time (sec)', fontsize = xlabel_font)
        fig4ax2.set_ylabel("y", fontsize = ylabel_font)
        fig4ax2.tick_params(labelsize = ticks_font)
        fig4ax2.legend(loc = 'upper right', fontsize = legend_font)
        fig4ax2.grid(linestyle='--')

        fig4ax3.plot(test_timespan, x_test_out_noisy[:,2], color = 'red', label = 'Act.')
        fig4ax3.plot(test_timespan, self.x_test_sim_noisy_smoothdif_ensemble[:,2], color = 'blue', label = 'Pred.')
        fig4ax3.set_title("Z", fontweight="bold", fontsize = title_font)
        fig4ax3.set_xlabel('Time (sec)', fontsize = xlabel_font)
        fig4ax3.set_ylabel("z", fontsize = ylabel_font)
        fig4ax3.tick_params(labelsize = ticks_font)
        fig4ax3.legend(loc = 'upper right', fontsize = legend_font)
        fig4ax3.grid(linestyle='--')
        fig4.savefig(f"./LorenzSystem/Figures/SINDyFigures/Actual vs Predicted with Smoothed Finite Difference & Ensemble(Noisy Data).png")

        fig5 = plt.figure(figsize = figsize)
        fig5ax1 = fig5.add_subplot(1,3,1)
        fig5ax2 = fig5.add_subplot(1,3,2)
        fig5ax3 = fig5.add_subplot(1,3,3)
        fig5ax1.plot(test_timespan, x_test_out_noisy[:,0], color = 'red', label = 'Act.')
        fig5ax1.plot(test_timespan, self.x_test_sim_noisy_smoothdif_ensemble_weakform[:,0], color = 'blue', label = 'Pred.')
        fig5ax1.set_title("X", fontweight="bold", fontsize = title_font)
        fig5ax1.set_xlabel('Time (sec)', fontsize = xlabel_font)
        fig5ax1.set_ylabel("x", fontsize = ylabel_font)
        fig5ax1.tick_params(labelsize = ticks_font)
        fig5ax1.legend(loc = 'upper right', fontsize = legend_font)
        fig5ax1.grid(linestyle='--')

        fig5ax2.plot(test_timespan, x_test_out_noisy[:,1], color = 'red', label = 'Act.')
        fig5ax2.plot(test_timespan, self.x_test_sim_noisy_smoothdif_ensemble_weakform[:,1], color = 'blue', label = 'Pred.')
        fig5ax2.set_title("Y", fontweight="bold", fontsize = title_font)
        fig5ax2.set_xlabel('Time (sec)', fontsize = xlabel_font)
        fig5ax2.set_ylabel("y", fontsize = ylabel_font)
        fig5ax2.tick_params(labelsize = ticks_font)
        fig5ax2.legend(loc = 'upper right', fontsize = legend_font)
        fig5ax2.grid(linestyle='--')

        fig5ax3.plot(test_timespan, x_test_out_noisy[:,2], color = 'red', label = 'Act.')
        fig5ax3.plot(test_timespan, self.x_test_sim_noisy_smoothdif_ensemble_weakform[:,2], color = 'blue', label = 'Pred.')
        fig5ax3.set_title("Z", fontweight="bold", fontsize = title_font)
        fig5ax3.set_xlabel('Time (sec)', fontsize = xlabel_font)
        fig5ax3.set_ylabel("z", fontsize = ylabel_font)
        fig5ax3.tick_params(labelsize = ticks_font)
        fig5ax3.legend(loc = 'upper right', fontsize = legend_font)
        fig5ax3.grid(linestyle='--')
        fig5.savefig(f"./LorenzSystem/Figures/SINDyFigures/Actual vs Predicted with Smoothed Finite Difference & Ensemble & WeakForm (Noisy Data).png")

        fig6 = plt.figure(figsize = (10, 8))
        ax = fig6.add_subplot(111, projection='3d')
        ax.plot(x_test_out_noisy[:, 0], x_test_out_noisy[:, 1], x_test_out_noisy[:, 2], color='blue', label='Noisy Data')
        ax.plot(self.x_test_sim_noisy_smoothdif_ensemble_weakform[:, 0], self.x_test_sim_noisy_smoothdif_ensemble_weakform[:, 1], self.x_test_sim_noisy_smoothdif_ensemble_weakform[:, 2], color='purple', label='Simulation')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        ax.set_title('Lorenz Attractor (Noisy Data vs Simulation)')
        fig6.savefig(f"./LorenzSystem/Figures/SINDyFigures/Actual vs Predicted Attractor (Noisy Data).png")

        print("-> ModelPlots Node Executed...")
        
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

    def DifferentiatorPlot(self, x, dt, deriv, diff_name):
        plt.figure(figsize = (40, 12))
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
        plt.figure(figsize = (40, 12))
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
    
    def plot_ensemble_results(self, model, mean_ensemble, std_ensemble):
        # Plot results
        xticknames = model.get_feature_names()
        for i in range(10):
            xticknames[i] = "$" + xticknames[i] + "$"
        plt.figure(figsize=(15, 8))
        colors = ["b", "r", "k"]
        plt.xlabel("Candidate terms", fontsize = 12)
        plt.ylabel("Coefficient values", fontsize = 12)
        for i in range(3):
            plt.errorbar(
                range(10),
                mean_ensemble[i, :],
                yerr=std_ensemble[i, :],
                fmt="o",
                color=colors[i],
                label=f"Equation for $\\dot{{{self.feature_names[i]}}}$",
            )
        ax = plt.gca()
        plt.grid(True)
        ax.set_xticks(range(10))
        plt.xticks(fontsize = 15)
        plt.yticks(fontsize = 15)
        ax.set_xticklabels(xticknames, verticalalignment="top")
        plt.legend(fontsize=18)
        plt.title("Error Bounds & Coefficient Values with Ensembling", fontsize = 20)
        plt.savefig(f"./LorenzSystem/Figures/SINDyFigures/EnsembleEffect.png")
    
if __name__ == "__main__":

    # Create Lorenz System Object 
    LorenzSys = LorenzSINDy()

    # To see hyperparameter effect
    LorenzSys.HyperparameterEffect()

    #To see differentiator effect
    LorenzSys.DifferentiatorEffect()

    # To see ensemble effect
    LorenzSys.EnsembleEffect()

    # SINDy Model outputs for both clean and noisy data
    LorenzSys.ModelTrainEval(threshold_clean = 0.1, threshold_noisy = 0.3)
    LorenzSys.ModelPlots()