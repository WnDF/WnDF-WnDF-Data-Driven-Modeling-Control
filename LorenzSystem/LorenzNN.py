import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
import Lorenz as LS
warnings.filterwarnings("ignore")

class LorenzSystemNNModel(torch.nn.Module):
    def __init__(self, RANDOM_SEED = int()):
        super().__init__()

        self.RANDOM_SEED = RANDOM_SEED
        self.optimizer = torch.optim
        self.lossfunction = torch.nn
        self.lr = torch.float

        self.LS = None
        self.is_noisy = str()
        self.dataset_dataframe = pd.DataFrame()
        self.prediction_dataframe = pd.DataFrame()
        self.metricesbyepochs = pd.DataFrame()

        self.input_train_data = torch.tensor
        self.input_test_data = torch.tensor
        self.input_validate_data = torch.tensor

        self.output_train_data = torch.tensor
        self.output_test_data = torch.tensor
        self.output_validate_data = torch.tensor

        self.train_timespan_data = torch.tensor
        self.test_timespan_data = torch.tensor
        self.validate_timespan = torch.tensor

        self.model = torch.nn.Sequential(torch.nn.Linear(in_features=3, out_features=128),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(in_features=128, out_features=64),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(in_features=64, out_features=128),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(in_features=128, out_features=3))

    def DataGeneration(self, noisy = bool(), is_traning_data = bool(), sigma=10, rho=28, beta=8/3,
                        t_stop = 10, sim_step_size = 100, dt = 0.01):
        
        if noisy:
            self.is_noisy = '(Noise = Noisy)'
        else:
            self.is_noisy = '(Noise = Normal)'

        self.LS = LS.LorenzSystem(sigma = sigma, rho = rho, beta = beta, noisy = noisy)
        
        np.random.seed(self.RANDOM_SEED)
        dataset_dataframe = pd.DataFrame()

        for i in range(1, sim_step_size):
            init_x, init_y, init_z = 10*np.random.rand(3)
            init_states = [init_x, init_y, init_z]

            t_span = (0, t_stop)
            t_eval = np.arange(0, t_stop, dt)

            df = self.LS.simulate(init_states, t_span, t_eval)
            dataset_dataframe = pd.concat([dataset_dataframe, df], axis = 0)
        
        dataset_dataframe = dataset_dataframe.reset_index(drop=True)

        input_data = dataset_dataframe[['X', 'Y', 'Z']].loc[0:dataset_dataframe.shape[0]-2].reset_index(drop = True)
        output_data = dataset_dataframe[['X', 'Y', 'Z']].loc[1:dataset_dataframe.shape[0]-1].reset_index(drop = True)
        timespan = dataset_dataframe[['Time']].loc[0:dataset_dataframe.shape[0]-2].reset_index(drop = True)

        df_input_output = pd.DataFrame({'Time': timespan['Time'], 
                                        'x(t)': input_data['X'], 'y(t)': input_data['Y'], 'z(t)': input_data['Z'],
                                        'x(t + dt)':output_data['X'], 'y(t+dt)': output_data['Y'], 'z(t + dt)': output_data['Z']})  

        input_data = torch.from_numpy(np.array(input_data)).type(torch.float)
        output_data = torch.from_numpy(np.array(output_data)).type(torch.float)
        timespan = torch.from_numpy(np.array(timespan)).type(torch.float)

        if is_traning_data:
            self.dataset_dataframe = df_input_output
            self.input_train_data, self.input_test_data, self.output_train_data, self.output_test_data, self.train_timespan_data, self.test_timespan_data = train_test_split(input_data, 
                                                                                                                                                                            output_data, 
                                                                                                                                                                            timespan,
                                                                                                                                                                            test_size=0.2,
                                                                                                                                                                            random_state = np.random.seed(self.RANDOM_SEED))
            self.SaveSimulationData(data = self.dataset_dataframe, PATH = f'./LorenzSystem/Dataset/NNDataset/LorenzSystemTrainingDataset{self.is_noisy}.csv')
        
        return self.is_noisy, input_data, output_data, timespan
        

    def SaveSimulationData(self, data = pd.DataFrame(), PATH = str()):
        data.to_csv(PATH, encoding='utf-8')

    def LossFunction(self, lossfunction = torch.nn.MSELoss()):
        self.lossfunction = lossfunction

    def Optimizer(self, optimizer = torch.optim.Adam, lr = 0.001):
        self.optimizer = optimizer(self.model.parameters(), lr = lr)
        self.lr = lr

    def AccuracyFunc(self, y_true = torch.tensor, y_pred = torch.tensor):
        acc_rates = []

        # Iterate over each column
        for i in range(y_true.shape[1]):
            true_col = y_true[:, i]
            pred_col = y_pred[:, i]

            abs_error_percentage = torch.mean(torch.abs((true_col - pred_col)/true_col)*100)
            acc_rates.append(abs_error_percentage)

        return acc_rates

    def Train(self, epochs = int()):
        np.random.seed(self.RANDOM_SEED)

        for epoch in range(epochs):
            self.model.train()

            y_preds = self.model(self.input_train_data)
            loss = self.lossfunction(y_preds, self.output_train_data)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.model.eval()

            with torch.inference_mode():
                test_preds = self.model(self.input_test_data)
                test_loss = self.lossfunction(test_preds, self.output_test_data)
                test_acc = self.AccuracyFunc(y_true = self.output_test_data,
                                            y_pred = test_preds)
                
                dfmetrices = pd.DataFrame({'Epoch': epoch, 'Lr': [self.optimizer.param_groups[0]['lr']],
                                           'x_ErrorRate': test_acc[0], 'y_ErrorRate': test_acc[1], 'z_ErrorRate': test_acc[2], 
                                           'TrainLoss': loss, 'TestLoss': test_loss})
                self.metricesbyepochs = pd.concat([self.metricesbyepochs, dfmetrices], axis = 0)
            
            if epoch % 10 == 0:
                print(f"Epoch: {epoch} | Loss: {loss:.5f} | Test Loss: {test_loss:.5f} | x_Error: {test_acc[0]:.2f}% | y_Error: {test_acc[1]:.2f}% | z_Error: {test_acc[2]:.2f}%")

        self.SaveModel(PATH = f'./LorenzSystem/TrainedModels/LSNNModel{self.is_noisy}.pth')

    def ModelEvaluation(self, noisy, is_traning_data):
        self.model.eval()

        with torch.inference_mode():
            is_noisy, self.input_validate_data, self.output_validate_data, self.validate_timespan = self.DataGeneration(noisy = noisy, is_traning_data = is_traning_data)

            predictions = self.model(self.input_validate_data).numpy()
            self.output_validate_data = self.output_validate_data.numpy()
            timespan = self.validate_timespan.numpy()[:,0]

            x = self.output_validate_data[:,0]
            x_pred = predictions[:,0]
            y = self.output_validate_data[:,1]
            y_pred = predictions[:,1]
            z = self.output_validate_data[:,2]
            z_pred = predictions[:,2]

        pred_df = pd.DataFrame({'Time': timespan, 'x': x, 'y': y, 'z': z,
                                'x_pred': x_pred, 'y_pred': y_pred,'z_pred': z_pred})
        
        self.prediction_dataframe = pd.concat([self.prediction_dataframe, pred_df], axis = 1)
            
        Accuracy = self.AccuracyFunc(y_pred = torch.from_numpy(predictions), y_true = torch.from_numpy(self.output_validate_data))

        print(f"\nValidation Result ------->\n X Error Rate: {Accuracy[0]:.5f}\n Y Error Rate: {Accuracy[1]:.5f}\n Z Error Rate: {Accuracy[2]:.5f}")

        self.SaveSimulationData(data = self.prediction_dataframe, PATH = f'./LorenzSystem/Dataset/NNDataset/LorenzSystemEvaluationDataset{is_noisy}.csv')

    def forward(self, x):
        return self.model(x)
    
    def SaveModel(self, PATH = str()):
        torch.save(self.model, PATH)

    def DataPlots(self):
        x = self.prediction_dataframe['x']
        y = self.prediction_dataframe['y']
        z = self.prediction_dataframe['z']

        x_pred = self.prediction_dataframe['x_pred']
        y_pred = self.prediction_dataframe['y_pred']
        z_pred = self.prediction_dataframe['z_pred']

        t_span = self.validate_timespan

        epochs = self.metricesbyepochs['Epoch']
        TrainLoss = self.metricesbyepochs['TrainLoss']
        TestLoss = self.metricesbyepochs['TestLoss']
        lr = self.metricesbyepochs['Lr']

        x_ErrorRate = self.metricesbyepochs['x_ErrorRate']
        y_ErrorRate = self.metricesbyepochs['y_ErrorRate']
        z_ErrorRate = self.metricesbyepochs['z_ErrorRate']
        
        FigSize = (35, 12)
        dotsize = 8
        linewidth = 3
        title_font = 24
        xlabel_font = 20
        ylabel_font = 20
        ticks_font = 16
        legend_font = 18

        fig1 = plt.figure(figsize = FigSize)
        fig1ax1 = fig1.add_subplot(1,3,1)
        fig1ax2 = fig1.add_subplot(1,3,2)
        fig1ax3 = fig1.add_subplot(1,3,3)

        fig1ax1.plot(t_span, x_pred, color = 'red', label = 'Pred.')
        fig1ax1.plot(t_span, x, color = 'blue', label = 'True')
        fig1ax1.set_title("X", fontweight="bold", fontsize = 24)
        fig1ax1.set_xlabel('Time (sec)', fontsize = 20)
        fig1ax1.set_ylabel("x", fontsize = 20)
        fig1ax1.tick_params(labelsize = 16)
        fig1ax1.legend(loc = 'upper right', fontsize = 18)
        fig1ax1.grid(linestyle='--')

        fig1ax2.plot(t_span, y_pred, color = 'red', label = 'Pred.')
        fig1ax2.plot(t_span, y, color = 'blue', label = 'True.')
        fig1ax2.set_title("Y", fontweight="bold", fontsize = 24)
        fig1ax2.set_xlabel('Time (sec)', fontsize = 20)
        fig1ax2.set_ylabel("y", fontsize = 20)
        fig1ax2.tick_params(labelsize = 16)
        fig1ax2.legend(loc = 'upper right', fontsize = 18)
        fig1ax2.grid(linestyle='--')

        fig1ax3.plot(t_span, z_pred, color = 'red', label = 'Pred.')
        fig1ax3.plot(t_span, z, color = 'blue', label = 'True.')
        fig1ax3.set_title("Z", fontweight="bold", fontsize = 24)
        fig1ax3.set_xlabel('Time (sec)', fontsize = 20)
        fig1ax3.set_ylabel("z", fontsize = 20)
        fig1ax3.tick_params(labelsize = 16)
        fig1ax3.legend(loc = 'upper right', fontsize = 18)
        fig1ax3.grid(linestyle='--')
        fig1.savefig(f"./LorenzSystem/Figures/NNFigures/XYZ - {self.is_noisy}.png")

        fig2 = plt.figure(figsize = (60,50))
        fig2ax1 = fig2.add_subplot(3,1,1)
        fig2ax2 = fig2.add_subplot(3,1,2)
        fig2ax3 = fig2.add_subplot(3,1,3)

        fig2ax1.plot(epochs, TestLoss, color = 'blue', label = 'Test Loss', linewidth=linewidth)
        fig2ax1.set_title('Test Loss by Epoch', fontweight="bold", fontsize = 50)
        fig2ax1.set_xlabel('Epoch', fontsize = 35)
        fig2ax1.set_ylabel('Loss', fontsize = 35)
        fig2ax1.tick_params(labelsize = 35)
        fig2ax1.legend(loc = 'upper right', fontsize = 35)
        fig2ax1.grid(linestyle='--', linewidth=linewidth)

        fig2ax2.plot(epochs, TrainLoss, color = 'blue', label = 'Train Loss', linewidth=linewidth)
        fig2ax2.set_title('Train Loss by Epoch', fontweight="bold", fontsize = 50)
        fig2ax2.set_xlabel('Epoch', fontsize = 35)
        fig2ax2.set_ylabel('Loss', fontsize = 35)
        fig2ax2.tick_params(labelsize = 35)
        fig2ax2.legend(loc = 'upper right', fontsize = 35)
        fig2ax2.grid(linestyle='--', linewidth=linewidth)

        fig2ax3.plot(epochs, lr, color = 'blue', label = 'Lr', linewidth=linewidth)
        fig2ax3.set_title('Learning Rate by Epoch', fontweight="bold", fontsize = 50)
        fig2ax3.set_xlabel('Epoch', fontsize = 35)
        fig2ax3.set_ylabel('Lr', fontsize = 35)
        fig2ax3.tick_params(labelsize = 35)
        fig2ax3.legend(loc = 'upper right', fontsize = 35)
        fig2ax3.grid(linestyle='--', linewidth=linewidth)
        fig2.savefig(f"./LorenzSystem/Figures/NNFigures/Loss&LearningRate{self.is_noisy}.png")

        fig3 = plt.figure(figsize = (60,50))
        fig3ax1 = fig3.add_subplot(3,1,1)
        fig3ax2 = fig3.add_subplot(3,1,2)
        fig3ax3 = fig3.add_subplot(3,1,3)

        fig3ax1.plot(epochs, x_ErrorRate, color = 'blue', label = 'Error. Rate (%)', linewidth=linewidth)
        fig3ax1.set_title('Prediction Error Rate - X' , fontweight="bold", fontsize = 50)
        fig3ax1.set_xlabel('Epoch', fontsize = 35)
        fig3ax1.set_ylabel('Error (%)', fontsize = 35)
        fig3ax1.tick_params(labelsize = 35)
        fig3ax1.legend(loc = 'upper right', fontsize = 35)
        fig3ax1.grid(linestyle='--', linewidth=linewidth)

        fig3ax2.plot(epochs, y_ErrorRate, color = 'blue', label = 'Error Rate (%)', linewidth=linewidth)
        fig3ax2.set_title('Prediction Error Rate - Y' , fontweight="bold", fontsize = 50)
        fig3ax2.set_xlabel('Epoch', fontsize = 35)
        fig3ax2.set_ylabel('Error (%)', fontsize = 35)
        fig3ax2.tick_params(labelsize = 35)
        fig3ax2.legend(loc = 'upper right', fontsize = 35)
        fig3ax2.grid(linestyle='--', linewidth=linewidth)

        fig3ax3.plot(epochs, z_ErrorRate, color = 'blue', label = 'Error Rate (%)', linewidth=linewidth)
        fig3ax3.set_title('Prediction Error Rate - Z' , fontweight="bold", fontsize = 50)
        fig3ax3.set_xlabel('Epoch', fontsize = 35)
        fig3ax3.set_ylabel('Error (%)', fontsize = 35)
        fig3ax3.tick_params(labelsize = 35)
        fig3ax3.legend(loc = 'upper right', fontsize = 35)
        fig3ax3.grid(linestyle='--', linewidth=linewidth)
        fig3.savefig(f"./LorenzSystem/Figures/NNFigures/ErrorRates{self.is_noisy}.png")

        fig4 = plt.figure(figsize = (15, 12))
        ax = fig4.add_subplot(111, projection='3d')
        ax.plot(x, y, z, color='blue', label=f'Act.')
        ax.plot(x_pred, y_pred, z_pred, color='purple', label='Simulation')
        ax.set_xlabel('X', fontsize = 20)
        ax.set_ylabel('Y', fontsize = 20)
        ax.set_zlabel('Z', fontsize = 20)
        ax.legend(loc = 'upper right', fontsize = 18)
        ax.tick_params(labelsize = 16)
        ax.set_title('Lorenz Attractor (Clean Data vs Simulation)', fontsize = 24)
        fig4.savefig(f"./LorenzSystem/Figures/NNFigures/Actual vs Predicted Attractor {self.is_noisy}.png") 

if __name__ == "__main__":
    LorenzModel = LorenzSystemNNModel(RANDOM_SEED = 52)
    LorenzModel.DataGeneration(t_stop = 10, sim_step_size = 3, dt = 0.01, noisy = True, is_traning_data = True)
    LorenzModel.LossFunction(lossfunction = torch.nn.MSELoss())
    LorenzModel.Optimizer(optimizer = torch.optim.Adam, lr = 0.001)
    LorenzModel.Train(epochs = 20)
    LorenzModel.ModelEvaluation(noisy = True, is_traning_data = False)
    LorenzModel.DataPlots()