import torch
import matplotlib.pyplot as plt
import DoublePendulum as dp
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

class DoublePendulumnNNModel(torch.nn.Module):
    def __init__(self, RANDOM_SEED = int()):
        super().__init__()

        self.RANDOM_SEED = RANDOM_SEED
        self.optimizer = torch.optim
        self.lossfunction = torch.nn
        self.lr = torch.float

        self.DP = None
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

        self.model = torch.nn.Sequential(torch.nn.Linear(in_features=4, out_features=1024),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(in_features=1024, out_features=512),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(in_features=512, out_features=1024),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(in_features=1024, out_features=512),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(in_features=512, out_features=1024),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(in_features=1024, out_features=4))
        
    def DataGeneration(self, noisy = bool(), is_traning_data = bool(), t_stop = 10, dt = 0.001, sim_step_size=1, 
                         m1 = 0.2704, m2 = 0.2056, cg1 = 0.191, cg2 = 0.1621, 
                         L1 = 0.2667, L2 = 0.2667, I1 = 0.003, I2 = 0.0011, g = 9.81):
        
        if noisy:
            self.is_noisy = '(Noise = Noisy)'
        else:
            self.is_noisy = '(Noise = Normal)'
        
        self.DP = dp.DoublePendulumSS(m1 = m1, m2 = m2, cg1 = cg1, cg2 = cg2, L1 = L1, L2 = L2, I1 = I1, I2 = I2, g = g, noisy = noisy)
        np.random.seed(self.RANDOM_SEED)

        dataset_dataframe = pd.DataFrame()

        for i in range(1, sim_step_size + 1):

            theta1_init = 180.0*np.random.rand()
            theta2_init = 180.0*np.random.rand()
            omega1_init = 180.0*np.random.rand()
            omega2_init = 180.0*np.random.rand()

            # initial states
            state = np.radians([theta1_init, theta2_init, omega1_init, omega2_init])

            t_span = (0, t_stop)
            t_eval = np.arange(0, t_stop, dt)

            df = self.DP.simulate(state, t_span, t_eval)
            dataset_dataframe = pd.concat([dataset_dataframe, df], axis = 0)
        
        dataset_dataframe = dataset_dataframe.reset_index(drop=True)

        input_data = dataset_dataframe[['Theta1', 'Theta2', 'Omega1', 'Omega2']].loc[0:dataset_dataframe.shape[0]-2].reset_index(drop = True)
        output_data = dataset_dataframe[['Theta1', 'Theta2', 'Omega1', 'Omega2']].loc[1:dataset_dataframe.shape[0]-1].reset_index(drop = True)
        timespan = dataset_dataframe[['Time']].loc[0:dataset_dataframe.shape[0]-2].reset_index(drop = True)

        df_input_output = pd.DataFrame({'Time': timespan['Time'], 
                                        'Theta1 (t)': input_data['Theta1'], 'Theta2 (t)': input_data['Theta2'], 'Omega1 (t)': input_data['Omega1'], 'Omega2 (t)': input_data['Omega2'],
                                        'Theta1 (t + dt)':output_data['Theta1'], 'Theta2 (t+dt)': output_data['Theta2'], 'Omega1 (t + dt)': output_data['Omega1'], 'Omega2 (t + dt)': output_data['Omega2']})  

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
            self.SaveSimulationData(data = self.dataset_dataframe, PATH = f'./DoublePedulum/Dataset/NNDataset/DoublePendulumTrainingDataset{self.is_noisy}.csv')
        
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

        for epoch in range(epochs+1):
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
                                           'Theta1Accuracy': test_acc[0], 'Theta2Accuracy': test_acc[1],
                                           'Omega1Accuracy': test_acc[2], 'Omega2Accuracy': test_acc[3],
                                           'TrainLoss': loss, 'TestLoss': test_loss})
                self.metricesbyepochs = pd.concat([self.metricesbyepochs, dfmetrices], axis = 0)
            
            if epoch % 10 == 0:
                print(f"Epoch: {epoch} | Loss: {loss:.5f} | Test Loss: {test_loss:.5f} | Theta1ErrorRate: {test_acc[0]:.2f}% | Theta2ErrorRate: {test_acc[1]:.2f}% | Omega1ErrorRate: {test_acc[2]:.2f}% | Omega2ErrorRate: {test_acc[3]:.2f}%")

        self.SaveModel(PATH = f'./DoublePedulum/TrainedModels/DPNNModel{self.is_noisy}.pth')
        
    def PoseByAxis(self, theta1_true = list(), theta1_pred = list(), theta2_true = list(), theta2_pred = list()):
        x1_true =  self.DP.L1*np.sin(theta1_true)
        y1_true = -self.DP.L1*np.cos(theta1_true)

        x2_true = self.DP.L2*np.sin(theta2_true) + x1_true
        y2_true = -self.DP.L2*np.cos(theta2_true) + y1_true

        x1_pred = self.DP.L1*np.sin(theta1_pred)
        y1_pred = -self.DP.L1*np.cos(theta1_pred)

        x2_pred = self.DP.L2*np.sin(theta2_pred) + x1_pred
        y2_pred = -self.DP.L2*np.cos(theta2_pred) + y1_pred

        return x1_true, y1_true, x2_true, y2_true, x1_pred, y1_pred, x2_pred, y2_pred

    def ModelEvaluation(self, noisy, is_traning_data, sim_step_size):

        self.model.eval()

        with torch.inference_mode():
            is_noisy, self.input_validate_data, self.output_validate_data, self.validate_timespan = self.DataGeneration(noisy = noisy, sim_step_size = sim_step_size, is_traning_data = is_traning_data)
            
            predictions = self.model(self.input_validate_data).numpy()
            self.output_validate_data = self.output_validate_data.numpy()
            timespan = self.validate_timespan.numpy()[:,0]

            Theta1 = self.output_validate_data[:,0]
            Theta1_pred = predictions[:,0]
            Theta2 = self.output_validate_data[:,1]
            Theta2_pred = predictions[:,1]
            Omega1= self.output_validate_data[:,2]
            Omega1_pred = predictions[:,2]
            Omega2 = self.output_validate_data[:,3]
            Omega2_pred = predictions[:,3]

            x1_true, y1_true, x2_true, y2_true, x1_pred, y1_pred, x2_pred, y2_pred = self.PoseByAxis(theta1_true = Theta1, 
                                                                                                     theta1_pred = Theta1_pred, 
                                                                                                     theta2_true = Theta2, 
                                                                                                     theta2_pred = Theta2_pred)

            pred_df = pd.DataFrame({'Time': timespan, 'Theta1': Theta1, 'Theta2': Theta2, 
                                    'Omega1': Omega1, 'Omega2': Omega2, 
                                    'x1_true': x1_true, 'y1_true': y1_true, 
                                    'x2_true':x2_true, 'y2_true':y2_true ,
                                    'Theta1_pred': Theta1_pred, 'Theta2_pred': Theta2_pred, 
                                    'Omega1_pred': Omega1_pred, 'Omega2_pred': Omega2_pred,
                                    'x1_pred': x1_pred, 'y1_pred': y1_pred,
                                    'x2_pred': x2_pred, 'y2_pred': y2_pred
                                    })
            
            self.prediction_dataframe = pd.concat([self.prediction_dataframe, pred_df], axis = 1)
            
            Accuracy = self.AccuracyFunc(y_pred = torch.from_numpy(predictions), y_true = torch.from_numpy(self.output_validate_data))

            print(f"\nValidation Result ------->\n Theta1 ErrorRate: {Accuracy[0]:.5f}\n Theta2 ErrorRate: {Accuracy[1]:.5f}\n Omega1 ErrorRate: {Accuracy[2]:.5f}\n Omega2 ErrorRate: {Accuracy[3]:.5f}")

            self.SaveSimulationData(data = self.prediction_dataframe, PATH = f'./DoublePedulum/Dataset/NNDataset/DoublePendulumEvaluationDataset{is_noisy}.csv')
        
    def forward(self, x):
        return self.model(x)
    
    def SaveModel(self, PATH = str()):
        torch.save(self.model, PATH)

    def DataPlots(self):

        Theta1 = self.prediction_dataframe['Theta1']
        Theta2 = self.prediction_dataframe['Theta2']
        Omega1= self.prediction_dataframe['Omega1']
        Omega2 = self.prediction_dataframe['Omega2']
        x1_true = self.prediction_dataframe['x1_true']
        y1_true = self.prediction_dataframe['y1_true']
        x2_true = self.prediction_dataframe['x2_true']
        y2_true = self.prediction_dataframe['y2_true']

        Theta1_pred = self.prediction_dataframe['Theta1_pred']
        Theta2_pred = self.prediction_dataframe['Theta2_pred']
        Omega1_pred = self.prediction_dataframe['Omega1_pred']
        Omega2_pred = self.prediction_dataframe['Omega2_pred']
        x1_pred = self.prediction_dataframe['x1_pred']
        y1_pred = self.prediction_dataframe['y1_pred']
        x2_pred = self.prediction_dataframe['x2_pred']
        y2_pred =self.prediction_dataframe['y2_pred']

        t_span = self.validate_timespan

        epochs = self.metricesbyepochs['Epoch']
        TrainLoss = self.metricesbyepochs['TrainLoss']
        TestLoss = self.metricesbyepochs['TestLoss']
        lr = self.metricesbyepochs['Lr']

        Theta1Accuracy = self.metricesbyepochs['Theta1Accuracy']
        Theta2Accuracy = self.metricesbyepochs['Theta2Accuracy']
        Omega1Accuracy = self.metricesbyepochs['Omega1Accuracy']
        Omega2Accuracy = self.metricesbyepochs['Omega2Accuracy']
        
        FigSize = (60,50)
        dotsize = 8
        linewidth = 3

        plt.rc('axes', titlesize=50)        # Controls Axes Title
        plt.rc('axes', labelsize=35)        # Controls Axes Labels
        plt.rc('xtick', labelsize=35)       # Controls x Tick Labels
        plt.rc('ytick', labelsize=35)       # Controls y Tick Labels
        plt.rc('legend', fontsize=35)       # Controls Legend Font

        fig1 = plt.figure(figsize = FigSize)
        fig1ax1 = fig1.add_subplot(2,1,1)
        fig1ax2 = fig1.add_subplot(2,1,2)
        fig1ax1.scatter(t_span, Theta1_pred, color = 'red',s=dotsize, label = 'Pred.')
        fig1ax1.scatter(t_span, Theta1, color = 'blue', s=dotsize, label = 'True')
        fig1ax1.set_title("First Link Angle - " r'$\theta_{1}$', fontweight="bold")
        fig1ax1.set_xlabel('Time (sec)')
        fig1ax1.set_ylabel(r'$\theta_{1}$'u'\xb0' " (deg)")
        fig1ax1.legend(loc = 'upper right')
        fig1ax1.grid(linestyle='--', linewidth=linewidth)
        fig1ax2.scatter(t_span,Theta2_pred, color = 'red', s=dotsize, label = 'Pred.')
        fig1ax2.scatter(t_span, Theta2, color = 'blue', s=dotsize, label = 'True.')
        fig1ax2.set_title("Second Link Angle - "r'$\theta_{2}$', fontweight="bold")
        fig1ax2.set_xlabel('Time (sec)')
        fig1ax2.set_ylabel(r'$\theta_{2}$'u'\xb0'" (deg)")
        fig1ax2.legend(loc = 'upper right')
        fig1ax2.grid(linestyle='--', linewidth=linewidth)
        fig1.savefig(f"./DoublePedulum/Figures/NNFigures/AnglePredictions{self.is_noisy}.png")

        fig2 = plt.figure(figsize = FigSize)
        fig2ax1 = fig2.add_subplot(2,1,1)
        fig2ax2 = fig2.add_subplot(2,1,2)
        fig2ax1.scatter(t_span, Omega1_pred, color = 'red', s=dotsize, label = 'Pred.')
        fig2ax1.scatter(t_span, Omega1, color = 'blue', s=dotsize, label = 'True')
        fig2ax1.set_title("First Link Angular Speed - " r'$\omega_{1}$', fontweight="bold")
        fig2ax1.set_xlabel('Time (sec)')
        fig2ax1.set_ylabel(r'$\omega_{1}$'" (deg/s)")
        fig2ax1.legend(loc = 'upper right')
        fig2ax1.grid(linestyle='--', linewidth=linewidth)
        fig2ax2.scatter(t_span, Omega2_pred, color = 'red', s=dotsize, label = 'Pred.')
        fig2ax2.scatter(t_span, Omega2, color = 'blue', s=dotsize, label = 'True')
        fig2ax2.set_title("Second Link Angular Speed - " r'$\omega_{2}$', fontweight="bold")
        fig2ax2.set_xlabel('Time (sec)')
        fig2ax2.set_ylabel(r'$\omega_{2}$'" (deg/s)")
        fig2ax2.legend(loc = 'upper right')
        fig2ax2.grid(linestyle='--', linewidth=linewidth)
        fig2.savefig(f"./DoublePedulum/Figures/NNFigures/OmegaPredictions{self.is_noisy}.png")

        fig3 = plt.figure(figsize = FigSize)
        fig3ax1 = fig3.add_subplot(2,1,1)
        fig3ax2 = fig3.add_subplot(2,1,2)
        fig3ax1.scatter(t_span, x1_true, color = 'blue', s=dotsize, label = 'True')
        fig3ax1.scatter(t_span, x1_pred, color = 'red', s=dotsize, label = 'Pred.')
        fig3ax1.set_title('First Link Position - X Axis', fontweight="bold")
        fig3ax1.set_xlabel('Time (sec)')
        fig3ax1.set_ylabel('x (m)')
        fig3ax1.grid(linestyle='--', linewidth=linewidth)
        fig3ax1.legend(loc = 'upper right')
        fig3ax2.scatter(t_span, y1_true, color = 'blue', s=dotsize, label = 'True')
        fig3ax2.scatter(t_span, y1_pred, color = 'red', s=dotsize, label = 'Pred.')
        fig3ax2.set_title('First Link Position - Y Axis', fontweight="bold")
        fig3ax2.set_xlabel('Time (sec)')
        fig3ax2.set_ylabel('y (m)')
        fig3ax2.legend(loc = 'upper right')
        fig3ax2.grid(linestyle='--', linewidth=linewidth)
        fig3.savefig(f"./DoublePedulum/Figures/NNFigures/FirstLinkPose{self.is_noisy}.png")

        fig4 = plt.figure(figsize = FigSize)
        fig4ax1 = fig4.add_subplot(2,1,1)
        fig4ax2 = fig4.add_subplot(2,1,2)
        fig4ax1.scatter(t_span, x2_true, color = 'blue', s=dotsize, label = 'True')
        fig4ax1.scatter(t_span, x2_pred, color = 'red', s=dotsize, label = 'Pred.')
        fig4ax1.set_title('Second Link Position - X Axis', fontweight="bold")
        fig4ax1.set_xlabel('Time (sec)')
        fig4ax1.set_ylabel('x (m)')
        fig4ax1.legend(loc = 'upper right')
        fig4ax1.grid(linestyle='--', linewidth=linewidth)
        fig4ax2.scatter(t_span, y2_true, color = 'blue', s=dotsize, label = 'True')
        fig4ax2.scatter(t_span, y2_pred, color = 'red', s=dotsize, label = 'Pred.')
        fig4ax2.set_title('Second Link Position - Y Axis', fontweight="bold")
        fig4ax2.set_xlabel('Time (sec)')
        fig4ax2.set_ylabel('y (m)')
        fig4ax2.legend(loc = 'upper right')
        fig4ax2.grid(linestyle='--', linewidth=linewidth)
        fig4.savefig(f"./DoublePedulum/Figures/NNFigures/SecondLinkPose{self.is_noisy}.png")

        fig5 = plt.figure(figsize = FigSize)
        fig5ax1 = fig5.add_subplot(3,1,1)
        fig5ax2 = fig5.add_subplot(3,1,2)
        fig5ax3 = fig5.add_subplot(3,1,3)
        fig5ax1.plot(epochs, TestLoss, color = 'blue', label = 'Test Loss', linewidth=linewidth)
        fig5ax1.set_title('Test Loss by Epoch', fontweight="bold")
        fig5ax1.set_xlabel('Epoch')
        fig5ax1.set_ylabel('Loss')
        fig5ax1.legend(loc = 'upper right')
        fig5ax1.grid(linestyle='--', linewidth=linewidth)
        fig5ax2.plot(epochs, TrainLoss, color = 'blue', label = 'Train Loss', linewidth=linewidth)
        fig5ax2.set_title('Train Loss by Epoch', fontweight="bold")
        fig5ax2.set_xlabel('Epoch')
        fig5ax2.set_ylabel('Loss')
        fig5ax2.legend(loc = 'upper right')
        fig5ax2.grid(linestyle='--', linewidth=linewidth)
        fig5ax3.plot(epochs, lr, color = 'blue', label = 'Lr', linewidth=linewidth)
        fig5ax3.set_title('Learning Rate by Epoch', fontweight="bold")
        fig5ax3.set_xlabel('Epoch')
        fig5ax3.set_ylabel('Lr')
        fig5ax3.legend(loc = 'upper right')
        fig5ax3.grid(linestyle='--', linewidth=linewidth)
        fig5.savefig(f"./DoublePedulum/Figures/NNFigures/Loss&LearningRate{self.is_noisy}.png")

        fig6 = plt.figure(figsize = FigSize)
        fig6ax1 = fig6.add_subplot(2,1,1)
        fig6ax2 = fig6.add_subplot(2,1,2)
        fig6ax1.plot(epochs, Theta1Accuracy, color = 'blue', label = 'Error. Rate (%)', linewidth=linewidth)
        fig6ax1.set_title('Prediction Error Rate - 'r'$\theta_{1}$'u'\xb0' , fontweight="bold")
        fig6ax1.set_xlabel('Epoch')
        fig6ax1.set_ylabel('Error (%)')
        fig6ax1.legend(loc = 'upper right')
        fig6ax1.grid(linestyle='--', linewidth=linewidth)
        fig6ax2.plot(epochs, Theta2Accuracy, color = 'blue', label = 'Error Rate (%)', linewidth=linewidth)
        fig6ax2.set_title('Prediction Error Rate - 'r'$\theta_{2}$'u'\xb0' , fontweight="bold")
        fig6ax2.set_xlabel('Epoch')
        fig6ax2.set_ylabel('Error (%)')
        fig6ax2.legend(loc = 'upper right')
        fig6ax2.grid(linestyle='--', linewidth=linewidth)
        fig6.savefig(f"./DoublePedulum/Figures/NNFigures/ThetaError{self.is_noisy}.png")

        fig7 = plt.figure(figsize = FigSize)
        fig7ax1 = fig7.add_subplot(2,1,1)
        fig7ax2 = fig7.add_subplot(2,1,2)
        fig7ax1.plot(epochs, Omega1Accuracy, color = 'blue', label = 'Error Rate (%)', linewidth=linewidth)
        fig7ax1.set_title('Prediction Error Rate - 'r'$\omega_{1}$'u'\xb0', fontweight="bold")
        fig7ax1.set_xlabel('Epoch')
        fig7ax1.set_ylabel('Error (%)')
        fig7ax1.legend(loc = 'upper right')
        fig7ax1.grid(linestyle='--', linewidth=linewidth)
        fig7ax2.plot(epochs, Omega2Accuracy, color = 'blue', label = 'Error Rate (%)', linewidth = linewidth)
        fig7ax2.set_title('Error Rate - 'r'$\omega_{2}$'u'\xb0' , fontweight="bold")
        fig7ax2.set_xlabel('Epoch')
        fig7ax2.set_ylabel('Error (%)')
        fig7ax2.legend(loc = 'upper right')
        fig7ax2.grid(linestyle='--', linewidth = linewidth)
        fig7.savefig(f"./DoublePedulum/Figures/NNFigures/OmegaError{self.is_noisy}.png")

if __name__ == "__main__":
    DpNNModel = DoublePendulumnNNModel(RANDOM_SEED = 42)
    DpNNModel.DataGeneration(t_stop = 10, sim_step_size = 100, dt = 0.001, noisy = True, is_traning_data = True)
    DpNNModel.LossFunction(lossfunction = torch.nn.MSELoss())
    DpNNModel.Optimizer(optimizer = torch.optim.Adam, lr = 0.001)
    DpNNModel.Train(epochs = 100)
    DpNNModel.ModelEvaluation(noisy = True, sim_step_size = 1, is_traning_data = False)
    DpNNModel.DataPlots()