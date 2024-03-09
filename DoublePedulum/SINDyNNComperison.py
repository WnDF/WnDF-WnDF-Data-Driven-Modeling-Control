import torch
import matplotlib.pyplot as plt
import DoublePendulum as dp
import numpy as np
import pandas as pd
import pickle

class DP_SINDyNNComperison:
    def __init__(self, DPNNModel = None, SINDyModel = None):
        self.RANDOM_SEED = 45
        self.DPNNModel = DPNNModel
        self.SINDyModel = SINDyModel

        self.DP = ''
        self.is_noisy = None
        self.dataset_dataframe = pd.DataFrame()
        self.input_data = 0
        self.output_data = 0
        self.timespan = 0
        self.NNModelPredictions = pd.DataFrame()
        self.SINDyPredictions = pd.DataFrame()
    
    def ValidationDataGeneration(self, Noisy, t_stop = 10, dt = 0.001, sim_step_size=2, 
                         m1 = 0.2704, m2 = 0.2056, cg1 = 0.191, cg2 = 0.1621, 
                         L1 = 0.2667, L2 = 0.2667, I1 = 0.003, I2 = 0.0011, g = 9.81):
        
        if Noisy:
            self.is_noisy = ' (Noise = Noisy)'
        else:
            self.is_noisy = ' (Noise = Normal)'
        
        # create double dendulumn state-space model
        self.DP = dp.DoublePendulumSS(m1, m2, cg1, cg2, L1, L2, I1, I2, g)
        np.random.seed(self.RANDOM_SEED)

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
            self.dataset_dataframe = pd.concat([self.dataset_dataframe, df], axis = 0)
        
        self.SaveValidationDataset(data = self.dataset_dataframe, PATH = f'./DoublePedulum/Dataset/DoublePendulumValidationDataset{self.is_noisy}.csv')

        input_data = np.array(self.dataset_dataframe[['Theta1_dot', 'Theta2_dot', 'Omega1_dot', 'Omega2_dot']])
        output_data = np.array(self.dataset_dataframe[['Theta1', 'Theta2', 'Omega1', 'Omega2']])
        timespan = np.array(self.dataset_dataframe[['Time']])

        self.input_data = torch.from_numpy(input_data).type(torch.float)
        self.output_data = torch.from_numpy(output_data).type(torch.float)
        self.timespan = torch.from_numpy(timespan).type(torch.float)
    
    def SaveValidationDataset(self, data, PATH):
        data.to_csv(PATH, encoding='utf-8')

    def AccuracyFunc(self, y_true, y_pred, threshold=0.01):
        acc_rates = []

        # Iterate over each column
        for i in range(y_true.shape[1]):
            true_col = y_true[:, i]
            pred_col = y_pred[:, i]

            # Calculate absolute error
            abs_error = (torch.abs(true_col - pred_col))/true_col

            # Calculate accuracy for this column
            correct = torch.sum(abs_error <= threshold).item()
            acc = (correct / len(true_col)) * 100
            acc_rates.append(acc)

        return acc_rates
    
    def NNModelEval(self):
        dpnnmodel = torch.load(self.DPNNModel)

        dpnnmodel.eval()
        with torch.inference_mode():
            predictions = dpnnmodel(self.input_data)
            Accuracy = self.AccuracyFunc(y_true = self.output_data,
                                            y_pred = predictions)    
            fig = plt.figure()
            figax1 = fig.add_subplot(2,1,1)
            figax1.scatter(self.timespan, predictions[:,1], color = 'red', s=5, label = 'Pred.')
            figax1.scatter(self.timespan, self.output_data[:,1], color = 'blue', s=5, label = 'True')
            figax1.set_title("TEST")
            figax1.set_xlabel('Time (sec)')
            figax1.set_ylabel("Predictions")
            figax1.legend(loc = 'upper right')
            plt.show()

            print(f"Accuracy: {Accuracy}%")

    def SINDyModelEval(self):
        pass

    def DataPlots(self):
        pass

SINDyNNComperison = DP_SINDyNNComperison(DPNNModel = './DoublePedulum/TrainedModels/DPNNModel(Noise = Normal).pth')
SINDyNNComperison.ValidationDataGeneration(t_stop = 10, sim_step_size = 1, dt = 0.001, Noisy = False)
SINDyNNComperison.NNModelEval()