import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
import LorenzSystem as LS
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
        self.out_validate_data = torch.tensor

        self.train_timespan_data = torch.tensor
        self.test_timespan_data = torch.tensor
        self.validate_timespan = torch.tensor

        self.model = torch.nn.Sequential(torch.nn.Linear(in_features=3, out_features=128),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(in_features=128, out_features=64),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(in_features=64, out_features=16),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(in_features=16, out_features=3))

    def DataGeneratinon(self, noisy = bool(), is_traning_data = bool(), sigma=10, rho=28, beta=8/3,
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

    def AccuracyFunc(self, y_true = torch.tensor, y_pred = torch.tensor, threshold = float()):
        acc_rates = []

        # Iterate over each column
        for i in range(y_true.shape[1]):
            true_col = y_true[:, i]
            pred_col = y_pred[:, i]

            abs_error = (torch.abs(true_col - pred_col)/true_col).detach().numpy()
            correct = np.count_nonzero(abs_error <= threshold)
            acc = (correct / len(true_col)) * 100
            acc_rates.append(acc)

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
                                            y_pred = test_preds,
                                            threshold = 0.01)
                
                dfmetrices = pd.DataFrame({'Epoch': epoch, 'Lr': [self.optimizer.param_groups[0]['lr']],
                                           'x_Accuracy': test_acc[0], 'y_Accuracy': test_acc[1], 'z_Accuracy': test_acc[2], 
                                           'TrainLoss': loss, 'TestLoss': test_loss})
                self.metricesbyepochs = pd.concat([self.metricesbyepochs, dfmetrices], axis = 0)
            
            if epoch % 10 == 0:
                print(f"Epoch: {epoch} | Loss: {loss:.5f} | Test Loss: {test_loss:.5f} | x_Acc: {test_acc[0]:.2f}% | y_Acc: {test_acc[1]:.2f}% | z_Acc: {test_acc[2]:.2f}%")

    def ModelEvaluation(self):
        pass
    
    def forward(self, x):
        return self.model(x)
    
    def SaveModel(self, PATH = str()):
        torch.save(self.model, PATH)

    def DataPlots(self):
        pass

if __name__ == "__main__":
    LorenzModel = LorenzSystemNNModel(RANDOM_SEED = 52)
    LorenzModel.DataGeneratinon(t_stop = 10, sim_step_size = 100, dt = 0.01, noisy = False, is_traning_data = True)
    LorenzModel.LossFunction(lossfunction = torch.nn.MSELoss())
    LorenzModel.Optimizer(optimizer = torch.optim.Adam, lr = 0.001)
    LorenzModel.Train(epochs = 2000)