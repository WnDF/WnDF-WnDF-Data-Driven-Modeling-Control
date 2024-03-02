import torch
import matplotlib.pyplot as plt
import DoublePendulum as dp
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class DoublePendulumnNNModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.RANDOM_SEED = 42
        self.optimizer = None
        self.lossfunction = None
        self.lr = None

        self.dataset_dataframe = pd.DataFrame()
        self.input_train_data = 0
        self.input_test_data = 0
        self.output_train_data = 0
        self.output_test_data = 0
        self.train_timespan_data = 0
        self.test_timespan_data = 0
        self.prediction_dataframe = pd.DataFrame()
        self.metricesbyepochs = pd.DataFrame()


        # 2. Create 2 nn.Linear layers capable of handling X and y input and output shapes
        self.model = torch.nn.Sequential(torch.nn.Linear(in_features=4, out_features=1024),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(in_features=1024, out_features=512),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(in_features=512, out_features=256),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(in_features=256, out_features=128),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(in_features=128, out_features=64),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(in_features=64, out_features=16),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(in_features=16, out_features=4))
        
    def DataGeneration(self, t_stop = 10, dt = 0.001, sim_step_size=2, 
                         m1 = 0.2704, m2 = 0.2056, cg1 = 0.191, cg2 = 0.1621, 
                         L1 = 0.2667, L2 = 0.2667, I1 = 0.003, I2 = 0.0011, g = 9.81):
        
        # create double dendulumn state-space model
        double_pendulum = dp.DoublePendulumSS(m1, m2, cg1, cg2, L1, L2, I1, I2, g)
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

            df = double_pendulum.simulate(state, t_span, t_eval)
            self.dataset_dataframe = pd.concat([self.dataset_dataframe, df], axis = 0)
        
        self.SaveSimulationData()

        input_data = np.array(self.dataset_dataframe[['Theta1_dot', 'Theta2_dot', 'Omega1_dot', 'Omega2_dot']])
        output_data = np.array(self.dataset_dataframe[['Theta1', 'Theta2', 'Omega1', 'Omega2']])
        timespan = np.array(self.dataset_dataframe[['Time']])

        input_data = torch.from_numpy(input_data).type(torch.float)
        output_data = torch.from_numpy(output_data).type(torch.float)
        timespan = torch.from_numpy(timespan).type(torch.float)

        self.input_train_data, self.input_test_data, self.output_train_data, self.output_test_data, self.train_timespan_data, self.test_timespan_data = train_test_split(input_data, 
                                                                                                                                                                        output_data, 
                                                                                                                                                                        timespan,
                                                                                                                                                                        test_size=0.2,
                                                                                                                                                                        random_state = np.random.seed(self.RANDOM_SEED))
    
    def SaveSimulationData(self):
        self.dataset_dataframe.to_csv('DoublePendulumDataset.csv', encoding='utf-8')
    
    def LossFunction(self, lossfunction = torch.nn.MSELoss()):
        self.lossfunction = lossfunction

    def Optimizer(self, optimizer = torch.optim.Adam, lr = 0.001):
        self.optimizer = optimizer(self.model.parameters(), lr = lr)
        self.lr = lr

    def AccuracyFunc(self, y_true, y_pred, treshold = 0.01):
        acc_rates = np.empty(1,4)
        for i in range(0,len(y_true)):
            error = torch.abs(y_true[i] - y_pred[i])
            correct = torch.sum(error <= treshold).item()
            acc = (correct / len(y_true)) * 100
            acc_rates.append(acc)
        return acc_rates

    def Train(self, epochs = 100):
        np.random.seed(self.RANDOM_SEED)

        for epoch in range(epochs):
            self.model.train()

            y_preds = self.model(self.input_train_data)
            loss = self.lossfunction(y_preds, self.output_train_data)
            accuracy = self.AccuracyFunc(y_true = self.output_train_data, 
                                         y_pred = y_preds)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.model.eval()

            with torch.inference_mode():
                test_preds = self.model(self.input_test_data)
                test_loss = self.lossfunction(test_preds, self.output_test_data)
                test_acc = self.AccuracyFunc(y_true = self.output_test_data,
                                            y_pred = test_preds,
                                            treshold = 0.01)
                
                dfmetrices = pd.DataFrame({'Epoch': epoch, 'Lr': self.optimizer.param_groups['lr'],
                                           'Theta1Accuracy': test_acc[0], 'Theta2Accuracy': test_acc[1],
                                           'Omega1Accuracy': test_acc[2], 'Omega2Accuracy': test_acc[3],
                                           'TrainLoss': loss, 'TestLoss': test_loss})
                self.metricesbyepochs = pd.concat([self.metricesbyepochs, dfmetrices], axis = 0)
            
            if epoch % 100 == 0:
                print(f"Epoch: {epoch} | Loss: {loss:.5f} | Theta1Acc: {test_acc[0]:.2f}% | Theta2Acc: {test_acc[1]:.2f} | Test Loss: {test_loss:.5f}")

    def DataPlots(self):
        self.model.eval()

        with torch.inference_mode():
            predictions = self.model(self.input_test_data).numpy()
            self.output_test_data = self.output_test_data.numpy()
            Theta1 = self.output_test_data[:,0]
            Theta1_pred = predictions[:,0]
            Theta2 = self.output_test_data[:,1]
            Theta2_pred = predictions[:,1]
            Omega1= self.output_test_data[:,2]
            Omega1_pred = predictions[:,2]
            Omega2 = self.output_test_data[:,3]
            Omega2_pred = predictions[:,3]
            t_span = self.test_timespan_data

            epochs = self.metricesbyepochs['Epoch']
            TrainLoss = self.metricesbyepochs['TrainLoss']
            TestLoss = self.metricesbyepochs['TestLoss']
            lr = self.metricesbyepochs['lr']
            
            Theta1Accuracy = self.metricesbyepochs['Theta1Accuracy']
            Theta2Accuracy = self.metricesbyepochs['Theta2Accuracy']
            Omega1Accuracy = self.metricesbyepochs['Omega1Accuracy']
            Omega2Accuracy = self.metricesbyepochs['Omega2Accuracy']
            
            plt.subplot(2, 2, 1)
            plt.scatter(t_span, Theta1_pred, s=0.3)
            plt.scatter(t_span, Theta1, s=0.3)
            plt.title('Theta1')
            plt.xlabel('Time')
            plt.ylabel('Theta1-Theta1_pred')
            plt.grid(True)

            plt.subplot(2, 2, 2)
            plt.scatter(t_span,Theta2_pred, s=0.3)
            plt.scatter(t_span, Theta2, s=0.3)
            plt.title('Theta2')
            plt.xlabel('Time')
            plt.ylabel('Theta2-Theta2_pred')
            plt.grid(True)

            plt.subplot(2, 2, 3)
            plt.scatter(t_span,Omega1_pred, s=0.3)
            plt.scatter(t_span, Omega1, s=0.3)
            plt.title('Omega1')
            plt.xlabel('Time')
            plt.ylabel('Omega1-Omega1_pred')
            plt.grid(True)

            plt.subplot(2, 2, 4)
            plt.scatter(t_span, Omega2_pred, s=0.3)
            plt.scatter(t_span, Omega2, s=0.3)
            plt.title('Omega2')
            plt.xlabel('Time')
            plt.ylabel('Omega2-Omega2_pred')
            plt.grid(True)

            plt.show()

            plt.subplot(1, 3, 1)
            plt.scatter(epochs, TestLoss, s = 0.3)
            plt.title('Test Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid(True)

            plt.subplot(1, 3, 2)
            plt.scatter(epochs, TrainLoss, s = 0.3)
            plt.title('Train Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid(True)

            plt.subplot(1, 3, 3)
            plt.scatter(epochs, lr, s = 0.3)
            plt.title('Learning Rate')
            plt.xlabel('Epoch')
            plt.ylabel('Lr')
            plt.grid(True)

            plt.show()

            plt.subplot(2, 2, 1)
            plt.scatter(epochs, Theta1Accuracy, s = 0.3)
            plt.title('Theta1 Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.grid(True)

            plt.subplot(2, 2, 2)
            plt.scatter(epochs, Theta2Accuracy, s = 0.3)
            plt.title('Theta2 Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.grid(True)

            plt.subplot(2, 2, 3)
            plt.scatter(epochs, Omega1Accuracy, s = 0.3)
            plt.title('Omega1 Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.grid(True)

            plt.subplot(2, 2, 4)
            plt.scatter(epochs, Omega2Accuracy, s = 0.3)
            plt.title('Omega2 Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.grid(True)

            plt.show()

    def forward(self, x):
        return self.model(x)

DpNNModel = DoublePendulumnNNModel()
DpNNModel.DataGeneration(t_stop = 10, sim_step_size = 1)
DpNNModel.LossFunction()
DpNNModel.Optimizer(optimizer = torch.optim.Adam, lr = 0.001)
DpNNModel.Train(epochs = 10000)