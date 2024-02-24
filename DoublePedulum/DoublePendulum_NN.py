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
        self.dataset_dataframe = pd.DataFrame()
        self.train_data = pd.DataFrame()
        self.test_data = pd.DataFrame()

        # 2. Create 2 nn.Linear layers capable of handling X and y input and output shapes
        self.layer_stack = torch.nn.Sequential(torch.nn.Linear(in_features=4, out_features=64),
                                                torch.nn.ReLU(),
                                                torch.nn.Linear(in_features=64, out_features=32),
                                                torch.nn.ReLU(), 
                                                torch.nn.Linear(in_features=32, out_features=16),
                                                torch.nn.ReLU(),
                                                torch.nn.Linear(in_features=16, out_features=4),
                                                torch.nn.ReLU())
        
    def DpDataGeneration(self, t_stop = 10, dt = 0.001, sim_step_size=2, 
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

        self.dataset_dataframe.to_csv('DoublePendulumDataset.csv', encoding='utf-8')

        print(self.dataset_dataframe)
        input_data = np.array(self.dataset_dataframe[['Theta1_dot', 'Theta2_dot', 'Omega1_dot', 'Omega2_dot']])
        output_data = np.array(self.dataset_dataframe[['Theta1', 'Theta2', 'Omega1', 'Omega2']])

        input_data = torch.from_numpy(input_data).type(torch.float)
        output_data = torch.from_numpy(output_data).type(torch.float)


        input_train, input_test, output_train, output_test = train_test_split(input_data,
                                                                              output_data,
                                                                              test_size=0.2,
                                                                              random_state = np.random.seed(self.RANDOM_SEED))
    def DataPlots(self):
        pass

    def forward(self, x):
        return self.layer_stack(x)

DpNNModel = DoublePendulumnNNModel()
DpNNModel.DpDataGeneration()

    