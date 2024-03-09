import torch
import matplotlib.pyplot as plt
import DoublePendulum as dp
import DoublePendulum_NN as dpNN
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

class DP_SINDyNNComperison:
    def __init__(self, RANDOM_SEED = int, DPNNModel = None, SINDyModel = None):
        self.RANDOM_SEED = RANDOM_SEED
        self.DPNNModel = DPNNModel
        self.SINDyModel = SINDyModel

        self.DP = None
        self.is_noisy = str()
        self.input_data = torch.tensor
        self.output_data = torch.tensor
        self.timespan = torch.tensor
        self.NNModelPredictions = torch.tensor
        self.SINDyPredictions = None

    def NNModelEval(self, split, noisy):
        dpNNTrained = torch.load(self.DPNNModel)

        dpNNTrained.eval()
        with torch.inference_mode():
            dpNNClass = dpNN.DoublePendulumnNNModel(RANDOM_SEED = self.RANDOM_SEED)
            self.is_noisy, self.input_data, self.output_data, self.timespan = dpNNClass.DataGeneration(t_stop = 10, sim_step_size = 1, dt = 0.001, noisy = noisy, split = split)
            self.NNModelPredictions = dpNNTrained(self.input_data)
            Accuracy = dpNNClass.AccuracyFunc(y_true = self.output_data,
                                        y_pred = self.NNModelPredictions,
                                        threshold = 0.01)    

            print(f"Thea 1 Accuracy: %{Accuracy[0]} | Theta 2 Accuracy: %{Accuracy[1]} | Omega 1 Accuracy: %{Accuracy[2]} | Omega 2 Accuracy: %{Accuracy[3]}")

            plt.scatter(self.timespan, self.NNModelPredictions[:,2], color = 'red', s=0.5, label = 'Pred.')
            plt.scatter(self.timespan, self.output_data[:,2], color = 'blue', s=0.5, label = 'True')
            plt.title("TEST")
            plt.xlabel('Time (sec)')
            plt.ylabel("Predictions")
            plt.legend(loc = 'upper right')
            plt.show()

    def SINDyModelEval(self):
        pass

    def DataPlots(self):
        pass

if __name__ == "__main__":
    SINDyNNComperison = DP_SINDyNNComperison(RANDOM_SEED = 50, DPNNModel = './DoublePedulum/TrainedModels/DPNNModel(Noise = Normal).pth')
    SINDyNNComperison.NNModelEval(noisy = False, split = False)