import torch
import torch.nn as torch



class NeuralNetwork(torch.Module):
    def __init__(self):
        super().__init__()

        self.layers = torch.Sequential(
            torch.Linear(65, 128),
            torch.ReLU(),
            torch.Linear(128, 64),
            torch.ReLU(),
            torch.Linear(64, 1)
        )

    def forward(self, x):
        return self.layers(x)


