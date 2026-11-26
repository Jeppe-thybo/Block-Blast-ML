import torch
import torch.nn as nn


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()


        self.layers = nn.Sequential(
            nn.Linear(65, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.layers(x)



class BoardCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.head = nn.Sequential(
            nn.Linear(64 * 8 * 8 + 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, board, extras):
        x = self.conv(board)
        x = x.flatten(1)                         
        x = torch.cat([x, extras], dim=1)       
        return self.head(x)