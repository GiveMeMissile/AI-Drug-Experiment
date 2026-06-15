from AIManagement import hyperparameters as hp
from torch import nn
import torch


class LSTM(nn.Module):
    pass


class SimpleNN(nn.Module):
    def __init__(self, input_size=hp.INPUT_SIZE, output_size=hp.OUTPUT_SIZE, hidden_size=hp.HIDDEN_SIZE, num_layers=hp.NUM_LAYERS):
        super().__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.num_layers = num_layers
        self.hidden_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(in_features=hidden_size, out_features=hidden_size),
                    nn.ReLU(),
                    nn.Linear(in_features=hidden_size, out_features=hidden_size)
                ) for _ in range(num_layers)
            ]
        )
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.input_layer(x)
        for i in range(self.num_layers):
            x = self.hidden_layers[i](x)
        return self.output_layer(x)

