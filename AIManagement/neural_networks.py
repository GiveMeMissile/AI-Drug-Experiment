from AIManagement import hyperparameters as hp
from torch import nn
import torch


class LSTM(nn.Module):
    def __init__(self, input_size=hp.INPUT_SIZE, output_size=hp.OUTPUT_SIZE, hidden_size=hp.HIDDEN_SIZE, num_layers=hp.NUM_LAYERS):
        super().__init__()
        
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.out = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x, device, h0=None, c0=None):
        if h0 is None:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        if c0 is None:
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        x, (h0, c0) = self.lstm(x, (h0, c0))
        x = self.out(x)
        return x, h0, c0


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

