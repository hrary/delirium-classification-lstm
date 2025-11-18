import torch
from torch import nn

class ClassificationAlgorithm(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim=1):
        # input dim: number of features in the input data (per packet, not per sequence)
        # hidden dim: number of features in the hidden state of the LSTM
        # num layers: number of stacked LSTM layers

        super(ClassificationAlgorithm, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x has shape (batch_size, seq_length, input_dim)
        # batch: number of sequences processed together
        # batch size: number of training samples in a batch (eg if i have 32 example sequences, batch size is 32)
        h_0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c_0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)

        lstm_out, _ = self.lstm(x, (h_0, c_0))
        last_out = self.fc(lstm_out[:, -1, :])
        return last_out