import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.lstm1 = nn.LSTM(7, 32, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(32, 16, num_layers=1, batch_first=True)
        self.fc = nn.Linear(16, 1, bias=True)

    def forward(self, inputs):
        hidden, _ = self.lstm1(inputs)
        hidden, _ = self.lstm2(hidden)
        hidden = hidden[:, -1, :]
        logit = self.fc(hidden)
        return logit
