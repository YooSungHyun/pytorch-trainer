import torch.nn as nn


class Net(nn.Module):
    # # 기본변수, layer를 초기화해주는 생성자
    def __init__(self, input_dim, hidden_dim, seq_len, output_dim, layers, dropout=0.0):
        super(Net, self).__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.output_dim = output_dim
        self.layers = layers

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=layers,
            dropout=(dropout if layers > 1 else 0.0),
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, output_dim, bias=True)

    # 예측을 위한 함수
    def forward(self, x):
        x, _status = self.lstm(x)
        x = self.fc(x[:, -1])
        return x
