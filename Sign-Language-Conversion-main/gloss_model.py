import torch
from torch import nn

class GlossModel(nn.Module):
    def __init__(self, input_size: int, class_no: int, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super().__init__()
        self.device = device

        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.lstm1 = nn.LSTM(input_size, 128, batch_first=True).to(device)
        self.lstm2 = nn.LSTM(128, 64, batch_first=True).to(device)
        self.fc1 = nn.Linear(64, 32).to(device)
        self.fc2 = nn.Linear(32, class_no).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm1_out = self.sigmoid(self.lstm1(x)[0])
        lstm2_out = self.sigmoid(self.lstm2(lstm1_out)[0])
        fc1_out = self.sigmoid(self.fc1(lstm2_out[:,-1,:]))
        out = self.softmax(self.fc2(fc1_out))
        return out
