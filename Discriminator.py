import torch
import torch.nn as nn


class Critic1DCNN(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=64):
        super(Critic1DCNN, self).__init__()
        self.conv1 = (nn.Conv1d(input_dim, hidden_dim, kernel_size=7, padding=3))
        self.conv2 = (nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1))
        self.conv3 = (nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1))
        self.conv4 = (nn.Conv1d(hidden_dim, 1, kernel_size=3, padding=1))
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc = nn.Linear(8, 1)  # Fully connected layer for final output

    def forward(self, c, x):
        c = c.permute(0, 2, 1)
        x = x.permute(0, 2, 1)  # Change shape to [batch_size, input_dim, 256]
        x = self.pool(torch.relu(self.conv1(torch.cat([x, c], dim=1))))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.pool(self.conv4(x))  # Shape [batch_size, 1, 256]
        #x = x.view(x.size(0), -1)  # Flatten to [batch_size, 256]
        #x = self.fc(x)  # Shape [batch_size, 1]
        return x