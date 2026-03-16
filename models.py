import torch
import torch.nn as nn
import torch.nn.functional as F

# MLP
class MLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MLP, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 400),
            nn.BatchNorm1d(400),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(400, 300),
            nn.BatchNorm1d(300),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(300, 200),
            nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(200, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(100, 50),
            nn.BatchNorm1d(50),
            nn.ReLU(),
            nn.Dropout(0.05),
        )
        self.classifier = nn.Linear(50, num_classes)

    def forward(self, x):
        x = self.encoder(x)
        return self.classifier(x)

# CNN1D
class CNN1D(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=200, kernel_size=4, stride=3)
        self.bn1 = nn.BatchNorm1d(200)
        self.conv2 = nn.Conv1d(in_channels=200, out_channels=200, kernel_size=5, stride=1)
        self.bn2 = nn.BatchNorm1d(200)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(0.05)

        conv_output_len = self._compute_conv_output_length(input_dim)
        self.fc1 = nn.Linear(conv_output_len * 200, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 50)
        self.classifier = nn.Linear(50, num_classes)

    def _compute_conv_output_length(self, input_length):
        x = torch.zeros(1, 1, input_length)
        x = self.pool(self.bn2(self.conv2(self.bn1(self.conv1(x)))))
        return x.shape[2]

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.dropout(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        return self.classifier(x)

# SAE
class SAEClassifier(nn.Module):
    def __init__(self, layer_dims, num_classes):
        super(SAEClassifier, self).__init__()
        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(layer_dims[i], layer_dims[i+1]),
                nn.ReLU(),
                nn.Dropout(0.05)
            ) for i in range(len(layer_dims) - 1)
        ])
        self.classifier = nn.Linear(layer_dims[-1], num_classes)

    def forward(self, x):
        for encoder in self.encoders:
            x = encoder(x)
        return self.classifier(x)

    def encode(self, x):
        for encoder in self.encoders:
            x = encoder(x)
        return x
