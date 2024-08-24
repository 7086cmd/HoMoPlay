import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from src.config import device


class GestureClassifier(nn.Module):
    def __init__(self):
        super(GestureClassifier, self).__init__()
        # Define convolution and pooling layers
        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, padding=1, stride=1)
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=1)
        # Fully connected layers
        self.fc1 = nn.Linear(40, 64)  # Adjusted to match the output shape after conv and pooling
        self.fc2 = nn.Linear(64, 4)
        self.output = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.7)

    def forward(self, x):
        batch_size = x.size(0)
        slices = []

        # Loop over the 5 slices in the input
        for i in range(5):
            slice = x[:, :, i, :].unsqueeze(1)  # Extract the i-th slice and add channel dimension
            conv_out = self.conv(slice)
            pool_out = self.pooling(conv_out)
            slices.append(pool_out)

        # Stack the slices and flatten
        x = torch.cat(slices, dim=1)  # Concatenate slices along the channel dimension
        x = x.view(batch_size, -1)  # Flatten to (batch_size, 5 * 3 * 3)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.output(x)
        return x


model = GestureClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 10
