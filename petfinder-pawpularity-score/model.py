from torch import nn


class CNN(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels):
        super(CNN, self).__init__()

        self.layer1 = nn.Sequential(nn.Conv2d(in_channels, hidden_channels, kernel_size=(4, 4), stride=(1, 1)),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=(4, 4)))

        self.layer2 = nn.Sequential(nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=(4, 4), stride=(1, 1)),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=(4, 4)))

        self.flatten = nn.Flatten()

        self.last = nn.Sequential(nn.Linear(hidden_channels * 2 * 15 * 15, hidden_channels * 2),
                                  nn.Linear(hidden_channels * 2, hidden_channels),
                                  nn.Linear(hidden_channels, out_channels))

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.flatten(out)
        out = self.last(out)
        return out
