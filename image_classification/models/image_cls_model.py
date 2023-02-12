from torch import nn


class ImageClassificationModel(nn.Module):
    def __init__(
        self,
        num_labels: int,
    ):
        super(ImageClassificationModel).__init__()
        # input shape(256, 256, 3)
        # conv shape(11x11, stride 5)
        # conv output(50, 50, 64)
        # max pool shape(4x4, 3, stride 2)
        # max pool output(24, 24, 64)
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=3),
            nn.BatchNorm2d(),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=2),
        )
        # input shape(24, 24, 64)
        # conv shape(5x5, stride 1)
        # conv output(20, 20, 256)
        # max pool shape(4x4, stride 2)
        # max pool output(9, 9, 256)
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, stride=1),
            nn.BatchNorm2d(),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=2),
        )

        self.fc_layer = nn.Linear(9 * 9 * 256, num_labels)

    def forward(self, x):
        output = self.conv_layer1(x)
        output = self.conv_layer2(output)
        output = nn.Flatten()(output)
        output = self.fc_layer(output)

        return output
