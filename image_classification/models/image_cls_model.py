from torch import nn


class ImageClassificationModel(nn.Module):
    def __init__(
        self,
        num_labels: int,
    ):
        super().__init__()
        # input shape(256, 256, 3)
        # conv shape(11x11, stride 5) -> conv output(50, 50, 64)
        # max pool shape(4x4, 3, stride 2) -> max pool output(24, 24, 64)
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=5),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=2),
        )
        # input shape(24, 24, 64)
        # conv shape(5x5, stride 1) -> conv output(20, 20, 256)
        # max pool shape(4x4, stride 2) -> max pool output(9, 9, 256)
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=5, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=2),
        )

        self.fc_layer = nn.Linear(9 * 9 * 256, num_labels)

    def forward(self, x):
        try:
            output = self.conv_layer1(x)
            output = self.conv_layer2(output)
            output = nn.Flatten()(output)
            output = self.fc_layer(output)
        except Exception as e:
            print(f"error : {e}")

        return output


if __name__ == "__main__":
    import torch

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model = ImageClassificationModel(num_labels=10)
    model.to(device)

    dummy_data = torch.rand([1, 3, 256, 256])
    dummy_data = dummy_data.to(device)

    output = model(dummy_data)
    print(output)
