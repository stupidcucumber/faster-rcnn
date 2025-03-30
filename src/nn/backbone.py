import torch
from torch.nn import BatchNorm2d, Conv2d, MaxPool2d, Module, ReLU


class ResNetBlock(Module):
    """PyTorch implementation of a ResNet Borrleneck Block.

    Parameters
    ----------
    in_channels : int
        Expected number of channels in the input.
    out_channels : int
        Expected number of channels on the output.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(ResNetBlock, self).__init__()
        self.conv_1 = Conv2d(
            in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding="same"
        )
        self.bn_1 = BatchNorm2d(out_channels)
        self.relu_1 = ReLU()
        self.conv_2 = Conv2d(
            out_channels,
            out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding="same",
        )
        self.bn_2 = BatchNorm2d(out_channels)
        self.conv_1x1 = Conv2d(
            in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1)
        )
        self.relu_2 = ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform forward propagation of the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to perform propagation on.

        Returns
        -------
        torch.Tensor
            Output results.
        """
        residual_features = self.conv_1x1(x)
        x = self.bn_1(self.conv_1(x))
        x = self.relu_1(x)
        x = self.bn_2(self.conv_2(x))
        return self.relu_2(residual_features + x)


class ResNetBlockStack(Module):
    """Helper class to incapsulate an arbitrary number of Bottleneck blocks.

    Parameters
    ----------
    in_channels : int
        Expected number of channels in the input.
    out_channels : int
        Expected number of channels on the output.
    nblocks : int
        Number of Bottleneck blocks inside.
    """

    def __init__(self, in_channels: int, out_channels: int, nblocks: int) -> None:
        super(ResNetBlockStack, self).__init__()

        self.seq = torch.nn.Sequential(
            *[
                ResNetBlock(in_channels if index == 0 else out_channels, out_channels)
                for index in range(nblocks)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform forward propagation of the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to perform propagation on.

        Returns
        -------
        torch.Tensor
            Output results.
        """
        return self.seq(x)


class FasterRCNNBackbone(Module):
    """PyTorch implementation of ResNet-50 backbone."""

    def __init__(self) -> None:
        super(FasterRCNNBackbone, self).__init__()

        self.conv_0 = Conv2d(3, 64, (7, 7), (2, 2))
        self.max_pool_0 = MaxPool2d((3, 3), (2, 2))

        self.seq = torch.nn.Sequential(
            ResNetBlockStack(64, 128, 3),
            ResNetBlockStack(128, 256, 4),
            ResNetBlockStack(256, 512, 5),
            ResNetBlockStack(512, 1024, 6),
            ResNetBlockStack(1024, 2048, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform forward propagation of the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to perform propagation on.

        Returns
        -------
        torch.Tensor
            Output results.
        """
        x = self.max_pool_0(self.conv_0(x))
        return self.seq(x)
