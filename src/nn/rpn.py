import torch
from torch.nn import Conv2d, Module, ReLU


def apply_transform(logits: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    """Transform logits to generate either proposals
    or detections.

    Parameters
    ----------
    logits : torch.Tensor
        Tensor of shape [batch_size, num_proposals, num_classes, 4].
    anchors : torch.Tensor
        Tensor of shape [num_proposals, 4].

    Returns
    -------
    torch.Tensor
        Transformed logits to the boxes of shape
        [batch_size, num_proposals, num_classes, 4].
    """
    anchors_w = anchors[..., 2] - anchors[..., 0]
    anchors_h = anchors[..., 3] - anchors[..., 1]
    anchors_center_x = (anchors[..., 2] + anchors[..., 0]) / 2
    anchors_center_y = (anchors[..., 3] + anchors[..., 1]) / 2

    dx = logits[..., 0]
    dy = logits[..., 1]
    dw = logits[..., 2]
    dh = logits[..., 3]

    predicted_center_x = dx * anchors_w + anchors_center_x
    predicted_center_y = dy * anchors_h + anchors_center_y
    predicted_width = anchors_w * torch.exp(dw)
    predicted_height = anchors_h * torch.exp(dh)

    return torch.stack(
        [predicted_center_x, predicted_center_y, predicted_width, predicted_height],
        dim=3,
    )


class RegionProposalNetwork(Module):
    """Network for extracting regions for further classification.

    Parameters
    ----------
    scales : list[int]
        Scales to use for detection of objects, e.g. 128, 256, 512.
    aspect_rations : list[float]
        Aspect ratios of achor boxes to track for.
    in_channels : int
        Channels to recieve from the feature extractor layer. By default, it is 512.
    """

    def __init__(
        self, scales: list[int], aspect_ratios: list[float], in_channels: int = 512
    ) -> None:
        super(RegionProposalNetwork, self).__init__()

        self.scales = scales
        self.aspect_ratios = aspect_ratios
        self.nanchors = len(self.scales) * len(self.aspect_ratios)

        self.conv_1 = Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )

        self.relu_1 = ReLU()

        self.cls_conv_1x1 = Conv2d(
            in_channels=in_channels,
            out_channels=self.nanchors,
            kernel_size=(1, 1),
            stride=(1, 1),
        )

        self.bbox_conv_1x1 = Conv2d(
            in_channels=in_channels,
            out_channels=self.nanchors * 4,
            kernel_size=(1, 1),
            stride=(1, 1),
        )

    def generate_anchors(
        self, images: torch.Tensor, features: torch.Tensor
    ) -> torch.Tensor:
        """Generate anchor boxes.

        Parameters
        ----------
        images : torch.Tensor
            Images to generate anchor boxes for.
        features : torhc.Tensor
            Features recieved from the feature extraction layer.

        Returns
        -------
        torch.Tensor
            Tensor of shape [n_anchors * stride_w * stride_h, 1, 4], where
            each anchor box represented as [xmin, ymin, xmax, ymax].
        """
        grid_h, grid_w = features.shape[-2:]
        image_h, image_w = images.shape[-2:]

        stride_h = torch.tensor(
            image_h // grid_h, dtype=torch.int64, device=features.device
        )
        stride_w = torch.tensor(
            image_w // grid_w, dtype=torch.int64, device=features.device
        )

        scales = torch.as_tensor([128, 256, 512])
        aspect_ratios = torch.as_tensor([0.33, 1.0, 1.33])

        h_ratios = torch.sqrt(aspect_ratios)
        w_ratios = 1 / h_ratios

        height_sizes = h_ratios[..., None] * scales
        width_sizes = w_ratios[..., None] * scales

        base_anchors = (
            torch.stack([-width_sizes, -height_sizes, width_sizes, height_sizes], dim=2)
            / 2
        ).round()

        shifts_x = (
            torch.arange(0, grid_w, dtype=torch.int64, device=features.device)
            * stride_w
        )
        shifts_y = (
            torch.arange(0, grid_h, dtype=torch.int64, device=features.device)
            * stride_h
        )
        shifts_y, shifts_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")

        shifts = torch.stack([shifts_x, shifts_y, shifts_x, shifts_y], dim=2)

        anchors = shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)
        return anchors.view(-1, 1, 4)

    def forward(
        self, images: torch.Tensor, features: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = images.shape[0]

        rpn_features = self.relu_1(self.conv_1(features))

        cls_scores: torch.Tensor = self.cls_conv_1x1(rpn_features)
        bbox_proposals: torch.Tensor = self.bbox_conv_1x1(rpn_features)

        anchors = self.generate_anchors(images, features)

        cls_scores = cls_scores.permute(0, 2, 3, 1).reshape(batch_size, -1, 1)
        bbox_proposals = (
            bbox_proposals.view(batch_size, 9, 4, 54, 54)
            .permute(0, 3, 4, 1, 2)
            .reshape(batch_size, -1, 1, 4)
        )

        return cls_scores, apply_transform(bbox_proposals, anchors).squeeze(2)
