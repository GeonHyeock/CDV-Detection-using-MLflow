import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
from PIL import Image
from sklearn.cluster import DBSCAN
from scipy.sparse import csr_array


class FilterConv(nn.Module):
    def __init__(self, in_channels=3):
        super(FilterConv, self).__init__()
        self.filter_conv = self.gh_conv2d(in_channels=in_channels)
        self.filter_conv2 = self.gh_conv2d(in_channels=1)

    def gh_conv2d(
        self,
        in_channels=1,
        out_channels=1,
        kernel_size=3,
        stride=1,
        padding=1,
        threshold=-11,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        for n, p in conv.named_parameters():
            if "weight" == n:
                p = torch.zeros(p.size()) + 1
                p[:, :, 1, 1] = 10
                conv.weight.data = p
            elif "bias" == n:
                p = torch.zeros(p.size()) + threshold
                conv.bias.data = p
        return conv

    def forward(self, x, layer=5):
        x = self.filter_conv(x)
        x = torch.where(x > 0, 1., 0.)

        for _ in range(layer - 1):
            x = self.filter_conv2(x)
            x = torch.where(x > 0, 1., 0.)
        return x


class CNNReLUBlock(nn.Module):
    def __init__(self):
        super(CNNReLUBlock, self).__init__()
        self.net = FilterConv()

    def forward(self, x):
        x = T.ToTensor()(x).to("cuda")
        x = self.net(x) 
        x = [(int(i), int(j)) for i, j in zip(*(x > 0).nonzero(as_tuple=True)[1:])]
        clustering = DBSCAN(eps=3, min_samples=2).fit(x)

        row = np.array([i[0] for i in x])
        col = np.array([i[1] for i in x])
        data = clustering.labels_

        x = csr_array((data, (row, col)), shape=(2048, 2048)).toarray()
        x = np.where(x < 0, 0, x)
        x = np.repeat(np.expand_dims(x, axis=0), 3, axis=0).astype(np.uint8)
        return Image.fromarray(x.swapaxes(0, 2))


if __name__ == "__main__":
    model = CNNReLUBlock()
