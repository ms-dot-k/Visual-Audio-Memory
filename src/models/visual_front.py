from torch import nn
from src.models.resnet import BasicBlock, ResNet

class Visual_front(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()

        self.in_channels = in_channels

        self.frontend = nn.Sequential(
            nn.Conv3d(self.in_channels, 64, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.PReLU(64),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )

        self.resnet = ResNet(BasicBlock, [2, 2, 2, 2], relu_type='prelu')

        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        #B,C,S,H,W
        x = self.frontend(x)    #B,C,T,H,W
        B, C, T, H, W = x.size()
        x = x.transpose(1, 2).contiguous().view(B*T, C, H, W)
        x = self.resnet(x)
        x = self.dropout(x)
        x = x.view(B, T, -1)
        return x
