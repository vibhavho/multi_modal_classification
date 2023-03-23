import torch.nn as nn
import torch.nn.functional as F


class ThreeDCNN(nn.Module):
    def __init__(self):
        super(ThreeDCNN, self).__init__()

        self.base = 64

        self.encoder = nn.Sequential(
            nn.Conv3d(3, self.base, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm3d(self.base),
            nn.LeakyReLU(0.1, inplace = True),
            nn.Dropout(0.3),
            nn.MaxPool3d(2, 2),

            nn.Conv3d(self.base, self.base * 2, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm3d(self.base * 2),
            nn.LeakyReLU(0.1, inplace = True),
            nn.Dropout(0.3),
            nn.MaxPool3d(2, 2),

            nn.Conv3d(self.base * 2, self.base * 4, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm3d(self.base * 4),
            nn.LeakyReLU(0.1, inplace = True),
            nn.Dropout(0.3),
            nn.MaxPool3d(2, 2),

            nn.Conv3d(self.base * 4, self.base * 8, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm3d(self.base * 8),
            nn.LeakyReLU(0.1, inplace = True)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(self.base * 8, self.base * 4, kernel_size = 3, stride = 2, padding = 1, output_padding = 1),
            nn.BatchNorm3d(self.base * 4),
            nn.LeakyReLU(0.1, inplace = True),
            nn.Dropout(0.3),
            nn.Upsample(size = (3, 8, 8)),

            nn.ConvTranspose3d(self.base * 4, self.base * 2, kernel_size = 3, stride = 2, padding = 1, output_padding = 1),
            nn.BatchNorm3d(self.base * 2),
            nn.LeakyReLU(0.1, inplace = True),
            nn.Dropout(0.3),
            # nn.Upsample(size = (7, 32, 32)),
            nn.Upsample(size = (7, 16, 16)),

            nn.ConvTranspose3d(self.base * 2, self.base, kernel_size = 3, stride = 2, padding = 1, output_padding = 1),
            nn.BatchNorm3d(self.base),
            nn.LeakyReLU(0.1, inplace = True),
            nn.Dropout(0.3),
            # nn.Upsample(size = (13, 64, 64)),
            nn.Upsample(size = (13, 32, 32)),

            nn.ConvTranspose3d(self.base, 3, kernel_size = 3, stride = 2, padding = 1, output_padding = 1),
            nn.Upsample(size = (25, 64, 64)),
            # nn.Upsample(size = (28, 84, 84)),

            nn.Sigmoid()
        )

    def forward(self, x):
        x_enc = self.encoder(x)
        x_cap = self.decoder(x_enc)
        return x_cap, x_enc