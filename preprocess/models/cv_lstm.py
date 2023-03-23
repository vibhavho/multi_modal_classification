import torch
import torch.nn as nn

from .convlstm_layer import *

class ConvLSTMAutoEncoder(nn.Module):
    def __init__(
        self, lstm_layers = 3
    ):
        super(
            ConvLSTMAutoEncoder, self
        ).__init__()
        
        self.base = 64
        self.lstm_layers = lstm_layers

        self.encoder = nn.Sequential(
            nn.Conv2d(3, self.base, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(self.base),
            nn.LeakyReLU(0.1, True),
            nn.Dropout(0.2),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(self.base, self.base * 2, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(self.base * 2),
            nn.LeakyReLU(0.1, True),
            nn.Dropout(0.2),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(self.base * 2, self.base * 4, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(self.base * 4),
            nn.LeakyReLU(0.1, True),
            nn.Dropout(0.2),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(self.base * 4, self.base * 8, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(self.base * 8),
            nn.MaxPool2d(2, 2)
        )

        self.convlstm = nn.Sequential()
        self.convlstm.add_module(
            "convlstm1", ConvLSTM(
                in_channels = self.base * 8,
                out_channels = self.base * 8,
                kernel_size = (3, 3),
                padding = (1, 1),
                frame_size = (4, 4)
            )
        )
        self.convlstm.add_module(
            "batchnorm1", nn.BatchNorm3d(self.base * 8)
        )

        for layer_idx in range(2, self.lstm_layers + 1):
            self.convlstm.add_module(
                f"convlstm{layer_idx}",
                ConvLSTM(
                    in_channels = self.base * 8,
                    out_channels = self.base * 8,
                    kernel_size = (3, 3),
                    padding = (1, 1),
                    frame_size = (4, 4)
                )
            )
            self.convlstm.add_module(
                f"batchnorm{layer_idx}",
                nn.BatchNorm3d(self.base * 8)
            )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.base * 8, self.base * 4, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(self.base * 4),
            nn.LeakyReLU(0.1, True),
            nn.Dropout(0.2),
            nn.Upsample(size=(8, 8)),

            nn.ConvTranspose2d(self.base * 4, self.base * 2, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(self.base * 2),
            nn.LeakyReLU(0.1, True),
            nn.Dropout(0.2),
            nn.Upsample(size=(16, 16)),

            nn.ConvTranspose2d(self.base * 2, self.base, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(self.base),
            nn.LeakyReLU(0.1, True),
            nn.Dropout(0.2),
            nn.Upsample(size=(32, 32)),

            nn.ConvTranspose2d(self.base, 3, kernel_size = 3, stride = 1, padding = 1),
            nn.Upsample(size=(64, 64)),

            nn.Sigmoid()
        )
        

    def forward(self, x):
        enc_x = self.encoder(x) # (28, 512, 5, 5)
        enc_x = torch.unsqueeze(enc_x, dim = 0) # (1, 28, 512, 5, 5)
        enc_x = enc_x.permute(0, 2, 1, 3, 4)
        conv_emb = torch.squeeze(self.convlstm(enc_x), dim = 0) # (512, 28, 5, 5) 
        conv_emb = conv_emb.permute(1, 0, 2, 3)
        dec_x = self.decoder(conv_emb) # (28, 1, 84, 84)
        return dec_x, conv_emb