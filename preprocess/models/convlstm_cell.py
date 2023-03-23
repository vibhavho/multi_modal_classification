import torch
import torch.nn as nn

# Original ConvLSTM cell as proposed by Shi et al.
class ConvLSTMCell(nn.Module):

    def __init__(
        self, in_channels, out_channels, kernel_size, padding, frame_size, bias = True
    ):

        super(ConvLSTMCell, self).__init__()  
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.frame_size = frame_size
        self.bias = bias
        # Idea adapted from https://github.com/ndrplz/ConvLSTM_pytorch
        self.conv = nn.Conv2d(
            in_channels = self.in_channels + self.out_channels, 
            out_channels = 4 * self.out_channels, 
            kernel_size = self.kernel_size, 
            padding = self.padding, 
            bias = self.bias
        )           
        height, width = self.frame_size
        # Initialize weights for Hadamard Products
        # self.W_ci = nn.Parameter(torch.Tensor(self.out_channels, *self.frame_size))
        # self.W_co = nn.Parameter(torch.Tensor(self.out_channels, *self.frame_size))
        # self.W_cf = nn.Parameter(torch.Tensor(self.out_channels, *self.frame_size))
        self.W_ci = nn.Parameter(torch.zeros(self.out_channels, height, width))
        self.W_co = nn.Parameter(torch.zeros(self.out_channels, height, width))
        self.W_cf = nn.Parameter(torch.zeros(self.out_channels, height, width))



    def forward(
        self, X, H_prev, C_prev
    ):

        # Idea adapted from https://github.com/ndrplz/ConvLSTM_pytorch
        conv_output = self.conv(torch.cat([X, H_prev], dim = 1))

        # Idea adapted from https://github.com/ndrplz/ConvLSTM_pytorch
        i_conv, f_conv, c_conv, o_conv = torch.split(conv_output, self.out_channels, dim = 1)

        input_gate = torch.sigmoid(i_conv + self.W_ci * C_prev)
        forget_gate = torch.sigmoid(f_conv + self.W_cf * C_prev)

        # Current Cell output
        C_curr = (forget_gate * C_prev) + (input_gate * torch.tanh(c_conv))
        output_gate = torch.sigmoid(o_conv + self.W_co * C_curr)
        # Current Hidden State
        H = output_gate * torch.tanh(C_curr)

        return H, C_curr