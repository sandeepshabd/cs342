

import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_spatial_argmax(logit):

    # Reshape and apply softmax
    weights = F.softmax(logit.view(logit.size(0), -1), dim=-1).view_as(logit)
    
    # Calculate argmax
    argmax_x = (weights.sum(1) * torch.linspace(-1, 1, logit.size(2), device=logit.device)).sum(1)
    argmax_y = (weights.sum(2) * torch.linspace(-1, 1, logit.size(1), device=logit.device)).sum(1)
    
    return torch.stack((argmax_x, argmax_y), dim=1)


class Planner(nn.Module):

    def __init__(self, channels=[16, 32, 64, 32]):
        super(Planner, self).__init__()

        # Define convolutional blocks
        def conv_block(in_channels, out_channels):
            return [nn.BatchNorm2d(out_channels), nn.Conv2d(out_channels, in_channels, 5, 2, 2), nn.ReLU(True)]

        # Define upconvolutional blocks
        def upconv_block(in_channels, out_channels):
            return [nn.BatchNorm2d(out_channels), nn.ConvTranspose2d(out_channels, in_channels, 4, 2, 1),
                    nn.ReLU(True)]

        # Building convolutional and upconvolutional layers
        current_channels, conv_layers, upconv_layers = 3, [], []
        for channel in channels:
            conv_layers += conv_block(channel, current_channels)
            current_channels = channel

        for channel in reversed(channels[:-2]):
            upconv_layers += upconv_block(channel, current_channels)
            current_channels = channel

        upconv_layers += [nn.BatchNorm2d(current_channels), nn.Conv2d(current_channels, 1, 1, 1, 0)]

        self.conv = nn.Sequential(*conv_layers)
        self.upconv = nn.Sequential(*upconv_layers)
        
        # Normalization parameters
        self.mean = torch.FloatTensor([0.4519, 0.5590, 0.6204])
        self.std = torch.FloatTensor([0.0012, 0.0018, 0.0020])

    def forward(self, img):

        # Normalize the image
        norm_img = (img - self.mean[None, :, None, None].to(img.device)) / self.std[None, :, None, None].to(img.device)
        conv_output = self.conv(norm_img)
        upconv_output = self.upconv(conv_output)

        spatial_argmax = (1 + compute_spatial_argmax(upconv_output.squeeze(1)))
        width, height = img.size(3), img.size(2)
        output = spatial_argmax * torch.as_tensor([width - 1, height - 1], dtype=torch.float32, device=img.device)

        return output  # Output in 300/400 range

