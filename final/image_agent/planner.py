import torch
import torch.nn.functional as F

def spatial_argmax(logit):
    
    weights = F.softmax(logit.view(logit.size(0), -1), dim=-1).view_as(logit)
    return torch.stack(((weights.sum(1) * torch.linspace(-1, 1, logit.size(2)).to(logit.device)[None]).sum(1),
                        (weights.sum(2) * torch.linspace(-1, 1, logit.size(1)).to(logit.device)[None]).sum(1)), 1)
    
import torch
import torch.nn as nn

class Planner(nn.Module):
    def __init__(self, channels=[16, 32, 64, 32]):
        super().__init__()

        # Define methods for creating convolutional and up-convolutional blocks
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=2, padding=2),
                nn.ReLU(inplace=True)
            )

        def upconv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                nn.ReLU(inplace=True)
            )

        # Initialize variables
        in_channels = 3
        _conv = []
        _upconv = []

        # Create convolutional layers
        for out_channels in channels:
            _conv.append(conv_block(in_channels, out_channels))
            in_channels = out_channels

        # Create up-convolutional layers
        for out_channels in channels[:-3:-1]:
            _upconv.append(upconv_block(in_channels, out_channels))
            in_channels = out_channels

        # Add final layer to upconvolution
        _upconv.append(nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, 1, kernel_size=1)
        ))

        # Combine layers into sequential models
        self._conv = nn.Sequential(*_conv)
        self._upconv = nn.Sequential(*_upconv)
        self._mean = torch.FloatTensor([0.4519, 0.5590, 0.6204])
        self._std = torch.FloatTensor([0.0012, 0.0018, 0.0020])
                        
    def forward(self, img):
        normalized_img = (img - self._mean.to(img.device)) / self._std.to(img.device)
        conv_output = self._conv(normalized_img)
        upconv_output = self._upconv(conv_output)
        spatial_max = spatial_argmax(upconv_output.squeeze(1))
        scale = torch.tensor([img.size(3) - 1, img.size(2) - 1], dtype=torch.float32, device=img.device)
        output = (1 + spatial_max) * scale
        return output



def save_model(model):
    from torch import save
    from os import path
    if isinstance(model, Planner):
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'planner.th'))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model():
    from torch import load
    from os import path
    r = Planner()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'planner.th'), map_location='cpu'))
    return r

