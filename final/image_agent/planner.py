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
        super(Planner, self).__init__()

        self._conv = nn.Sequential(*self._make_layers(channels, conv=True))
        self._upconv = nn.Sequential(*self._make_layers(channels, conv=False))

        # Normalization parameters
        self._mean = torch.FloatTensor([0.4519, 0.5590, 0.6204]).view(1, 3, 1, 1)
        self._std = torch.FloatTensor([0.0012, 0.0018, 0.0020]).view(1, 3, 1, 1)

    def _make_layers(self, channels, conv=True):
        layers = []
        in_channels = 3

        for i, out_channels in enumerate(channels):
            if conv or i > 0:  # Skip first for upconvolution
                layers.append(nn.BatchNorm2d(in_channels))

            if conv:
                layers.append(nn.Conv2d(in_channels, out_channels, 5, 2, 2))
            else:
                layers.append(nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1))

            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels

        if not conv:
            # Additional layer for the upconvolution path
            layers.append(nn.Conv2d(in_channels, 1, 1))

        return layers
                        
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

