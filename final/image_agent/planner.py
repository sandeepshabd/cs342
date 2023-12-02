
import torch
import torch.nn.functional as F



def spatial_argmax(logit):
    
    weights = F.softmax(logit.view(logit.size(0), -1), dim=-1).view_as(logit)
    return torch.stack(((weights.sum(1) * torch.linspace(-1, 1, logit.size(2)).to(logit.device)[None]).sum(1),
                        (weights.sum(2) * torch.linspace(-1, 1, logit.size(1)).to(logit.device)[None]).sum(1)), 1)
    
                        
import torch
import torch.nn as nn
import torch.nn.functional as F

class Planner(nn.Module):
    def __init__(self, channels=[16, 32, 64, 32]):
        super(Planner, self).__init__()

        self.conv_layers = self._make_layers(channels, self._conv_block)
        self.upconv_layers = self._make_layers(channels[::-1], self._upconv_block, reverse=True)

        self.normalization_mean = torch.FloatTensor([0.4519, 0.5590, 0.6204]).view(1, 3, 1, 1)
        self.normalization_std = torch.FloatTensor([0.0012, 0.0018, 0.0020]).view(1, 3, 1, 1)

    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, 5, 2, 2),
            nn.ReLU(inplace=True)
        )

    def _upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1),
            nn.ReLU(inplace=True)
        )

    def _make_layers(self, channels, block_fn, reverse=False):
        layers = []
        in_channels = 3
        for out_channels in channels[:-1] if reverse else channels:
            layers.append(block_fn(in_channels, out_channels))
            in_channels = out_channels
        if reverse:
            layers.append(nn.Conv2d(in_channels, 1, 1))
        return nn.Sequential(*layers)

    def forward(self, img):
        normalized_img = (img - self.normalization_mean.to(img.device)) / self.normalization_std.to(img.device)
        conv_output = self.conv_layers(normalized_img)
        upconv_output = self.upconv_layers(conv_output)

        output = (1 + spatial_argmax(upconv_output.squeeze(1)))
        width, height = img.size(3), img.size(2)
        output *= torch.tensor([width - 1, height - 1], dtype=torch.float32, device=img.device)

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


