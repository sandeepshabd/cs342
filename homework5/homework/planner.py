import torch
import torch.nn.functional as F
import torch.nn as nn


def spatial_argmax(logit):
    """
    Compute the soft-argmax of a heatmap
    :param logit: A tensor of size BS x H x W
    :return: A tensor of size BS x 2 the soft-argmax in normalized coordinates (-1 .. 1)
    """
    weights = F.softmax(logit.view(logit.size(0), -1), dim=-1).view_as(logit)
    return torch.stack(((weights.sum(1) * torch.linspace(-1, 1, logit.size(2)).to(logit.device)[None]).sum(1),
                        (weights.sum(2) * torch.linspace(-1, 1, logit.size(1)).to(logit.device)[None]).sum(1)), 1)


        

class Planner(torch.nn.Module):
    
    class UpBlock(torch.nn.Module):
        def __init__(self, n_input, n_output, kernel_size=3, stride=2):
            super().__init__()
            self.c1 = torch.nn.ConvTranspose2d(n_input, n_output, kernel_size=kernel_size, padding=kernel_size // 2,
                                                stride=stride, output_padding=1)

        def forward(self, x):
            return F.relu(self.c1(x))
    
    class Block(torch.nn.Module):
        def __init__(self, n_input, n_output, kernel_size=3, stride=2):
            super().__init__()
            self.conv_layers = torch.nn.Sequential(
                torch.nn.Conv2d(n_input, n_output, kernel_size, stride=stride, padding=kernel_size // 2),
                torch.nn.BatchNorm2d(n_output),
                torch.nn.ReLU(),
                torch.nn.Conv2d(n_output, n_output, kernel_size, padding=kernel_size // 2),
                torch.nn.BatchNorm2d(n_output),
                torch.nn.ReLU(),
                torch.nn.Conv2d(n_output, n_output, kernel_size, padding=kernel_size // 2),
                torch.nn.BatchNorm2d(n_output),
                torch.nn.ReLU()
            )
            self.skip = torch.nn.Conv2d(n_input, n_output, kernel_size=1, stride=stride)

        def forward(self, x):
            return F.relu(self.conv_layers(x) + self.skip(x))
        
    def __init__(self, layers=[16, 32, 64, 128], n_class=1, kernel_size=3, use_skip=True):
        super().__init__()
        self.input_mean = torch.Tensor([0.2788, 0.2657, 0.2629])
        self.input_std = torch.Tensor([0.2064, 0.1944, 0.2252])
        self.use_skip = use_skip
        self.n_conv = len(layers)

        self.conv_blocks, self.upconv_blocks = self.create_conv_blocks(layers, kernel_size)
        self.classifier = torch.nn.Conv2d(layers[-1], n_class, 1)

    def create_conv_blocks(self, layers, kernel_size):
        conv_blocks = torch.nn.ModuleList()
        upconv_blocks = nn.ModuleList()
        channels = 3

        for l in layers:
            conv_blocks.append(self.Block(channels, l, kernel_size))
            channels = l

        for i, l in enumerate(reversed(layers)):
            up_channels = l + (layers[i - 1] if self.use_skip and i > 0 else 0)
            upconv_blocks.append(self.UpBlock(channels, up_channels, kernel_size))
            channels = up_channels

        return conv_blocks, upconv_blocks


    def forward(self, x):
        x = self.normalize_input(x)
        skip_connections = []
        for conv in self.conv_blocks:
            skip_connections.append(x)
            x = conv(x)

        for i, upconv in enumerate(reversed(self.upconv_blocks)):
            x = upconv(x)
            if self.use_skip:
                skip = skip_connections[-(i + 1)]
                x = F.interpolate(x, size=skip.shape[2:])
                x = torch.cat([x, skip], dim=1)

        encoder = self.classifier(x)
        encoder = torch.squeeze(encoder, dim=1)
        decoder = spatial_argmax(encoder)
        return decoder

    def normalize_input(self, x):
        return (x - self.input_mean[None, :, None, None].to(x.device)) / self.input_std[None, :, None, None].to(x.device)


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


if __name__ == '__main__':
    from .controller import control
    from .utils import PyTux
    from argparse import ArgumentParser


    def test_planner(args):
        # Load model
        planner = load_model().eval()
        pytux = PyTux()
        for t in args.track:
            steps, how_far = pytux.rollout(t, control, planner=planner, max_frames=1000, verbose=args.verbose)
            print(steps, how_far)
        pytux.close()


    parser = ArgumentParser("Test the planner")
    parser.add_argument('track', nargs='+')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    test_planner(args)
