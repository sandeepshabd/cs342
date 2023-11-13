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
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(n_output, n_output, kernel_size, padding=kernel_size // 2),
                torch.nn.BatchNorm2d(n_output),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(n_output, n_output, kernel_size, padding=kernel_size // 2),
                torch.nn.BatchNorm2d(n_output),
                torch.nn.ReLU(inplace=True)
            )
            self.skip = torch.nn.Conv2d(n_input, n_output, kernel_size=1, stride=stride)

        def forward(self, x):
            return F.relu(self.conv_layers(x) + self.skip(x))
        
    def __init__(self, layers=[16, 32, 64, 128], n_class=1, kernel_size=3, use_skip=True):
        super().__init__()

        self.input_mean = torch.Tensor([0.3, 0.25, 0.27])
        self.input_std = torch.Tensor([0.19, 0.19, 0.18])
        
        c_input = 3
        self.use_skip = use_skip
        self.n_conv = len(layers)
        
        skip_layer_size = [3] + layers[:-1]
        for i, layer_i in enumerate(layers):
            self.add_module('conv%d' % i, self.Block(c_input, layer_i, kernel_size, 2))
            c_input = layer_i


        for i, layer_i in list(enumerate(layers))[::-1]:
            self.add_module('upconv%d' % i, self.UpBlock(c_input, layer_i, kernel_size, 2))
            c_input = layer_i
            if self.use_skip:
                c_input += skip_layer_size[i]
                
                
        self.classifier = torch.nn.Conv2d(c_input, n_class, 1)


    def forward(self, x):
        """
        Your code here
        Predict the aim point in image coordinate, given the supertuxkart image
        @img: (B,3,96,128)
        return (B,2)
        """
        z = (x - self.input_mean[None, :, None, None].to(x.device)) / self.input_std[None, :, None, None].to(x.device)
        up_activation = []
        
        for i in range(self.n_conv):
            up_activation.append(z)
            z = self._modules['conv%d' % i](z)

        for i in reversed(range(self.n_conv)):
            z = self._modules['upconv%d' % i](z)
            z = z[:, :, :up_activation[i].size(2), :up_activation[i].size(3)]
            if self.use_skip:
                z = torch.cat([z, up_activation[i]], dim=1)
                
                
        encoder = self.classifier(z)
        encoder = torch.squeeze(encoder,dim=1)
        
        decoder = spatial_argmax(encoder)
        return decoder

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
