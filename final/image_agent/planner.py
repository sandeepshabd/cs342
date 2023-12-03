import torch
import torch.nn.functional as F

def spatial_argmax(logit):
    
    weights = F.softmax(logit.view(logit.size(0), -1), dim=-1).view_as(logit)
    return torch.stack(((weights.sum(1) * torch.linspace(-1, 1, logit.size(2)).to(logit.device)[None]).sum(1),
                        (weights.sum(2) * torch.linspace(-1, 1, logit.size(1)).to(logit.device)[None]).sum(1)), 1)
    
                        
class Planner(torch.nn.Module):
    def __init__(self, channels=[16, 32, 64, 32]):
        super().__init__()
        
        def conv_block(in_channels, out_channels):

            return [
                torch.nn.BatchNorm2d(in_channels),
                torch.nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=2, padding=2),
                torch.nn.ReLU(inplace=True)]


        def upconv_block(in_channels, out_channels):

            return [
                torch.nn.BatchNorm2d(in_channels),
                torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                torch.nn.ReLU(inplace=True)]
            

        h, _conv, _upconv = 3, [], []
        for c in channels:
            _conv += conv_block(c, h)
            h = c

        for c in channels[:-3:-1]:
            _upconv += upconv_block(c, h)
            h = c

        _upconv += [torch.nn.BatchNorm2d(h), torch.nn.Conv2d(h, 1, 1, 1, 0)]

        self._conv = torch.nn.Sequential(*_conv)
        self._upconv = torch.nn.Sequential(*_upconv)   


    def forward(self, img):
        
        img = (img - self._mean[None, :, None, None].to(img.device)) / self._std[None, :, None, None].to(img.device)
        h = self._conv(img)
        x = self._upconv(h)

        output = (1 + spatial_argmax(x.squeeze(1))) 
        width = img.size(3)
        height = img.size(2)
        output = output * torch.as_tensor([width - 1,    height - 1]).float().to(
            img.device)

        return  output #300/400 range

        #return(x)

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


