import torch


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, input, target):
        input = input.contiguous().view(input.size(0), input.size(1), -1)
        input = input.transpose(1,2)
        input = input.contiguous().view(-1, input.size(2)).squeeze()
        target = target.contiguous().view(target.size(0), target.size(1), -1)
        target = target.transpose(1,2)
        target = target.contiguous().view(-1, target.size(2)).squeeze()
        BCE_loss = torch.nn.functional.binary_cross_entropy_with_logits(input, target, reduction=self.reduction,pos_weight=torch.tensor([.69,.95,.85]).to(target.device))
        pt = torch.exp(-BCE_loss) # prevents nans when probability 0
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()


def extract_peak(heatmap, max_pool_ks=7, min_score=-5, max_det=100):
    """
       Your code here.
       Extract local maxima (peaks) in a 2d heatmap.
       @heatmap: H x W heatmap containing peaks (similar to your training heatmap)
       @max_pool_ks: Only return points that are larger than a max_pool_ks x max_pool_ks window around the point
       @min_score: Only return peaks greater than min_score
       @return: List of peaks [(score, cx, cy), ...], where cx, cy are the position of a peak and score is the
                heatmap value at the peak. Return no more than max_det peaks per image
    """
    #if not torch.is_tensor(heatmap):
        #heatmap = torch.tensor(heatmap)

    # Apply 2D max pooling
    pooled = torch.nn.functional.max_pool2d(heatmap[None, None], kernel_size=max_pool_ks, stride=1, padding=max_pool_ks // 2)
    pooled = pooled.to(heatmap.device)
    pooled = torch.squeeze(pooled)
    # Detect peaks: places where the original heatmap and its max pooled version are the same, and also above the threshold
    peaks = (heatmap > min_score)

    # Get the scores and their coordinates
    #scores, y_coords, x_coords = torch.where(peaks, heatmap, torch.tensor(0.).to(pooled.device)).flatten().topk(k=max_det,dim=0,sorted=False)
    #torch.where(peaks, heatmap, torch.tensor(float('-inf'))).flatten().topk(max_det)

    # Return list of peaks
    #return [(score.item(), x.item(), y.item()) for score, y, x in zip(scores, y_coords, x_coords)]
    

    new_heatmap = torch.where(peaks,heatmap,torch.tensor(0.).to(pooled.device)).to(pooled.device)
    comparison = heatmap>=pooled
    comparison.to(pooled.device)
    
    res = torch.where(comparison,new_heatmap,torch.tensor(0.).to(pooled.device))
    res = res.to(pooled.device)
    
    nonzero = torch.nonzero(res,as_tuple=True)
    nums = res[nonzero]
    
    indices = range(0,len(nums))
    if len(nums) > max_det:
        values,indices = torch.topk(nums,k=max_det,dim=0,sorted=False)
    peaks = list(zip(nums[indices],nonzero[1][indices],nonzero[0][indices]))
    return peaks


class Detector(torch.nn.Module):

    class DetectorBlock(torch.nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size=3,stride=2):
            super().__init__()
            
            self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2, stride=stride)
            self.batch1 = torch.nn.BatchNorm2d(out_channels)
            self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
            self.batch2 = torch.nn.BatchNorm2d(out_channels)
            self.conv3 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
            self.batch3 = torch.nn.BatchNorm2d(out_channels)
           
         
            if in_channels != out_channels:
                self.skip = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
            else:
                self.skip = torch.nn.Identity()
                
            
            #self.skip = torch.nn.Conv2d(n_input, n_output, kernel_size=1, stride=stride)

        def forward(self, x):
            return torch.nn.functional.relu(
                self.batch3(
                    self.conv3(
                        torch.nn.functional.relu(
                            self.batch2(
                                self.conv2(
                                    torch.nn.functional.relu(
                                        self.batch1(
                                            self.conv1(x)))))))) 
                          + 
                          self.skip(x))

    class DetectorUpBlock(torch.nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size=3, stride=2):
            super().__init__()
            self.conv = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, 
                                                 padding=kernel_size // 2, stride=stride, output_padding=1)

        def forward(self, x):
            return torch.nn.functional.relu(self.conv(x))

    def __init__(self, layers=[16, 32, 64, 128], n_class=3, kernel_size=3, use_skip=True):
        """
           Your code here.
           Setup your detection network
        """
        super().__init__()
        self.input_mean = torch.Tensor([0.2, 0.2, 0.2])
        self.input_std = torch.Tensor([0.2, 0.2, 0.2])

        c = 3
        self.use_skip = use_skip
        self.n_conv = len(layers)
        skip_layer_size = [3] + layers[:-1]
        for i, l in enumerate(layers):
            self.add_module('conv%d' % i, self.DetectorBlock(c, l, kernel_size, 2))
            c = l
        # Produce lower res output
        for i, l in list(enumerate(layers))[::-1]:
            self.add_module('upconv%d' % i, self.DetectorUpBlock(c, l, kernel_size, 2))
            c = l
            if self.use_skip:
                c += skip_layer_size[i]
        self.classifier = torch.nn.Conv2d(c, n_class, 1)
        self.size = torch.nn.Conv2d(c, 2, 1)

    def forward(self, x):
        """
           Your code here.
           Implement a forward pass through the network, use forward for training,
           and detect for detection
        """
        z = (x - self.input_mean[None, :, None, None].to(x.device)) / self.input_std[None, :, None, None].to(x.device)
        up_activation = []
        for i in range(self.n_conv):
            # Add all the information required for skip connections
            up_activation.append(z)
            z = self._modules['conv%d' % i](z)

        for i in reversed(range(self.n_conv)):
            z = self._modules['upconv%d' % i](z)
            # Fix the padding
            z = z[:, :, :up_activation[i].size(2), :up_activation[i].size(3)]
            # Add the skip connection
            if self.use_skip:
                z = torch.cat([z, up_activation[i]], dim=1)
        return self.classifier(z), self.size(z)

    def detect(self, image, **kwargs):

        cls, size = self.forward(image[None])
        size = size.cpu()
        return [[(s, x, y, float(size[0, 0, y, x]), float(size[0, 1, y, x]))  #returns a list of five-tuples
                        for s, x, y in extract_peak(c, max_det=30, **kwargs) 
                 ]for c in cls[0]]


def save_model(model):
    from torch import save
    from os import path
    return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'det.th'))


def load_model():
    from torch import load
    from os import path
    r = Detector()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'det.th'), map_location='cpu'))
    return r


if __name__ == '__main__':
    from .utils import DetectionSuperTuxDataset
    dataset = DetectionSuperTuxDataset('dense_data/valid', min_size=0)
    import torchvision.transforms.functional as TF
    from pylab import show, subplots
    import matplotlib.patches as patches

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    fig, axs = subplots(3, 4)
    model = load_model().eval().to(device)
    for i, ax in enumerate(axs.flat):
        im, kart, bomb, pickup = dataset[i]
        ax.imshow(TF.to_pil_image(im), interpolation=None)
        for k in kart:
            ax.add_patch(
                patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], facecolor='none', edgecolor='r'))
        for k in bomb:
            ax.add_patch(
                patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], facecolor='none', edgecolor='g'))
        for k in pickup:
            ax.add_patch(
                patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], facecolor='none', edgecolor='b'))
        detections = model.detect(im.to(device))
        for c in range(3):
            for s, cx, cy, w, h in detections[c]:
                ax.add_patch(patches.Circle((cx, cy), radius=max(2 + s / 2, 0.1), color='rgb'[c]))
        ax.axis('off')
    show()