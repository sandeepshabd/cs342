import torch



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
    
    class DetectorUpBlock(torch.nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size=3):
            super().__init__()
            self.conv = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2, stride=2, output_padding=1)

        def forward(self, x):
            return torch.nn.functional.relu(self.conv(x))
    
    class DetectorBlock(torch.nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size=3):
            super().__init__()
            
            self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2, stride=2)
            self.batch1 = torch.nn.BatchNorm2d(out_channels)
            self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
            self.batch2 = torch.nn.BatchNorm2d(out_channels)
            self.conv3 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
            self.batch3 = torch.nn.BatchNorm2d(out_channels)
           
           
           
            
            
            if in_channels != out_channels:
                self.skip = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2)
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
    
    def __init__(self, layers=[16, 32, 64, 128], n_output_channels=3, kernel_size=3, use_skip=True):
        super(Detector, self).__init__()
        
        self.blocks = torch.nn.ModuleList()
        in_channels = 3  # Input image has 3 channels
        
        for out_channels in layers:
            self.blocks.append(self.DetectorBlock(in_channels, out_channels, kernel_size))
            in_channels = out_channels

        # Adding an UpBlock at the end to upsample the output of the last block
        self.upblocks = torch.nn.ModuleList([
            self.DetectorUpBlock(128, 64, kernel_size),
            self.DetectorUpBlock(64, 32, kernel_size),
            self.DetectorUpBlock(32, 16, kernel_size),
            self.DetectorUpBlock(16, n_output_channels, kernel_size)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        for upblock in self.upblocks:
            x= upblock(x)
        return x
        


    def detect(self, image):
        """
           Your code here.
           Implement object detection here.
           @image: 3 x H x W image
           @return: Three list of detections [(score, cx, cy, w/2, h/2), ...], one per class,
                    return no more than 30 detections per image per class. You only need to predict width and height
                    for extra credit. If you do not predict an object size, return w=0, h=0.
           Hint: Use extract_peak here
           Hint: 'Make sure to return three python lists of tuples of (float, int, int, float, float) and not a pytorch
                 scalar. Otherwise pytorch might keep a computation graph in the background and your program will run
                 out of memory.'
        """

    
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        heatmaps = self.forward(image).to(device)
        heatmaps = model.squeeze(dim=0)
        
        result = []
        
        
        for i in range(0,3):
            peaks = extract_peak(heatmaps[i],max_det=30)
            channel_list = []
            
            for peak in peaks: 
                channel_list.append([peak[0].item(),peak[1].item(),peak[2].item(),0.,0.])
            result.append(channel_list)
        return result


    


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
    """
    Shows detections of your detector
    """
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
