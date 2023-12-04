from anyio import Path
from .planner import Planner, save_model 
import torch
from .utils import load_data
from . import dense_transforms
import torch.utils.tensorboard as tb
import numpy as np
import torch
import inspect
from datetime import datetime

from os import path
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from PIL import  ImageDraw

def train(args):

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = Planner().to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')).to(device)
    basic_transform = 'Compose([ColorJitter(0.9, 0.9, 0.9, 0.1), RandomHorizontalFlip(), ToTensor()])'
    
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))

    if args.continue_training:
        model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'planner.th')))

    #loss = torch.nn.HuberLoss()
    #loss = torch.nn.CrossEntropyLoss()
    loss = torch.nn.L1Loss()
    #loss = torch.nn.MSELoss(reduction='mean') 

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.25)
    transform = eval(basic_transform, {k: v for k, v in inspect.getmembers(dense_transforms) if inspect.isclass(v)})
    train_data = load_data(transform=transform, num_workers=4)
    isStart = True
    global_step = 0
    for epoch in range(300):

        model.train()
        losses = []
        for img, label in train_data:
            img, label = img.to(device), label.to(device)
            h, w = img.size()[2], img.size()[3]
            
            out  = model(img)
            x,y = label.chunk(2, dim=1)
            xy = torch.cat((x.clamp(min=0.0,max=w),y.clamp(min=0.0,max=h)),dim=1) 

            xy = xy.to(device)
            
            loss_val = loss(out, xy)
            #print(loss_val)
            train_logger.add_scalar('loss', loss_val, global_step)
            if global_step % 10 == 0:
                log(train_logger, img, label, out, global_step)

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            global_step += 1
            
            if(loss_val is not None ):
                losses.append(loss_val.detach().cpu().numpy())
        
        avg_loss = np.mean(losses)
        nowTime = datetime.now()
        date_time_str = nowTime.strftime("%Y-%m-%d %H:%M:%S")
        print("Current Time =", date_time_str,' ,epoch=',epoch,' ,Avergae Loss=',avg_loss)
        
        
        if isStart:
            lowest_loss_val = avg_loss
            isStart = False
            
        if (avg_loss - lowest_loss_val) < 1.5:
            save_model(model)
            print("Model save: Current Time =", date_time_str,' ,epoch=',epoch,' ,Avergae Loss=',avg_loss)
            lowest_loss_val = avg_loss
        
        #scheduler.step()


def log(logger, img, label, pred, global_step):
    import matplotlib.pyplot as plt
    import torchvision.transforms.functional as TF
    fig, ax = plt.subplots(1, 1)
    ax.imshow(TF.to_pil_image(img[0].cpu()))
    WH2 = np.array([img.size(-1), img.size(-2)])/2
    ax.add_artist(plt.Circle(WH2*(label[0].cpu().detach().numpy()+1), 2, ec='g', fill=False, lw=1.5))
    ax.add_artist(plt.Circle(WH2*(pred[0].cpu().detach().numpy()+1), 2, ec='r', fill=False, lw=1.5))
    logger.add_figure('viz', fig, global_step)
    del ax, fig


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here
    parser.add_argument('-n', '--num_epoch', type=int, default=300)
    parser.add_argument('-w', '--num_workers', type=int, default=4)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-c', '--continue_training', action='store_true')
    parser.add_argument('-t', '--transform', default='Compose([ColorJitter(0.2, 0.5, 0.5, 0.2), RandomHorizontalFlip(), ToTensor()])')

    args = parser.parse_args()
    train(args)
