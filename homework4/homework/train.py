import torch
import numpy as np

from .models import Detector, save_model
from .utils import load_detection_data
from . import dense_transforms
import torch.utils.tensorboard as tb
import inspect


def train(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    from os import path
    model = Detector()
    model = model.to(device)
    
    
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    """
    Your code here, modify your HW3 code
    Hint: Use the log function below to debug and visualize your model
    """
    if args.continue_training:
        model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'det.th')))



    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    
    #loss = FocalLoss(gamma=args.gamma, alpha=args.alpha).to(device)
    loss = torch.nn.BCEWithLogitsLoss().to(device)
    

    transform = eval('Compose([ColorJitter(0.9, 0.9, 0.9, 0.1), RandomHorizontalFlip(), ToTensor(), ToHeatmap()])', {k: v for k, v in inspect.getmembers(dense_transforms) if inspect.isclass(v)})
    validation_transform=eval('Compose([ToTensor(), ToHeatmap()])', {k: v for k, v in inspect.getmembers(dense_transforms) if inspect.isclass(v)})
    
    train_data = load_detection_data('dense_data/train', num_workers=4, transform=transform)
    valid_data = load_detection_data('dense_data/valid', num_workers=4, transform=validation_transform)
    
    global_step = 0
    for epoch in range(20):
        model.train()
        loss_value = []
        
        for image, label, size in train_data:
            image = image.to(device)
            label = image.to(device)
            size = image.to(device)
            print(image.shape)
            output = model(image).to(device)
            print(output.shape)
            print(label.shape)
            loss_data = loss(output, label).to(device)
            
            if global_step % 100 == 0  and train_logger is not None:
                log(train_logger, image, label, output, global_step)
                train_logger.add_scalar('loss', loss_data, global_step)
                loss_value.append(loss_data)
                
            optimizer.zero_grad()
            loss_data.backward()
            optimizer.step()
            global_step += 1
        
        if(len(loss_value) >0 )  :  
            avg_loss = sum(loss_value) / len(loss_value)
        print('avg loss for epoch',epoch,'=',avg_loss.item())
        model.eval()
  
        
        for image, label, size in valid_data:
            image, label, size = image.to(device), label.to(device), size.to(device)
            output = model(image)

        if valid_logger is not None:
            log(valid_logger, image, label, output, global_step)
        save_model(model)
            
            
        


def log(logger, imgs, gt_det, det, global_step):
    """
    logger: train_logger/valid_logger
    imgs: image tensor from data loader
    gt_det: ground-truth object-center maps
    det: predicted object-center heatmaps
    global_step: iteration
    """
    logger.add_images('image', imgs[:16], global_step)
    logger.add_images('label', gt_det[:16], global_step)
    logger.add_images('pred', torch.sigmoid(det[:16]), global_step)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    parser.add_argument('-c', '--continue_training', action='store_true')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
