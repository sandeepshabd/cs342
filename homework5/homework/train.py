from .planner import Planner, save_model 
import torch
from torch import save
from os import path
import torch.utils.tensorboard as tb
import numpy as np
from .utils import load_data
from . import dense_transforms

def train(args):

    model = Planner()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    
    import inspect
    transform = eval(args.transform, {k: v for k, v in inspect.getmembers(dense_transforms) if inspect.isclass(v)})
    train_loader = load_data('drive_data', num_workers=4, transform=transform)
    
    #loss_function = torch.nn.BCEWithLogitsLoss(reduction='none')
    size_loss_function = torch.nn.MSELoss(reduction='mean')
    
    trainData(model, optimizer, size_loss_function, train_loader, device, args, train_logger)


def trainData(model, optimizer, loss_function, train_data_loader, device, args, logger=None):
    model = model.to(device)

    if args.continue_training:
        load_model_state(model, args.model_path)

    global_step = 0
    for epoch in range(args.num_epochs):
        model.train()
        loss_vals = []

        for img, gt_det in train_data_loader:
            img, gt_det = img.to(device), gt_det.to(device)

            det = model(img)
            loss_val = loss_function(det, gt_det)

            if logger is not None and global_step % args.log_frequency == 0:
                log_metrics(logger, img, gt_det, det, loss_val, global_step)

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            loss_vals.append(loss_val.item())
            global_step += 1

        avg_loss = sum(loss_vals) / len(loss_vals)
        print(f'Average loss for epoch {epoch} = {avg_loss}')
        save_model(model, args.model_path)

def load_model_state(model, model_path):
    from os import path
    model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), model_path)))

def save_model(model, model_path):
    from torch import save
    from os import path
    save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), model_path))

def log_metrics(logger, img, gt_det, det, loss_val, global_step):
    logger.add_images('images', img, global_step)
    logger.add_images('ground_truth', gt_det, global_step)
    logger.add_images('detection', det, global_step)
    logger.add_scalar('loss', loss_val, global_step)


def log_metrics(logger, img, gt_det, det, loss_val, global_step):
    logger.add_images('images', img, global_step)
    logger.add_images('ground_truth', gt_det, global_step)
    logger.add_images('detection', det, global_step)
    logger.add_scalar('loss', loss_val, global_step)
    

def log(logger, img, label, pred, global_step):
    """
    logger: train_logger/valid_logger
    img: image tensor from data loader
    label: ground-truth aim point
    pred: predited aim point
    global_step: iteration
    """
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

    args = parser.parse_args()
    train(args)
