from anyio import Path
from .planner import Planner, save_model 
import torch
from .utils import load_data
from . import dense_transforms
import torch.utils.tensorboard as tb
import numpy as np
import torch
import inspect

from os import path
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from PIL import  ImageDraw

def train(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = Planner().to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')).to(device)
    basic_transform = 'Compose([ColorJitter(0.9, 0.9, 0.9, 0.1), RandomHorizontalFlip(), ToTensor()])'

    print(model)

    train_logger =  None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))

    if args.continue_training:
        model.load_state_dict(torch.load(path(__file__).resolve().parent / 'planner.th'))

    loss_fn = torch.nn.L1Loss()  
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    transform = eval(basic_transform, {k: v for k, v in inspect.getmembers(dense_transforms) if inspect.isclass(v)})

    print(f'----start loading data---')
    print(transform)
    train_data = load_data(transform=transform, num_workers=4)
    print(f'data loadded size {len(train_data)}')
   
    global_step = 0
    for epoch in range(args.num_epoch):
        model.train()
        losses = []

        for img, label in train_data:
            img, label = img.to(device), label.to(device)
            h, w = img.shape[2:]

            pred = model(img)
            xy = torch.cat((label[:, :1].clamp(min=0, max=w), label[:, 1:].clamp(min=0, max=h)), dim=1)

            loss_val = loss_fn(pred, xy)

            if train_logger and global_step % 10 == 0:
                with torch.no_grad():  # Saves memory and computations
                    train_logger.add_scalar('loss', loss_val.item(), global_step)
                    log(train_logger, img, label, pred, global_step)

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            losses.append(loss_val.item())
            global_step += 1
        
        avg_loss = np.mean(losses)
        if not train_logger:
            print(f'epoch {epoch:3d} \t loss = {avg_loss:.3f}')
        save_model(model)
    save_model(model)

def log(logger, img, label, pred, global_step):
    # Convert tensor to PIL Image
    pil_img = TF.to_pil_image(img[0].cpu())

    # Draw circles on the image
    draw = ImageDraw.Draw(pil_img)
    WH2 = np.array(pil_img.size)/2
    label_point = WH2*(label[0].cpu().detach().numpy()+1)
    pred_point = WH2*(pred[0].cpu().detach().numpy()+1)
    draw.ellipse((label_point[0]-2, label_point[1]-2, label_point[0]+2, label_point[1]+2), outline='green', width=2)
    draw.ellipse((pred_point[0]-2, pred_point[1]-2, pred_point[0]+2, pred_point[1]+2), outline='red', width=2)

    # Convert PIL Image back to tensor
    tensor_img = TF.to_tensor(pil_img).unsqueeze(0)

    # Use torchvision to make a grid (if you have more than one image, otherwise it's not necessary)
    # image_grid = vutils.make_grid(tensor_img)

    # Add image grid to TensorBoard
    logger.add_image('viz', tensor_img, global_step)
    del ax, fig


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here
    parser.add_argument('-w', '--num_workers', type=int, default=4)
    parser.add_argument('-n', '--num_epoch', type=int, default=60)
    parser.add_argument('-t', '--transform', default='Compose([ColorJitter(0.2, 0.5, 0.5, 0.2), RandomHorizontalFlip(), ToTensor()])')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-c', '--continue_training', action='store_true')
    args = parser.parse_args()
    train(args)
