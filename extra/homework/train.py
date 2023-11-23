import torch
import torch.nn as nn
from .models import TCN, save_model
from .utils import SpeechDataset, one_hot


def train(args):
    from os import path
    import torch.utils.tensorboard as tb
    
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    """
    Your code here, modify your code from prior assignments
    Hint: SGD might need a fairly high learning rate to work well here

    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = TCN().to(device)
    if args.continue_training:
        model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'tcn.th')))
    
    global_step = 0
    #optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-3)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    #loss = torch.nn.BCEWithLogitsLoss().to(device)
    loss = torch.nn.CrossEntropyLoss().to(device)
    
    train_data = list(SpeechDataset('data/train.txt'))
    valid_data = list(SpeechDataset('data/valid.txt'))
    
    
    def make_random_batch(batch_size, is_train_data=True):
        B = []
        data = train_data if is_train_data else valid_data
        for i in range(batch_size):
            B.append(data[np.random.randint(0, len(data) - 1)][:,:])
        return torch.stack(B, dim=0)
    

    for epoch in range(args.num_epoch):

        model.train()
        loss_vals, valid_loss_vals = [], []
        batch = make_random_batch(args.batch_size)
        batch_data = batch[:, :, :-1].to(device)
        batch_label = batch.argmax(dim=1).to(device)
        
        output = model(batch_data)
        loss_val = loss(output, batch_label)

        loss_vals.append(loss_val.detach().cpu().numpy())
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()
        global_step += 1
        
    avg_loss = sum(loss_vals) / len(loss_vals)
    train_logger.add_scalar('loss', avg_loss, global_step=epoch)

    model.eval()
    valid_batches = make_random_batch(args.batch_size, is_train_data=False)
    valid_batch_data = valid_batches[:, :, :-1].to(device)
    valid_batch_label = valid_batches.argmax(dim=1).to(device)
    valid_o = model(valid_batch_data)
    valid_loss_val = loss(valid_o, valid_batch_label)

    valid_loss_vals.append(valid_loss_val.detach().cpu().numpy())
    avg_valid_loss = sum(valid_loss_vals) / len(valid_loss_vals)
    valid_logger.add_scalar('loss', avg_valid_loss, global_step)
    print('epoch %-3d \t loss = %0.3f \t val loss = %0.3f' % (epoch, avg_loss, avg_valid_loss))
    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here
    parser.add_argument('-t', '--train_path', type=str)
    parser.add_argument('-v', '--valid_path', type=str)
    parser.add_argument('-ep', '--epochs', type=int, default=50)
    parser.add_argument('-b', '--batch_size', type=int, default=100)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01)
    parser.add_argument('-mom', '--momentum', type=float, default=0.9)
    parser.add_argument('-ls', '--log_suffix', type=str, default='')
    parser.add_argument('-cl','--clear_cache', type=bool, default=False)

    args = parser.parse_args()
    train(args)
