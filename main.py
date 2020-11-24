from __future__ import print_function

import argparse

import torch

from dataset import SyntheticCellDataset
from model import UNet
from loss import DiceLoss
from utils import save_checkpoint

parser = argparse.ArgumentParser(
    prog="training.py", description="UNet Training and evaluation"
)

parser.add_argument(
    'img_dir',
    help="Image dir name", type=str
)
parser.add_argument(
    'mask_dir',
    help="Mask dir name", type=str
)
parser.add_argument(
    'model_save_dir',
    help="model save dir name", type=str
)
parser.add_argument(
    '--use_cuda',
    action='store',
    default=True,
    help="Try to use CUDA. The default value is no. All default values are for CPU mode.",
    type=bool, dest='use_cuda'
)
parser.add_argument(
    'split_ratio',
    help="Train Test ratio", default=0.8, type=float,
)
parser.add_argument(
    'batch_size',
    help="Train batch", default=64, type=int,
)
parser.add_argument(
    'lr',
    help="Learning Rate", default=0.0001, type=float,
)

parser.add_argument(
    'N_epoch',
    help="Total epoch", default=100, type=int,
)

def train(model, train_loader, device, optimizer):
    model.train()
    steps = len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
    train_loss = 0.0
    
    for i, data in enumerate(train_loader):
        x,y = data

        optimizer.zero_grad()
        y_pred = model(x.to(device))
        loss = dsc_loss(y_pred, y.to(device))

        loss.backward()
        optimizer.step()
        scheduler.step()
        
        train_loss += loss.item()
    return model, train_loss/len(train_loader), optimizer
    
def validate(model, val_loader, device):

    with torch.no_grad():
        model.train()
        val_loss = 0.0
    
        for i, data in enumerate(val_loader):
            x,y = data

            y_pred = model(x.to(device))
            loss = dsc_loss(y_pred, y.to(device))

            val_loss += loss.item()
    return val_loss/len(val_loader)
    

def main():
    args = parser.parse_args()
    
    dataset = SyntheticCellDataset(arg.img_dir, arg.mask_dir)
    
    indices = torch.randperm(len(dataset)).tolist()
    sr = int(args.split_ratio * len(dataset))
    train_set = torch.utils.data.Subset(dataset, indices[:-sr])
    val_set = torch.utils.data.Subset(dataset, indices[-sr:])
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    
    device = torch.device("cpu" if not args.use_cuda else "cuda:0")
    
    model = UNet()
    model.to(device)
    
    dsc_loss = DiceLoss()
    
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    
    val_overall = 1000
    for epoch in args.N_epoch:
        model, train_loss, optimizer = train(model, train_loader, device, optimizer)
        val_loss = validate(model, val_loader, device)
        
        if val_loss < val_overall:
            save_checkpoint(args.model_save_dir + '/epoch_'+str(epoch+1), model, train_loss, val_loss, epoch)
            val_overall = val_loss
            
        print('[{}/{}] train loss :{} val loss : {}'.format(epoch+1, num_epoch, train_loss, val_loss))
    print('Training completed)
    
if __name__ == '__main__':
    main()