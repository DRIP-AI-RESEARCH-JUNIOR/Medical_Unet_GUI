import torch 
from model import UNet
from utils import load_checkpoint
from PIL import Image

import torchvision.transforms.functional as TF
from torchvision import transforms

import argparse

parser = argparse.ArgumentParser(
    prog="infer.py", description="UNet Inference"
)

parser.add_argument(
    'Image_file',
    help="Image to test", type=str
)
parser.add_argument(
    'weight',
    help="Weight of the UNet model.", type=str
)
parser.add_argument(
    '--use_cuda',
    action='store',
    default=True,
    help="Try to use CUDA. The default value is no. All default values are for CPU mode.",
    type=bool, dest='use_cuda'
)

def main():
    args = parser.parse_args()
    device = torch.device("cpu" if not args.use_cuda else "cuda:0")
    
    model = UNet()
    load_checkpoint(args.weight, model, device='cpu')
    model.to(device)
    
    img = Image.open(img_file)

    resize = transforms.Resize(size=(576, 576))
    im_r = TF.to_tensor(resize(img))
    im_r = im_r.unsqueeze(0)
    
    with torch.no_grad():
        pred = model(im_r.to(device))
        
    pred_mask = pred.detach().cpu().numpy().squeeze()
    
if __name__ == '__main__':
    main()