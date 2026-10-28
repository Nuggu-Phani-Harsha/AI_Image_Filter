import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from utils.utils import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--content_dir", type=str, default='/Users/harsha/Desktop/AI_Image_Filter/NST_Code/Content', 
                        help="Path to the content directory.")
    parser.add_argument("--style_dir", type=str, default='/Users/harsha/Desktop/AI_Image_Filter/NST_Code/Style', 
                        help="Path to the style directory.")
    parser.add_argument("--vgg_model_path", type=str, default='/Users/harsha/Desktop/AI_Image_Filter/NST_Code/vgg_normalised.pth',
                        help="Path to the pre-trained VGG model.")
    parser.add_argument("--experiment_name", type=str, default='experiment_1',
                        help="Name of the experiment.")
    
    parser.add_argument("--final_size", type=int, default=512,
                        help="Size of the final output image.")
    parser.add_argument("--content_size", type=int, default=512,
                        help="Size of the content image.")
    parser.add_argument("--style_size", type=int, default=512,
                        help="Size of the style image.")
    parser.add_argument("--crop",action='store_true', default=True,
                        help="Whether to crop the images.")
    return parser.parse_args()

def main():
    args=parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(f'./experiments/{args.experiment_name}')
    save_dir.mkdir(parents=True, exist_ok=True)

    #Save the arguments to a text file for future reference
    with open(save_dir / 'args.txt', 'w') as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")

    content_transform = get_transform(args.content_size, args.crop, args.final_size)
    style_transform = get_transform(args.style_size, args.crop, args.final_size)

    content_dataset = ImageDataset(args.content_dir,content_transform)
    style_dataset = ImageDataset(args.style_dir,style_transform)

    content_dataloader = DataLoader(content_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True,drop_last=True)
    style_dataloader = DataLoader(style_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True,drop_last=True)

if __name__ == "__main__":
    main()