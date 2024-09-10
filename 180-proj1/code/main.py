import os
import argparse
from utils import process_image
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--crop', action='store_true')
    parser.add_argument('--contrast', action='store_true')
    parser.add_argument('--white_balance', action='store_true')
    args = parser.parse_args()

    data_dir = './data'
    files = [filename for filename in os.listdir(data_dir) if filename.endswith('.jpg') or filename.endswith('.tif')]
    
    for filename in tqdm(files, desc="Processing images"):
        print(f"Currently working on: {filename}")
        
        if filename.endswith('.jpg'):
            process_image(os.path.join(data_dir, filename), 'jpg', crop=args.crop, contrast=args.contrast, white_balance=args.white_balance)
        elif filename.endswith('.tif'):
            process_image(os.path.join(data_dir, filename), 'tif', crop=args.crop, contrast=args.contrast, white_balance=args.white_balance)

if __name__ == "__main__":
    main()