import torch
import torch.nn as nn
from math import floor
import os
import random
import numpy as np
import pdb
import time
from datasets.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
from torch.utils.data import DataLoader
from torchvision import transforms, utils, models
from models.resnet_custom import resnet50_baseline
import argparse
from file_utils import save_hdf5
from PIL import Image
import h5py
import openslide
from tqdm import tqdm

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def collate_features(batch):
	img = torch.cat([item[0] for item in batch], dim = 0)
	coords = np.vstack([item[1] for item in batch])
	return [img, coords]



def compute_w_loader(file_path, output_path, wsi, model,
                     batch_size=8, verbose=0, print_every=20, pretrained=True,
                     custom_downsample=1, target_patch_size=-1):
    """
    args:
        file_path: directory of bag (.h5 file)
        output_path: directory to save computed features (.h5 file)
        model: pytorch model
        batch_size: batch_size for computing features in batches
        verbose: level of feedback
        pretrained: use weights pretrained on imagenet
        custom_downsample: custom defined downscale factor of image patches
        target_patch_size: custom defined, rescaled image size before embedding
    """
    dataset = Whole_Slide_Bag_FP(file_path=file_path, wsi=wsi, pretrained=pretrained,
                                 custom_downsample=custom_downsample, target_patch_size=target_patch_size)
    x, y = dataset[0]
    kwargs = {'num_workers': 4, 'pin_memory': True} if device.type == "cuda" else {}
    loader = DataLoader(dataset=dataset, batch_size=batch_size, **kwargs, collate_fn=collate_features)

    mode = 'w'
    for count, (batch, coords) in enumerate(loader):
        with torch.no_grad():
            batch = batch.to(device, non_blocking=True)

            features = model(batch)
            features = features.view(features.size(0), -1)
            features = features.cpu().numpy()

            asset_dict = {'features': features, 'coords': coords}
            save_hdf5(output_path, asset_dict, attr_dict=None, mode=mode)
            mode = 'a'

    return output_path


parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--slide_ext', type=str, default='.svs',choices=['.svs', '.tif'])
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--no_auto_skip', default=False, action='store_true')
parser.add_argument('--ssl_epoch', type=int, default=200)
parser.add_argument('--custom_downsample', type=int, default=1)
parser.add_argument('--target_patch_size', type=int, default=512)
parser.add_argument('--target_mag', type=int, default=20)
parser.add_argument('--data', type=str, default='SCH')
args = parser.parse_args()

             
args.data_slide_dir=f'/mnt/disk6/zhaolu/OS/server6/data/TCGA-{args.data}/svs'
args.csv_path=f'/mnt/disk6/zhaolu/OS/server6/data/TCGA-{args.data}/slide_id.csv'
args.data_h5_dir=f'/mnt/disk6/zhaolu/OS/libs_TCGA/libs_{args.data}/{args.target_mag}X_{args.target_patch_size}'
args.feat_dir=args.data_h5_dir+'/features_ResNet50'


def print_options(args, parser,print_info=False):
    """Print and save options
    It will print both current options and default values(if different).
    It will save options into a text file / [checkpoints_dir] / opt.txt
    """
    message = ''
    message += '----------------- Options ---------------\n'
    time_now = time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime(time.time()))
    message += '----------------- ' + time_now + '---------------\n'
    
    for k, v in vars(args).items():
        comment = ''
        # default = parser.get_default(k)
        # if v != default:
        #     comment = '\t[default: %s]' % str(default)
        message += '{:>5}: {:<50}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    if print_info:
        print(message)

    file_name = os.path.join(args.feat_dir, 'options.txt')
    with open(file_name, 'wt') as opt_file:
        opt_file.write(message)
        opt_file.write('\n')


if __name__ == '__main__': 
    csv_path = args.csv_path
    if csv_path is None:
        raise NotImplementedError

    model = resnet50_baseline(pretrained=True)
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.eval()

    bags_dataset = Dataset_All_Bags(csv_path)
    os.makedirs(os.path.join(args.feat_dir, 'pt_files'), exist_ok=True)
    os.makedirs(os.path.join(args.feat_dir, 'h5_files'), exist_ok=True)
    dest_files = os.listdir(os.path.join(args.feat_dir, 'pt_files'))
    
    print_options(args, parser,print_info=True)

    for bag_candidate_idx in range(len(bags_dataset)):
        print(f'computing {bag_candidate_idx}/{len(bags_dataset)}')
        slide_id = bags_dataset[bag_candidate_idx].split(args.slide_ext)[0]
        bag_name = slide_id + '.h5'
        h5_file_path = os.path.join(args.data_h5_dir, 'patches', bag_name)
        
        slide_file_path = os.path.join(args.data_slide_dir, slide_id + args.slide_ext)

        if not args.no_auto_skip and slide_id + '.pt' in dest_files:
            print('skipped {}'.format(slide_id))
            continue
        
        try:
            os.path.exists(h5_file_path)
        except:
            print(f'{h5_file_path} is broken')
            
        try:
            output_path = os.path.join(args.feat_dir, 'h5_files', bag_name)
            time_start = time.time()
            wsi = openslide.open_slide(slide_file_path)
            output_file_path = compute_w_loader(h5_file_path, output_path, wsi,
                                                model=model, batch_size=args.batch_size, verbose=0, print_every=50,
                                                custom_downsample=args.custom_downsample, target_patch_size=args.target_patch_size)
            time_elapsed = time.time() - time_start
            file = h5py.File(output_file_path, "r")
            features = file['features'][:]
            print('features size:', features.shape)
            features = torch.from_numpy(features)
            bag_base, _ = os.path.splitext(bag_name)
            torch.save(features, os.path.join(args.feat_dir, 'pt_files', bag_base + '.pt'))
            
        except openslide.lowlevel.OpenSlideError:
            print(f'{slide_file_path} is broken, encountered openslide.lowlevel.OpenSlideError')
            continue
