import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from torch.utils.data import Dataset
import random

import torchvision.transforms as transforms
import torchvision.transforms.functional as transforms_f


class BRAVOData(Dataset):
    def __init__(self, root_dir, transforms_x = None):
        self.root_dir = root_dir
        self.patients = os.listdir(self.root_dir)
        self.meta = self._load_metadata()
        
        self.transforms_x = transforms_x
        

    # load metadata for this class instance
    def _load_metadata(self):
        # iterate over datasets, processing inputs and targets jointly
        metadata = []
        for elem in os.walk(self.root_dir):
            if 'inputs_pretraining' in elem[0]:
                # parse subdirectory name to extract patch dimensions
                input_dir = elem[0]

                pat = input_dir.split('/')[-2]
                pat_idx = self.patients.index(pat)

                for input_filename in elem[2]:
                # define full filenames
                    input_fullname = os.path.join(input_dir, input_filename)

                    # parse input and target filenames to extract patch locations
                    input_loc = input_filename.split('-')[1].split('.')[0]

                    metadata.append((input_fullname, int(input_loc), pat_idx))
        return metadata
    
    def __getitem__(self, idx):
        # load binary files into numpy arrays
        inputx = np.fromfile(self.meta[idx][0], dtype='single')

        # reshape to multidimensional arrays
        shape = [12, 206, 178, 8]
        inputx = inputx.reshape(shape)

        # Picking a random number between 1 and 12
        random_number = random.randint(1, 11)
    
        img_clean, img_corrupt = torch.from_numpy(inputx[0]), torch.from_numpy(inputx[random_number])
        
        if self.transforms_x is not None:
            img_clean, img_corrupt = self.transforms_x(img_clean), self.transforms_x(img_corrupt)
            
            if torch.rand(1) > 0.5:
                img_clean = transforms_f.hflip(img_clean)
                img_corrupt = transforms_f.hflip(img_corrupt)
                
            if torch.rand(1) > 0.5:
                img_clean = transforms_f.vflip(img_clean)
                img_corrupt = transforms_f.vflip(img_corrupt)
        
        return img_clean, img_corrupt

    def __len__(self):
        return len(self.meta)
    
# mean = [3.0892, 1.6437, 1.3494, 0.9318, 1.0241, 0.6993, 0.7459, 0.7565]
# std = [4.4823, 2.6962, 2.1895, 1.4849, 1.6206, 1.1844, 1.1394, 1.1681]

if __name__ == '__main__':
    
    # simple augmentation
    transform_try = transforms.Compose([
            transforms.Lambda(lambda x: x.permute(2, 0, 1)),  # Move to (channels, height, width)
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[3.0892, 1.6437, 1.3494, 0.9318, 1.0241, 0.6993, 0.7459, 0.7565],
                                 std=[4.4823, 2.6962, 2.1895, 1.4849, 1.6206, 1.1844, 1.1394, 1.1681])
            ])
    
    dataset_train = BRAVOData('dataset/train', transforms_x = transform_try)
    
    # imagex_clean, imagex_corrupt = dataset_train[2]
    
    # plt.imshow(imagex_clean[0,:,:])
    # plt.pause(0.1)
    # plt.imshow(imagex_corrupt[0,:,:])
    
    ###%
    
    from timm.data.loader import MultiEpochsDataLoader
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    dataloader_cls = MultiEpochsDataLoader

    data_loader_train = dataloader_cls(
        dataset_train, sampler=sampler_train,
        batch_size=2,
        num_workers=0,
        drop_last=True,
    )
    
    samples = next(iter(data_loader_train))
    samples = torch.cat((samples[0], samples[1]), dim=0) 