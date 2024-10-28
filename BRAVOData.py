import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from torch.utils.data import Dataset

class BRAVOData(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.patients = os.listdir(self.root_dir)
        self.meta = self._load_metadata()

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

        inputx = inputx[0]

        return inputx

    def __len__(self):
        return len(self.meta)
    

if __name__ == '__main__':
    data = BRAVOData('dataset/train')
    
    imagex = data[0]
    
    plt.imshow(imagex[:,:,0])