import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.utils.data


class BRAVOData(torch.utils.data.Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.patients = os.listdir(self.root_dir)
        self.meta = self._load_metadata()

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        # load binary files into numpy arrays
        input = np.fromfile(self.meta[idx][0], dtype='single')

        # reshape to multidimensional arrays
        shape = [12, 206, 178, 8]
        input = input.reshape(shape)

        input = input[np.random.choice(np.arange(shape[0]), size=2, replace=False)]

        # transform to pytorch tensors and package for output
        sample = {
            'input': torch.tensor(input), 
            'input_location': torch.tensor(self.meta[idx][1], dtype=torch.int),
            'patient_index': torch.tensor(self.meta[idx][2], dtype=torch.int)
            }
        return sample

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
    

if __name__ == '__main__':
    data = BRAVOData('train')
    x = data[0]
    fig, ax = plt.subplots(1,2)
    ax[0].imshow(torch.rot90(x['input'][0,:,:,0]), cmap='gray') # data is multicoil, so display the first (compressed) coil
    ax[1].imshow(torch.rot90(x['input'][1,:,:,0]), cmap='gray') 
    plt.show()
