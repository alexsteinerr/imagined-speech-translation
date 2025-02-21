import glob
import torch
import pickle
import numpy as np
from torch.utils.data import Dataset

# use stokes theoreom to further process the data in a 3d environment 
# there is a voltage potential difference between the signal measured and 
# the emitted one due to the impedance can I if I have the measured one 
# and the impedance calculate the actual one
#
#

class Dataset(Dataset):
    def __init__(self, data_dir):
        self.n_channels = 125
        self.paths = glob.glob(data_dir + '/*.pkl')

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        with open(self.paths[idx], 'rb') as file:
            eeg_data = pickle.load(file)

        text = eeg_data[0]['text']

        values = np.array(eeg_data[0]['input_features'])
        values = values.reshape(self.n_channels, -1)
        values = torch.from_numpy(values)  

        return values, text