import pickle
import numpy as np

pkl_file_path = "data.pkl"
n_channels = 125

# sampling rate 500 Hz
# T = 1/500
# data len 1651
# recording len 1615 * 1/500 = 3.23 s

with open(pkl_file_path, 'rb') as file:
    eeg_data = pickle.load(file)

if len(eeg_data) > 0:
    text = eeg_data[0]['text']
    values = np.array(eeg_data[0]['input_features'])

    values = values.reshape(n_channels, -1)  