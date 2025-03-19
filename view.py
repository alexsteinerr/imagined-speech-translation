import pickle
import torch

def load_eeg_data(file_path):
    """Loads EEG data from a PKL file and checks its structure."""
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    if isinstance(data, dict):
        print("Loaded data is a dictionary. Keys:", data.keys())
        print("Dictionary contents:", data)  
        return data
    elif isinstance(data, list):
        print("Loaded data is a list. Checking contents...")
        
        if all(isinstance(item, (list, tuple)) for item in data):
            try:
                tensor_data = torch.tensor(data)
                print("Successfully converted list to tensor.")
                return tensor_data
            except Exception as e:
                print("Error converting list to tensor:", e)
        elif all(isinstance(item, dict) for item in data):
            print("List contains dictionaries. Returning as is.")
            print("List of dictionaries contents:", data) 
            return data
        else:
            print("List contains mixed or unsupported data types.")
            return data
    elif isinstance(data, torch.Tensor):
        print("Loaded data is already a tensor.")
        return data
    else:
        raise TypeError("Unsupported data format in PKL file.")

# Example Usage
if __name__ == "__main__":
    file_path = "ds005170-1.1.2/derivatives/preprocessed_pkl/sub-01/eeg/sub-01_task-imagine_run-01_eeg.pkl" 
    eeg_data = load_eeg_data(file_path)
    
    if isinstance(eeg_data, dict):
        for key, value in eeg_data.items():
            try:
                print(f"Key: {key}, Shape: {torch.tensor(value).shape}")
            except Exception as e:
                print(f"Key: {key}, Could not convert to tensor: {e}")
