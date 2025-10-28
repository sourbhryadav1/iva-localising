import scipy.io as sio
import os

# Use the exact path to one of the files that caused an error
file_path = '/home/gabbru/Desktop/padhy sir/334/left/0911986.mat' 

if not os.path.exists(file_path):
    print(f"Error: File not found at {file_path}")
    print("Please make sure this path is correct.")
else:
    try:
        mat_data = sio.loadmat(file_path)
        
        print(f"--- Inspecting File: {file_path} ---")
        print("Found the following variables (keys) and their shapes:")
        
        for key, value in mat_data.items():
            # Ignore the internal metadata keys
            if key.startswith('__'):
                continue
            
            # Check if it's a numpy array to get its shape
            if hasattr(value, 'shape'):
                print(f"  - Key: '{key}', Shape: {value.shape}")
            else:
                print(f"  - Key: '{key}', Type: {type(value)}")
        
        print("-----------------------------------------------")
        print("Find the key that has the shape (12, 5000). That is our new name.")

    except Exception as e:
        print(f"Error while inspecting {file_path}: {e}")