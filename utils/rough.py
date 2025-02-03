import numpy as np



file_path = "./data/gtea/extracted_frame_features/S1_Cheese_C1/frame_000002.npy"
data = np.load(file_path)

print("Shape of the loaded data:", data.shape)
print("Data preview:", data)
