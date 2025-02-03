import os
import numpy as np
from natsort import natsorted  # For correct frame ordering
from tqdm import tqdm



# Defining paths
frame_features_root = "./data/gtea/extracted_frame_features/"
output_features_root = "./data/gtea/merged_extracted_frame_features/"

# Ensuring that the output directory exists
os.makedirs(output_features_root, exist_ok=True)

# Iterating through each video sequence directory
for video_seq in tqdm(os.listdir(frame_features_root)):
    video_path = os.path.join(frame_features_root, video_seq)

    if os.path.isdir(video_path):  # Checking if it is a directory
        frame_files = natsorted([f for f in os.listdir(video_path) if f.endswith(".npy")])
        
        all_frames = []

        for frame_file in frame_files:
            frame_path = os.path.join(video_path, frame_file)
            frame_features = np.load(frame_path)  # Loading frame feature
            all_frames.append(frame_features)  # Appending frame features (shape: (2048,))
        
        # Stacking features along time dimension (axis=1) → Shape will be (2048, T)
        video_features = np.stack(all_frames, axis=1)  # (2048, T)

        # Saving merged features
        output_path = os.path.join(output_features_root, f"{video_seq}.npy")
        
        np.save(output_path, video_features)
        
print("\nVideo frame features have been merged and saved successfully!")
