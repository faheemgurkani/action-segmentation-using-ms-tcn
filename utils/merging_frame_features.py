import os
import numpy as np
from natsort import natsorted  # For correct frame ordering
from tqdm import tqdm


frame_features_root = "./data/gtea/extracted_frame_features/"
output_features_root = "./data/gtea/merged_extracted_frame_features/"

os.makedirs(output_features_root, exist_ok=True)

for video_seq in tqdm(os.listdir(frame_features_root)):
    video_path = os.path.join(frame_features_root, video_seq)

    # Processing only directories (each representing a video sequence)
    if os.path.isdir(video_path):
        # Sorting frame files naturally to maintain correct temporal order
        frame_files = natsorted([f for f in os.listdir(video_path) if f.endswith(".npy")])

        # # print(frame_files)  # For, testing
        # print(len(frame_files))

        all_frames = []

        for frame_file in frame_files:
            frame_path = os.path.join(video_path, frame_file)
            frame_features = np.load(frame_path)
            
            # Optional: Verify that the feature vector has the expected shape (2048,)
            if frame_features.shape != (2048,):
                raise ValueError(f"Unexpected feature shape {frame_features.shape} in {frame_path}")
            
            all_frames.append(frame_features)

            # break  # For, testing

        # print(f"Number of frames for {video_seq}: {len(all_frames)}")   # For, testing

        # Stacking the frame features along the time dimension.
        # This creates an array of shape (2048, T)
        video_features = np.stack(all_frames, axis=1)
        
        # # For, testing
        # print(f"Merged video features shape for {video_seq}: {video_features.shape}")

        # break  # For, testing

        output_path = os.path.join(output_features_root, f"{video_seq}.npy")
        np.save(output_path, video_features)

print("\nVideo frame features have been merged and saved successfully!")
