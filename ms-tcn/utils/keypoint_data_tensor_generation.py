import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import torch



# Defining the expected labels and the number of keypoint pairs you expect for each.
# For instance, for "hand" you might expect 1 pair (2 numbers), 
# and for a finger you might expect 5 pairs (10 numbers) or 4 pairs (8 numbers) depending on your annotation.
EXPECTED_LABELS = {
    "hand": 1,            # 1 pair: 2 numbers
    "thumb": 5,           # 5 pairs: 10 numbers
    "index_finger": 4,    # 4 pairs: 8 numbers
    "middle_finger": 4,   # 4 pairs: 8 numbers
    "ring_finger": 4,     # 4 pairs: 8 numbers
    "pinkie_finger": 4    # 4 pairs: 8 numbers
}

def parse_points(points_str, expected_pairs):
    """
    Parse a coordinate string into a fixed-length list of floats.
    The string is expected to contain coordinate pairs separated by ';', 
    with each pair in the format "x,y".
    
    If there are fewer than expected_pairs, the result is padded with zeros.
    If there are more, the result is truncated.
    """
    if pd.isna(points_str) or points_str.strip() == "":
        return [0.0] * (expected_pairs * 2)
    
    # Checking if the string even contains a comma. If not, treat it as a single value.
    if ',' not in points_str:
    
        try:
            value = float(points_str)
    
            return [value, value] if expected_pairs == 1 else [value] * (expected_pairs * 2)
        except:
            return [0.0] * (expected_pairs * 2)
    
    pairs = points_str.split(';')
    coords = []
    
    for pair in pairs:
        pair = pair.strip()
    
        if not pair:
            continue
    
        parts = pair.split(',')
    
        if len(parts) >= 2:
    
            try:
                x = float(parts[0].strip())
                y = float(parts[1].strip())
            except:
                x, y = 0.0, 0.0
    
            coords.extend([x, y])
    
    required_length = expected_pairs * 2
    
    # Padding or truncate the list to have the required length.
    if len(coords) < required_length:
        coords.extend([0.0] * (required_length - len(coords)))
    elif len(coords) > required_length:
        coords = coords[:required_length]
    
    return coords

def load_keypoints_from_csv(csv_path, frame_names, expected_labels=EXPECTED_LABELS):
    """
    For each frame in frame_names, group rows by frame name and extract features 
    for each expected label in a predefined order.
    
    The final feature vector for a frame is the concatenation of the parsed features 
    for each expected label.
    
    Returns:
      A NumPy array of shape (total_feature_dim, num_frames)
    """
    df = pd.read_csv(csv_path)
    
    # Ensuring frames are processed in a consistent order.
    frame_names = sorted(frame_names)
    all_frame_features = []
    
    for frame in frame_names:
        # Getting all rows corresponding to the current frame.
        frame_df = df[df['Frame Name'] == frame]
        frame_features = []
        
        # Processing each expected label in a fixed order.
        for label, expected_pairs in expected_labels.items():
            # Filtering rows for the current label.
            label_df = frame_df[frame_df['Label'] == label]
    
            if not label_df.empty:

                if label == "hand":
                    row = label_df.iloc[0]
                    points_str = f"{row['XTL/Points']}, {row['YTL/Points']}"  # Using both the XTL and YTL/Points column for coordinates of hand.
                    parsed = parse_points(points_str, expected_pairs)

                else:
                    row = label_df.iloc[0]
                    points_str = row['XTL/Points']  # We use the XTL/Points column for coordinates.

                    # print(points_str)  # For, testing

                    parsed = parse_points(points_str, expected_pairs)

                # print(parsed)   # For, testing
            else:
                parsed = [0.0] * (expected_pairs * 2)
    
            frame_features.extend(parsed)
    
        all_frame_features.append(frame_features)
    
    # Converting to a NumPy array of shape (num_frames, total_feature_dim)
    features_array = np.array(all_frame_features)
    
    # Transposing to get shape (total_feature_dim, num_frames)
    features_array = features_array.T
    
    return features_array

def process_csv_file(csv_path, output_dir):
    """
    Processes a single CSV file to extract keypoint features and saves the resulting
    tensor in .npy format.
    """
    df = pd.read_csv(csv_path)
    # Get the unique frame names from the CSV (sorted for consistent ordering)
    frame_names = sorted(df['Frame Name'].unique())
    kp_features = load_keypoints_from_csv(csv_path, frame_names)
    
    # Use the base filename (without extension) for saving
    base_name = os.path.basename(csv_path)
    name_without_ext = os.path.splitext(base_name)[0]
    npy_filename = os.path.join(output_dir, name_without_ext + '.npy')
    
    # Save the NumPy array to file
    np.save(npy_filename, kp_features)
    # print(f"Saved keypoint features for {name_without_ext} to {npy_filename} with shape {kp_features.shape}")



if __name__ == "__main__":
    # input_dir = "./data/gtea/extracted_xml_data"
    # output_dir = "./data/gtea/merged_extracted_keypoint_features"
    
    # os.makedirs(output_dir, exist_ok=True)
    
    # # Processing each CSV file in the input directory.
    # for file in tqdm(os.listdir(input_dir)):
        
    #     if file.endswith('.csv'):
    #         csv_file_path = os.path.join(input_dir, file)
        
    #         process_csv_file(csv_file_path, output_dir)


    # For, testing
    data = np.load("./data/gtea/merged_extracted_keypoint_features/S1_Cheese_C1.npy", allow_pickle=True)

    # Checking the type of the loaded object
    if isinstance(data, np.ndarray):
        print(f"Number of tensors: 1")
        print(f"Shape of tensor: {data.shape}")
    elif isinstance(data, list):
        print(f"Number of tensors: {len(data)}")
        print(f"Shape of first tensor: {data[0].shape if isinstance(data[0], np.ndarray) else 'Not an ndarray'}")
    elif isinstance(data, dict):
        print(f"Keys in the dictionary: {data.keys()}")
        for key, value in data.items():
            print(f"Shape of tensor {key}: {value.shape if isinstance(value, np.ndarray) else 'Not an ndarray'}")
    else:
        print("Unknown data type")


# if __name__ == "__main__":
#     csv_path = "./data/gtea/extracted_xml_data/S1_Cheese_C1.csv" 
    
#     # Supposing these are the frames
#     frame_names = ["frame_000785", "frame_000942"]
    
#     kp_features = load_keypoints_from_csv(csv_path, frame_names)
    
#     print("Keypoint features shape:", kp_features.shape)
    
#     # Creating a pytorch tensor
#     kp_tensor = torch.tensor(kp_features, dtype=torch.float)
    
#     # Expected shape (feature_dim, sequence_length)
#     print("PyTorch tensor shape:", kp_tensor.shape)
