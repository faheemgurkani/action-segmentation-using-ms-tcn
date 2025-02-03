import os
import argparse
from tqdm import tqdm
import numpy as np
import cv2
import matplotlib.pyplot as plt

def get_arguments() -> argparse.Namespace:
    """
    Parse all the arguments from the command line interface.
    Return a list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Convert pred and gt list to images.")
    parser.add_argument(
        "frames_dir",  # Changed to "frames_dir" from "videos_dir"
        type=str,
        help="Path to the directory containing folders of video frames",
    )
    parser.add_argument(
        "labels_path",
        type=str,
        help="Path to dataset labels",
    )
    parser.add_argument(
        "mapping_txt_path",
        type=str,
        help="Path to mapping labels",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Path to output images",
        default="output"
    )
    return parser.parse_args()

def load_action_dict(label_path):
    """
    Load the action labels into a dictionary (id to class and vice versa).
    """
    with open(label_path, "r", encoding='utf-8') as f:
        actions = f.read().split("\n")[:-1]

    id2class_map = dict()
    class2id_map = dict()
    for a in actions:
        id2class_map[int(a.split(" ")[0])] = a.split(" ")[1]
        class2id_map[a.split()[1]] = int(a.split()[0])

    return id2class_map, class2id_map

def parse_frame_directories(frames_dir):
    """
    Get a list of frame directory names from the given parent directory.
    Each directory represents a set of frames for one video.
    """
    frame_directories = [f for f in os.listdir(frames_dir) if os.path.isdir(os.path.join(frames_dir, f))]
    return frame_directories

def main() -> None:
    args = get_arguments()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Load the action dictionaries from the label file
    id2class_map, class2id_map = load_action_dict(args.labels_path)

    # Parse frame directories in the provided parent directory
    frame_dirs = parse_frame_directories(args.frames_dir)
    print(f"Found {len(frame_dirs)} frame directories in {args.frames_dir}.")

    # Loop over the frame directories and process them
    for frame_dir in tqdm(frame_dirs, desc="Processing frame directories"):
        frame_dir_path = os.path.join(args.frames_dir, frame_dir)
        video_name = frame_dir

        # Load the corresponding mapping file for actions for the current set of frames
        mapping_file = os.path.join(args.mapping_txt_path, f"{video_name}.txt")
        if not os.path.exists(mapping_file):
            print(f"Mapping file for {video_name} not found. Skipping...")
            continue

        with open(mapping_file, 'r') as f:
            mappings = f.readlines()

        # Load all frame files in the directory and sort them
        frame_files = [f for f in os.listdir(frame_dir_path) if f.endswith(('.jpg', '.png'))]
        frame_files.sort()  # Sorting ensures the frames are processed in the correct order

        # Process each frame or frame range, extract actions, and output images
        for frame_mapping in mappings:
            start_frame, end_frame, action_label = frame_mapping.strip().split(',')
            start_frame, end_frame = int(start_frame), int(end_frame)
            action_label = action_label.strip()

            # Convert action label to its corresponding ID
            action_id = class2id_map.get(action_label, -1)  # -1 if not found

            # Process each frame in the specified range
            for frame_idx in range(start_frame, end_frame + 1):
                if frame_idx < len(frame_files):
                    frame_file = frame_files[frame_idx]
                    frame_path = os.path.join(frame_dir_path, frame_file)

                    # Load the frame
                    frame = cv2.imread(frame_path)
                    if frame is None:
                        print(f"Failed to read {frame_path}. Skipping...")
                        continue

                    # Overlay action label onto the frame (optional: customize the text overlay)
                    overlay_text = f"Action: {action_label}"
                    cv2.putText(frame, overlay_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # Save the frame with the overlaid action label
                    output_image_path = os.path.join(args.output_dir, f"{video_name}_{frame_idx}_{action_label}.png")
                    cv2.imwrite(output_image_path, frame)

                    print(f"Generated image for {video_name}, frame {frame_idx} with action {action_label}.")

if __name__ == "__main__":
    main()