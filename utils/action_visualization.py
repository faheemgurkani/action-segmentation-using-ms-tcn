import os
import cv2
import argparse

def overlay_action_labels(data_dir, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    frame_dir = os.path.join(data_dir, 'FRAMES')
    label_dir = os.path.join(data_dir, 'actions')

    subfolders = os.listdir(frame_dir)

    for subfolder in subfolders:

        curr_folder = os.path.join(frame_dir, subfolder)
        curr_frames = os.listdir(curr_folder)
        curr_frames.sort()

        labels = open(os.path.join(label_dir, subfolder + '.txt'), 'r').readlines()

        output_subfolder = os.path.join(output_dir, subfolder)
        os.makedirs(output_subfolder, exist_ok=True)

        print(f"Processing {curr_folder} ")
        for frame_idx, frame in enumerate(curr_frames): 

            frame_path = os.path.join(curr_folder, frame)
            frame_img = cv2.imread(frame_path)
            frame_label = labels[frame_idx].strip()

            cv2.putText(frame_img, frame_label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imwrite(os.path.join(output_subfolder, frame), frame_img)
       
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Overlay action labels on frames")
    parser.add_argument("--data_dir", default='data', help="Directory containing frames (images)")
    parser.add_argument("--output_dir", default='data/visualized_actions', help="Directory to save the processed frames")
    args = parser.parse_args()
    overlay_action_labels(args.data_dir, args.output_dir)
