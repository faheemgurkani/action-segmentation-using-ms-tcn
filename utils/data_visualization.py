import cv2
import numpy as np
import json
import argparse
import os

FINGER_CONNECTIONS = {
    'thumb': [(0, 1), (1, 2), (2, 3), (3, 4)],
    'index': [(5, 6), (6, 7), (7, 8)],
    'middle': [(9, 10), (10, 11), (11, 12)],
    'ring': [(13, 14), (14, 15), (15, 16)],
    'pinky': [(17, 18), (18, 19), (19, 20)]
}

def visualize_annotations(image, anns, visualize_bboxes=True, visualize_keypoints=True):

  

    if visualize_bboxes:
        for ann in anns:
            bbox = ann.get('bbox', None)
            if bbox:
                x, y, w, h = map(int, bbox)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if visualize_keypoints:
        for ann in anns:
            keypoints = ann.get('keypoints', None)
            if keypoints:
                for i in range(0, len(keypoints), 3):
                    x, y, visibility = keypoints[i], keypoints[i+1], keypoints[i+2]
                    if visibility > 0:
                        cv2.circle(image, (int(x), int(y)), 5, (0, 0, 255), -1)

                for finger, connections in FINGER_CONNECTIONS.items():
                    for (start, end) in connections:
                        start_x, start_y = int(keypoints[start * 3]), int(keypoints[start * 3 + 1])
                        end_x, end_y = int(keypoints[end * 3]), int(keypoints[end * 3 + 1])

                        # Draw line between connected keypoints
                        cv2.line(image, (start_x, start_y), (end_x, end_y), (255, 0, 0), 2)

    return image

def process_folder(input_folder, annotations_file, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load annotations
    with open(annotations_file, 'r') as f:
        annotations_data = json.load(f)

    # Iterate over all PNG files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".png"):
            # Read the image
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)

            if image is None:
                print(f"Error: Could not load image {filename}")
                continue
            image_id = int(filename.split('.')[0].split('_')[-1])
            annotations = [annos for annos in annotations_data['annotations'] if annos['image_id']==image_id]
            action = annotations[0].get('action', 'dummy')
            cv2.putText(image, action, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Visualize annotations on the image
            annotated_image = visualize_annotations(image, annotations)

            # Save the annotated image in the output folder
            output_image_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_image_path, annotated_image)
            print(f"Processed {filename} and saved to {output_image_path}")

def main():
    parser = argparse.ArgumentParser(description='Process a folder of images and save annotated frames.')
    parser.add_argument('--input_folder', type=str, help='Path to the folder containing PNG frames',default="data")
    parser.add_argument('--annotations_file', type=str, help='Path to the JSON file containing annotations',default="data")
    parser.add_argument('--output_folder', type=str, help='Path to the folder where processed frames will be saved',default="output_folder")
    args = parser.parse_args()

    # Process the folder and save annotated frames
    process_folder(args.input_folder, args.annotations_file, args.output_folder)

if __name__ == '__main__':
    main()
