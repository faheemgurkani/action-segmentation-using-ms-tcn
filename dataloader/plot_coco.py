import os
import json
import cv2
import numpy as np
from tqdm import tqdm

def draw_coco_annotations(image, annotation, categories):
    """
    Draw bounding boxes, keypoints, and other metadata from a COCO annotation on an image.
    """
    bbox = annotation["bbox"]
    keypoints = annotation["keypoints"]
    category_id = annotation["category_id"]
    mode = annotation["mode"]
    action = annotation["action"]
    if bbox: 
        # Draw bounding box (x, y, width, height)
        x, y, w, h = map(int, bbox)
        color = (0, 255, 0) if mode == "right" else (0, 0, 255)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

    # Draw keypoints
    for i in range(0, len(keypoints), 3):
        kx, ky, visibility = keypoints[i:i+3]
        if visibility > 0:  # Only draw visible keypoints
            cv2.circle(image, (int(kx), int(ky)), 5, (255, 0, 0), -1)

    # Add category and action text
    category_name = next(cat["name"] for cat in categories if cat["id"] == category_id)
    if bbox: 
        cv2.putText(image, f"{category_name} ({mode})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.putText(image, f"Action: {action}", (20,  20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return image

def visualize_coco_annotations(json_file, image_dir, output_video):
    """
    Visualize COCO annotations and save them as a video.
    """
    # Load COCO JSON
    with open(json_file, 'r') as f:
        coco_data = json.load(f)

    annotations = coco_data["annotations"]
    images = {img["id"]: img for img in coco_data["images"]}
    categories = coco_data["categories"]

    # Video writer setup

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_video, fourcc, 30, (720, 405))

    print("Processing images and generating video...")
    for img_id, img_data in tqdm(images.items()):
        img_path = os.path.join(img_data["file_name"])
        image = cv2.imread(img_path)

        if image is None:
            print(f"Image {img_path} not found!")
            continue

        # Get annotations for the current image
        img_annotations = [anno for anno in annotations if anno["image_id"] == img_id]
        # Draw annotations on the image
        for annotation in img_annotations:
            image = draw_coco_annotations(image, annotation, categories)

        cv2.imwrite(f'plotted/{img_id}.png', image)
        video_writer.write(image)

    video_writer.release()
    print(f"Video saved at {output_video}")

if __name__ == "__main__":
    # Input paths
    coco_json_path = "/home/visionrd/testing/AI-Hackathon24/data/merged.json"  # Replace with your COCO JSON path
    image_dir = "frames"  # Replace with your images directory path
    output_video_path = "vis_coco.mp4"  # Replace with your desired output video path

    # Run visualization
    visualize_coco_annotations(coco_json_path, image_dir, output_video_path)
