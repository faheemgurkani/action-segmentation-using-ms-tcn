import cv2
import xml.etree.ElementTree as ET
import os

# Paths
xml_file = "/home/multi-gpu/Factory_copilot_Osama/data/GTEA_latest/s1coffeec1.xml"  # Path to your XML file
images_dir = "/home/multi-gpu/Factory_copilot_Osama/data/GTEA_latest/s1coffeec1"         # Directory containing the images
output_video = "/home/multi-gpu/Factory_copilot_Osama/data/GTEA_latest/s1coffeec1_vis.mp4"   # Path to save the output video


def main():
    parser = argparse.ArgumentParser(description='Process a folder of images and save annotated frames.')
    parser.add_argument('--xml_file', type=str, help='Path to the folder containing PNG frames',default="data/GTEA_latest/s1coffeec1.xml")
    parser.add_argument('--images_dir', type=str, help='Path to the JSON file containing annotations',default="data/GTEA_latest/s1coffeec1"  )
    parser.add_argument('--output_video', type=str, help='Path to the folder where processed frames will be saved',default="data/GTEA_latest/s1coffeec1_vis.mp4" )
    args = parser.parse_args()

    # Process the folder and save annotated frames
    process_folder(args.input_folder, args.annotations_file, args.output_folder)

if __name__ == '__main__':
    main()


# Parse the XML file
tree = ET.parse(xml_file)
root = tree.getroot()

# Get all image entries
images = root.findall("image")

# Initialize video writer
frame_width, frame_height = 720, 404  # Assuming fixed dimensions from the XML
fps = 30
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

# Process each image
for image in images:
    image_name = image.get("name")
    image_path = os.path.join(images_dir, image_name.replace('frame_', 's1coffeec1_') + ".jpg")
    frame = cv2.imread(image_path)

    if frame is None:
        print(f"Image not found: {image_path}")
        continue

    # Draw bounding boxes and polylines
    for box in image.findall("box"):
        xtl, ytl = int(float(box.get("xtl"))), int(float(box.get("ytl")))
        xbr, ybr = int(float(box.get("xbr"))), int(float(box.get("ybr")))
        label = box.get("label")
        cv2.rectangle(frame, (xtl, ytl), (xbr, ybr), (0, 255, 0), 2)
        cv2.putText(frame, label, (xtl, ytl - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    for polyline in image.findall("polyline"):
        points = polyline.get("points").split(";")
        points = [tuple(map(int, map(float, point.split(",")))) for point in points]
        label = polyline.get("label")
        for i in range(len(points) - 1):
            cv2.line(frame, points[i], points[i + 1], (255, 0, 0), 2)
        if points:
            cv2.putText(frame, label, points[0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Write frame to video
    video_writer.write(frame)

# Release video writer
video_writer.release()
print(f"Video saved at {output_video}")
