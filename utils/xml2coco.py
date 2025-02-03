import xml.etree.ElementTree as ET
import json
from tqdm import tqdm
import os
unique_img_ids = 0 

class Python_Switch:
    def finger(self, label,keypoints,right_or_left,filling):

        return getattr(self, '' + str(label))(keypoints,right_or_left,filling)

    def thumb(self,keypoints,right_or_left,filling):
        start_indx=0 
        for point in filling:
            # import pdb; pdb.set_trace()
            if point is not None:
                # try:
                keypoints[right_or_left][start_indx*3] = float(point.split(',')[0])
                keypoints[right_or_left][(start_indx*3)+1] = float(point.split(',')[1])
                keypoints[right_or_left][(start_indx*3)+2] = 2
                # except:
                #     import pdb; pdb.set_trace()

            start_indx+=1
        return keypoints
    def index_finger(self,keypoints,right_or_left,filling):
        start_indx=5 
        for point in filling:
            if point is not None:
                keypoints[right_or_left][start_indx*3] = float(point.split(',')[0])
                keypoints[right_or_left][(start_indx*3)+1] = float(point.split(',')[1])
                keypoints[right_or_left][(start_indx*3)+2] = 2
            start_indx+=1
        return keypoints
    def middle_finger(self,keypoints,right_or_left,filling):
        start_indx = 9
        for point in filling:
            if point is not None:
                keypoints[right_or_left][start_indx*3] = float(point.split(',')[0])
                keypoints[right_or_left][(start_indx*3)+1] = float(point.split(',')[1])
                keypoints[right_or_left][(start_indx*3)+2] = 2
            start_indx+=1
        return keypoints
    def ring_finger(self,keypoints,right_or_left,filling):
        start_indx = 13
        for point in filling:
            if point is not None:
                keypoints[right_or_left][start_indx*3] = float(point.split(',')[0])
                keypoints[right_or_left][(start_indx*3)+1] = float(point.split(',')[1])
                keypoints[right_or_left][(start_indx*3)+2] = 2
            start_indx+=1
        return keypoints
    def pinkie_finger(self,keypoints,right_or_left,filling):
        start_indx = 17
        for point in filling:
            if point is not None:
                keypoints[right_or_left][start_indx*3] = float(point.split(',')[0])
                keypoints[right_or_left][(start_indx*3)+1] = float(point.split(',')[1])
                keypoints[right_or_left][(start_indx*3)+2] = 2
            start_indx+=1
        return keypoints

my_switch = Python_Switch()
def process_xml_to_coco(xml_file, actions_file, coco_format, unique_id):
    global unique_img_ids
    tree = ET.parse(xml_file)
    vid_name = xml_file.split('/')[-1].split('.')[0]
    with open(actions_file, 'r') as f:
        actions = f.readlines()
    root = tree.getroot()
    for image in tqdm(root.findall('image'), desc=f"Processing {xml_file}"):
        image_id = int(image.get('id'))
        if image_id < len(actions):
            print(xml_file)
            frame_dir = "/".join(xml_file.split('/')[4:6])+'/FRAMES/'+xml_file.split('/')[-1].replace('.xml','')
            image_entry = {
                "id": unique_img_ids,
                "file_name": os.path.join(frame_dir, str(int(image.get('name').replace('frame_', vid_name+'/'+vid_name+'_')))+'.png'),
                "width": int(image.get('width')),
                "height": int(image.get('height'))
            }
            coco_format['images'].append(image_entry)

            bounding_boxes = {"right": [], "left": []}
            keypoints = {"right": [0] * 63, "left": [0] * 63}

            # Process bounding boxes
            for box in image.findall('box'):
                label = box.get('label')
                xtl = float(box.get('xtl'))
                ytl = float(box.get('ytl'))
                xbr = float(box.get('xbr')) - xtl
                ybr = float(box.get('ybr')) - ytl  # Convert to xywh
                attribute_value = box.find(".//attribute").text
                bounding_boxes[attribute_value] = [xtl, ytl, xbr, ybr]

            # Process keypoints
            for polyline in image.findall('polyline'):
                label = polyline.get('label')
                points = polyline.get('points').split(';')
                attribute_value = polyline.find(".//attribute").text
                try:
                    keypoints = my_switch.finger(label, keypoints, attribute_value, points)
                except: 
                    import pdb; pdb.set_trace()
                    print(f"Could not match annos for image {image_entry}, skipping")
                    continue 
            # Create annotations for right and left hands
            for mode in ["right", "left"]:
                annotation = {
                    "id": unique_id,
                    "image_id": unique_img_ids,
                    "bbox": bounding_boxes[mode],
                    "keypoints": keypoints[mode],
                    "mode": mode,
                    "category_id": 1,
                    "area": bounding_boxes[mode][2] * bounding_boxes[mode][3] if bounding_boxes[mode] else 0,
                    "iscrowd": 0,
                    "num_keypoints": sum(1 for i in range(2, len(keypoints[mode]), 3) if keypoints[mode][i] == 2),
                    "action": actions[image_id].split('\t')[-1].strip(),
                }
                coco_format['annotations'].append(annotation)
                unique_id += 1
            unique_img_ids += 1
            
    return unique_id

def main(xml_folder, actions_folder, output_file):
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "hand", "supercategory": "object", "keypoints": [], "skeleton": []}
        ]
    }

    unique_id = 0
    for xml_file in os.listdir(xml_folder):
        if xml_file.endswith('.xml'):
            actions_file = os.path.join(actions_folder, xml_file.replace('.xml', '.txt'))
            xml_path = os.path.join(xml_folder, xml_file)
            unique_id = process_xml_to_coco(xml_path, actions_file, coco_format, unique_id)

    with open(output_file, 'w') as f:
        json.dump(coco_format, f, indent=4)

if __name__ == "__main__":
    xml_folder = "/home/visionrd/testing/AI-Hackathon24/data/xmls"
    actions_folder = "/home/visionrd/testing/AI-Hackathon24/data/groundTruth"
    output_file = "/home/visionrd/testing/AI-Hackathon24/data/merged.json"
    main(xml_folder, actions_folder, output_file)
