import os
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset, DataLoader
import cv2 

class HandKeypointDataset(Dataset):
    def __init__(self, xml_path='xml_labels', image_dir='png', transform=None):
        """
        Args:
            xml_path (str): Path to the XML annotation folder.
            image_dir (str): Directory containing corresponding image files.
            transform (callable, optional): Transform to be applied on the images.
        """
        self.xml_path = xml_path
        self.image_dir = image_dir
        self.transform = transform
        self.annotations = self.parse_xml()

    def parse_xml(self):
        """Parses the XML file and extracts annotations."""
        complete_dataset = []

        for xml_name in os.listdir(self.xml_path): 
            xml_path = os.path.join(self.xml_path, xml_name)
            tree = ET.parse(xml_path)
            root = tree.getroot()
            num_annos = len(root.findall('image'))
            all_annos = root.findall('image')
            print(f"Parsing {num_annos} total annotations for {xml_name}")
            
            for i in range(num_annos):
                boxes = {'left':[], 'right':[]}
                keypoints = {'left':{}, 'right':{}}

                curr_annos = []
                for anno in all_annos: 
                    try:
                        if int(anno.get('id')) == int(all_annos[i].get('id')):
                            curr_annos.append(anno)
                    except ValueError:
                        # print(f"WARNING: Found strings in image ids for xml: {xml_name}")
                        if int(anno.get('id').replace('frame_', '')) == int(all_annos[i].get('id').replace('frame_', '')):
                            curr_annos.append(anno)

                annos = {curr_anno.findall('box')[0].findall(".//attribute[@name='hand_type']")[0].text: curr_anno for curr_anno in curr_annos}
                
                if len(annos.keys()) < 2:
                    continue

                width = int(annos['left'].get('width'))
                height = int(annos['right'].get('height'))

                for anno_side, anno_val in annos.items():
                    for box in anno_val.findall('box'):
                        xtl = float(box.get('xtl'))
                        ytl = float(box.get('ytl'))
                        xbr = float(box.get('xbr'))
                        ybr = float(box.get('ybr'))
                        label = box.get('label')

                        boxes[anno_side].append({'label': label, 'bbox': [xtl, ytl, xbr, ybr]})

                    for polyline in anno_val.findall('polyline'):
                        label = polyline.get('label')
                        points = [
                            tuple(map(float, point.split(',')))
                            for point in polyline.get('points').split(';')
                        ]
                        keypoints[anno_side][label] = points

                complete_dataset.append(
                    {
                        'id': annos['left'].get('id'),
                        'video_name': xml_name.replace('.xml', ''),
                        'frame_name': annos['left'].get('name'),
                        'width': width,
                        'height': height,
                        'boxes': boxes,
                        'keypoints': keypoints
                    }
                )

        return complete_dataset

    def __len__(self):
        return len(self.annotations)
    
    def plot_annotations(self, image, boxes, keypoints):
        """
        Plots bounding boxes and keypoints on the image using OpenCV.

        Args:
            image_path (str): Path to the image file.
            boxes (dict): Dictionary containing bounding boxes in the format:
                        {'left': [{'label': 'hand', 'bbox': [x_min, y_min, x_max, y_max]}], 'right': ...}
            keypoints (dict): Dictionary containing keypoints in the format:
                            {'left': {'thumb': [(x1, y1), (x2, y2), ...], ...}, 'right': ...}
        """

        for hand_side, hand_boxes in boxes.items():
            for box_info in hand_boxes:
                x_min, y_min, x_max, y_max = map(int, box_info['bbox'])
                label = box_info['label']
                color = (0, 255, 0) if hand_side == 'left' else (255, 0, 0)
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
                cv2.putText(image, f"{label} ({hand_side})", (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        for hand_side, hand_keypoints in keypoints.items():
            for finger, points in hand_keypoints.items():
                color = (0, 255, 255) if hand_side == 'left' else (255, 255, 0)
                for point in points:
                    x, y = map(int, point)
                    cv2.circle(image, (x, y), 3, color, -1)
                for i in range(len(points) - 1):
                    pt1 = tuple(map(int, points[i]))
                    pt2 = tuple(map(int, points[i + 1]))
                    cv2.line(image, pt1, pt2, color, 2)
        return image 

    def __getitem__(self, idx):
        
        # Get annotation

        image_path = os.path.join(self.image_dir, self.annotations[idx]['video_name'], self.annotations[idx]['frame_name']+'.png')
        image = cv2.imread(image_path)

        ### visualize if you want
        # plotted = self.plot_annotations(image, self.annotations[idx]['boxes'], self.annotations[idx]['keypoints'])
        # cv2.imwrite('test.png', plotted)

        ### your transformations  

        return image, self.annotations[idx]

# dataset = HandKeypointDataset()
# dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

# set batch_size = 1 and plot a video of your annotations if you want
# video_filename = 'aaa.avi'
# frame_width, frame_height = 720, 405  
# fps = 15  
# fourcc = cv2.VideoWriter_fourcc(*'XVID')  
# out = cv2.VideoWriter(video_filename, fourcc, fps, (frame_width, frame_height))

# for idx,batch in enumerate(dataloader):
#     img = batch[0].squeeze().numpy()
#     if len(img.shape) == 2:
#         img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#     elif img.shape[-1] != 3:
#         raise ValueError("Images must have 3 channels for BGR format")
#     try:
#         img_resized = cv2.resize(img, (frame_width, frame_height))
#     except:
#         import pdb; pdb.set_trace()
         
#     out.write(img_resized)

# out.release()
# print(f"Video saved as {video_filename}")

