import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import json
import os
import torch
import cv2
import math, copy
import numpy as np
import json
import yaml 
from typing import Optional, Dict
import torch.utils.data as data
import albumentations as A

def draw_msra_gaussian(heatmap, center, sigma):
  tmp_size = sigma * 3
  mu_x = int(center[0] + 0.5)
  mu_y = int(center[1] + 0.5)
  w, h = heatmap.shape[0], heatmap.shape[1]
  ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
  br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
  if ul[0] >= h or ul[1] >= w or br[0] < 0 or br[1] < 0:
    return heatmap
  size = 2 * tmp_size + 1
  x = np.arange(0, size, 1, np.float32)
  y = x[:, np.newaxis]
  x0 = y0 = size // 2
  g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
  g_x = max(0, -ul[0]), min(br[0], h) - ul[0]
  g_y = max(0, -ul[1]), min(br[1], w) - ul[1]
  img_x = max(0, ul[0]), min(br[0], h)
  img_y = max(0, ul[1]), min(br[1], w)
  heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
    heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]],
    g[g_y[0]:g_y[1], g_x[0]:g_x[1]])
  return heatmap


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_umich_gaussian(heatmap, center, radius, k=1):
  diameter = 2 * radius + 1
  gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
  
  x, y = int(center[0]), int(center[1])

  height, width = heatmap.shape[0:2]
    
  left, right = min(x, radius), min(width - x, radius + 1)
  top, bottom = min(y, radius), min(height - y, radius + 1)

  masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
  masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
  if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0: # TODO debug
    np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
  return heatmap


def gaussian_radius(det_size, min_overlap=0.7):
  height, width = det_size

  a1  = 1
  b1  = (height + width)
  c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
  sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
  r1  = (b1 + sq1) / 2

  a2  = 4
  b2  = 2 * (height + width)
  c2  = (1 - min_overlap) * width * height
  sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
  r2  = (b2 + sq2) / 2

  a3  = 4 * min_overlap
  b3  = -2 * min_overlap * (height + width)
  c3  = (min_overlap - 1) * width * height
  sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
  r3  = (b3 + sq3) / 2
  return min(r1, r2, r3)

def mapping_keypoints(label_keypoints):
  for filling in (0,21):
      if filling<21:
        for i in range(5):
          label_keypoints[filling+i] = 'right_thumb'
        for i in range(4):
          label_keypoints[filling+5+i] = 'right_index_finger'
        for i in range(4):

          label_keypoints[filling+9+i] = 'right_middle_finger'
        for i in range(4):
          label_keypoints[filling+13+i] = 'right_ring_finger'
        for i in range(4):
          label_keypoints[filling+17+i] = 'right_pinkie_finger'
      else:
        for i in range(5):
          label_keypoints[filling+i] = 'left_thumb'
        for i in range(4):
          label_keypoints[filling+5+i] = 'left_index_finger'
        for i in range(4):

          label_keypoints[filling+9+i] = 'left_middle_finger'
        for i in range(4):
          label_keypoints[filling+13+i] = 'left_ring_finger'
        for i in range(4):
          label_keypoints[filling+17+i] = 'left_pinkie_finger'

  return label_keypoints

class COCOHP(data.Dataset):
  def __init__(self):
    super(COCOHP,self).__init__()

    with open(os.path.join('dataloader','dataset.yml'), 'r') as file:
        configs = yaml.safe_load(file)
    file.close()
    with open(os.path.join('dataloader','config_hyper.yaml'), 'r') as file:
        configs_hyper = yaml.safe_load(file)
    file.close()

    self.edges = configs["edges"]
    self.reg_hp_offset = configs_hyper["reg_hp_offset"]
    self.dense_hp = configs_hyper["dense_hp"]
    self.hm_hp = configs_hyper["hm_hp"]
    self.reg_offset = configs_hyper["reg_offset"]
    self.albumentations = configs["albumentations"]
    self.num_classes = configs["num_classes"]
    self.down_ratio = configs["down_ratio"]
    self.kp_mapping = configs["kp_mapping"]
    self.colors_hp = configs["colors_hp"]
    self.ec = configs["ec"]
    self.mse_loss = configs_hyper["mse_loss"]
    self.acc_idxs = configs["acc_idxs"]
    self.mean = np.array(configs["mean"],
                   dtype=np.float32).reshape(1, 1, 3)
    self.std  = np.array(configs["std"],
                   dtype=np.float32).reshape(1, 1, 3) 
    self.data_dir = os.path.join(configs["dataset_dir"])

    self.annot_path = f"{self.data_dir}/merged.json"

    self.max_objs = configs["max_objs"]
    self._data_rng = np.random.RandomState(123)
    self._eig_val = np.array(configs["eig_val"],
                             dtype=np.float32)
    self._eig_vec = np.array(configs["eig_vect"], dtype=np.float32)
    self.action_mapping = configs["action_mapping"]


    self.action_classes = len(self.action_mapping.keys())
    self.coco = coco.COCO(self.annot_path)
    image_ids = self.coco.getImgIds()

    self.images = []
    for img_id in image_ids:
      idxs = self.coco.getAnnIds(imgIds=[img_id])
      if len(idxs) > 0:
        self.images.append(img_id)

    self.num_samples = len(self.images)

  def convert_eval_format(self, all_bboxes):
    detections = []
    for image_id in all_bboxes:
      for cls_ind in all_bboxes[image_id]:
        category_id = 1
        for dets in all_bboxes[image_id][cls_ind]:
          bbox = dets[:4]
          bbox[2] -= bbox[0]
          bbox[3] -= bbox[1]
          score = dets[4]
          bbox_out  = list(map(self._to_float, bbox))
          keypoints = np.concatenate([
            np.array(dets[5:47], dtype=np.float32).reshape(-1, 2), 
            np.ones((21, 1), dtype=np.float32)], axis=1).reshape(63).tolist()
          keypoints  = list(map(self._to_float, keypoints))

          detection = {
              "image_id": int(image_id),
              "category_id": int(category_id),
              "bbox": bbox_out,
              "score": float("{:.2f}".format(score)),
              "keypoints": keypoints
          }
          detections.append(detection)
    return detections



  def _coco_box_to_bbox(self, box):
        bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                        dtype=np.float32)
        return bbox

  def _get_border(self, border, size):
    i = 1
    while size - border // i <= border // i:
        i *= 2
    return border // i


  def get_class_weights(self):
    total_actions = []
    for img_id in range(self.__len__()):
      ann_ids = self.coco.getAnnIds(imgIds=[img_id])
      anns = self.coco.loadAnns(ids=ann_ids)
      total_actions.append(self.action_mapping[anns[0]['action']])
    full_event_seq = np.array(total_actions,dtype=np.float32)
    # full_event_seq = np.concatenate([self.data_dict[v]['event_seq_raw'] for v in self.video_list])
    self.class_num = len(self.action_mapping.keys())
    class_counts = np.zeros((self.class_num,))
    for c in range(self.class_num):
        class_counts[c] = (full_event_seq == c).sum()
                
    class_weights = class_counts.sum() / ((class_counts + 10) * self.class_num)
    return class_weights

  def draw_annotations(self, image, anns):
    """
    Draws bounding boxes and keypoints on the image based on the annotations.

    Args:
    - image: The image (numpy array) where the annotations will be drawn.
    - anns: A list of annotations containing bounding boxes and keypoints.

    Returns:
    - image: The image with drawn annotations (bounding box and keypoints).
    """
    for ann in anns:
        # Extract the bounding box (x, y, width, height)
        bbox = ann.get('bbox', [])
        if len(bbox) > 0:
            x, y, w, h = map(int, bbox)
            # Draw the bounding box
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Extract and draw the keypoints
        keypoints = ann.get('keypoints', [])
        for i in range(0, len(keypoints), 3):
            x, y, visibility = keypoints[i:i+3]
            if visibility == 2:  # Keypoint is visible
                cv2.circle(image, (int(x), int(y)), 5, (0, 0, 255), -1)
        
        # Draw the action text
        action = ann.get('action', '')
        cv2.putText(image, action, (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    
    return image

  def __getitem__(self, index):
    
    img_id = self.images[index]
    file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
    ann_ids = self.coco.getAnnIds(imgIds=[img_id])
    anns = self.coco.loadAnns(ids=ann_ids)
    num_objs = min(len(anns), self.max_objs)
    file_name = file_name.replace('AI-Hackathon24/','')
    img = cv2.imread(file_name)
    import pdb; pdb.set_trace()
    cv2.imwrite('test.png',self.draw_annotations(img,anns))
    return  torch.randn((3,50,50))
  
  def _to_float(self, x):
    return float("{:.2f}".format(x))



  def __len__(self):
    return self.num_samples

  def save_results(self, results, save_dir):
    with open(f'{save_dir}/results.json', 'w') as f:
        json.dump(self.convert_eval_format(results), f)


  def run_eval(self, results, save_dir):
    # result_json = os.path.join(opt.save_dir, "results.json")
    # detections  = convert_eval_format(all_boxes)
    # json.dump(detections, open(result_json, "w"))
    self.save_results(results, save_dir)
    coco_dets = self.coco.loadRes('{}/results.json'.format(save_dir))
    coco_eval = COCOeval(self.coco, coco_dets, "keypoints")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    coco_eval = COCOeval(self.coco, coco_dets, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

  def transformsTRAIN(self, img:None, bbox:list, keypoints: list,empty=False):
      # self.albumentations comes from our dataset configs (yaml)
      transform_keypoints = A.Compose([
        A.Resize(self.albumentations["resize_height"],self.albumentations["resize_width"],cv2.INTER_LINEAR),
        A.HorizontalFlip(p= self.albumentations["HorizontalFlip"]),
        A.VerticalFlip(p= self.albumentations["VerticalFlip"]),
        A.Affine(translate_percent = dict(x=(-0.1, 0.1),y=(-0.1, 0.1)),
                  rotate = (-10,10),
                  scale = (0.8, 1.2),
                  p= self.albumentations["Affine"]
                  )],
        keypoint_params=A.KeypointParams(format='xy', label_fields=['class_labels']), #kp mapping also comes from configs
        bbox_params=A.BboxParams(format='coco')
        )
      transformed = transform_keypoints(image=img, bboxes=bbox, keypoints=keypoints, class_labels=self.kp_mapping)
      #to allow indexing on the immutable set datatype
      special=False
      missing_labels = list(set(self.kp_mapping) - set(transformed["class_labels"]))
      for missing_label in missing_labels:
        transformed["class_labels"].insert(self.kp_mapping.index(missing_label),missing_label)
        transformed["keypoints"].insert(self.kp_mapping.index(missing_label),(0,0))
        print("Caution: Your augmentation has zeroed out one or more keypoints")
        print("Check special.png to see")
        special=True
      if self.albumentations["vis"]:
        
        vis_album(img,transformed["image"],bbox, transformed['bboxes'],keypoints,transformed["keypoints"],special=special)
      return transformed["image"], transformed['bboxes'], transformed['keypoints']

COCOHP()
dataset = COCOHP()
loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False,num_workers=0)
for batch in loader: 
  import pdb; pdb.set_trace()