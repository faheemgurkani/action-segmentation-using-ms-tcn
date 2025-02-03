import os
import torch
import numpy as np
import random
from grid_sampler import GridSampler, TimeWarpLayer


class BatchGenerator:
    def __init__(self, num_classes, actions_dict, gt_path, features_path, sample_rate, model_type):
        """
        Args:
            num_classes (int): Total number of action classes.
            actions_dict (dict): Mapping from action labels to class indices.
            gt_path (str): Path to ground truth annotation files.
            features_path (str): Path to `.npy` feature files.
            sample_rate (int): Sampling rate for downsampling/up-sampling frames.
            model_type (str): Model type, either 'mstcn' or 'asformer'.
        """
        self.index = 0
        self.num_classes = num_classes
        self.actions_dict = actions_dict
        self.gt_path = gt_path
        self.features_path = features_path
        self.sample_rate = sample_rate
        self.model_type = model_type
        self.timewarp_layer = TimeWarpLayer() if model_type == "asformer" else None
        self.list_of_examples = []

    def reset(self):
        self.index = 0
        self.shuffle_data()

    def has_next(self):
        return self.index < len(self.list_of_examples)

    def load_examples(self):
        """
        Load all feature files and ground truth files based on the directory structure.
        """
        # Get all .npy files in the features directory
        feature_files = [f for f in os.listdir(self.features_path) if f.endswith('.npy')]
        self.list_of_examples = [os.path.splitext(f)[0] for f in feature_files]

        self.gts = [os.path.join(self.gt_path, f"{vid}.txt") for vid in self.list_of_examples]
        self.features = [os.path.join(self.features_path, f"{vid}.npy") for vid in self.list_of_examples]

        assert len(self.gts) == len(self.features), "Mismatch between features and ground truth files."
        self.shuffle_data()

    def shuffle_data(self):
        """
        Shuffle data for randomness.
        """
        random_seed = random.randint(0, 100)
        random.seed(random_seed)
        random.shuffle(self.list_of_examples)
        random.seed(random_seed)
        random.shuffle(self.gts)
        random.seed(random_seed)
        random.shuffle(self.features)

    def warp_video(self, batch_input_tensor, batch_target_tensor):
        """
        Apply time warping for ASFormer model.
        """
        bs, _, T = batch_input_tensor.shape
        grid_sampler = GridSampler(T)
        grid = grid_sampler.sample(bs)
        grid = torch.from_numpy(grid).float()

        warped_batch_input_tensor = self.timewarp_layer(batch_input_tensor, grid, mode='bilinear')
        batch_target_tensor = batch_target_tensor.unsqueeze(1).float()
        warped_batch_target_tensor = self.timewarp_layer(batch_target_tensor, grid, mode='nearest')
        warped_batch_target_tensor = warped_batch_target_tensor.squeeze(1).long()

        return warped_batch_input_tensor, warped_batch_target_tensor

    def next_batch(self, batch_size, if_warp=False):
        """
        Generate the next batch of data.
        Args:
            batch_size (int): Number of samples in the batch.
            if_warp (bool): Apply time warping (ASFormer only).
        """
        batch = self.list_of_examples[self.index:self.index + batch_size]
        batch_gts = self.gts[self.index:self.index + batch_size]
        batch_features = self.features[self.index:self.index + batch_size]

        self.index += batch_size

        batch_input = []
        batch_target = []
        for idx, vid in enumerate(batch):
            features = np.load(batch_features[idx])
            with open(batch_gts[idx], 'r') as file_ptr:
                content = file_ptr.read().split('\n')[:-1]

            classes = np.zeros(min(features.shape[1], len(content)))
            for i in range(len(classes)):
                classes[i] = self.actions_dict.get(content[i], -1)

            feature = features[:, ::self.sample_rate]
            target = classes[::self.sample_rate]
            batch_input.append(feature)
            batch_target.append(target)

        max_seq_length = max(map(len, batch_target))
        batch_input_tensor = torch.zeros(len(batch_input), batch_input[0].shape[0], max_seq_length, dtype=torch.float)
        batch_target_tensor = torch.ones(len(batch_input), max_seq_length, dtype=torch.long) * (-100)
        mask = torch.zeros(len(batch_input), self.num_classes, max_seq_length, dtype=torch.float)

        for i, (feature, target) in enumerate(zip(batch_input, batch_target)):
            if self.model_type == "asformer" and if_warp:
                warped_input, warped_target = self.warp_video(
                    torch.from_numpy(feature).unsqueeze(0), torch.from_numpy(target).unsqueeze(0)
                )
                batch_input_tensor[i, :, :warped_input.shape[-1]] = warped_input.squeeze(0)
                batch_target_tensor[i, :warped_target.shape[-1]] = warped_target.squeeze(0)
            else:
                batch_input_tensor[i, :, :feature.shape[1]] = torch.from_numpy(feature)
                batch_target_tensor[i, :target.shape[0]] = torch.from_numpy(target)

            mask[i, :, :target.shape[0]] = torch.ones(self.num_classes, target.shape[0])

        return batch_input_tensor, batch_target_tensor, mask, batch


def parse_actions_dict(mapping_file):
    """
    Parse actions_dict from a mapping.txt file.
    Args:
        mapping_file (str): Path to the mapping file.
    Returns:
        dict: A dictionary mapping action names to indices.
    """
    actions_dict = {}
    with open(mapping_file, 'r') as file:
        for line_num, line in enumerate(file, 1):
            parts = line.strip().split()
            if len(parts) != 2:
                print(f"Warning: Skipping malformed line {line_num}: {line.strip()}")
                continue
            idx, action = parts
            if not idx.isdigit():
                print(f"Warning: Skipping line with non-integer index at line {line_num}: {line.strip()}")
                continue
            actions_dict[action] = int(idx)
    return actions_dict



if __name__ == '__main__':
    mapping_file = "mapping.txt" 
    actions_dict = parse_actions_dict(mapping_file)
    
    num_classes = len(actions_dict)
    gt_path = "./groundTruth" 
    features_path = "./features"  
    sample_rate = 1
    model_type = "mstcn"

    batch_gen = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate, model_type)
    batch_gen.load_examples()

    while batch_gen.has_next():
        batch_input, batch_target, mask, batch_videos = batch_gen.next_batch(1, if_warp=(model_type == "asformer"))
        print(f"Batch Input Shape: {batch_input.shape}, Target Shape: {batch_target.shape}, Mask Shape: {mask.shape}")
