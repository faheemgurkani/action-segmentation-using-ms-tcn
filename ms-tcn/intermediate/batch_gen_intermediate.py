#!/usr/bin/python2.7
import torch
import numpy as np
import random



class BatchGenerator(object):

    def __init__(self, num_classes, actions_dict, gt_path, features_path, kp_features_path, sample_rate):
        self.list_of_examples = list()
        self.index = 0
        self.num_classes = num_classes
        self.actions_dict = actions_dict
        self.gt_path = gt_path
        self.features_path = features_path
        self.kp_features_path = kp_features_path
        self.sample_rate = sample_rate

    def reset(self):
        self.index = 0
        random.shuffle(self.list_of_examples)

    def has_next(self):
        return self.index < len(self.list_of_examples)

    def read_data(self, vid_list_file):
        with open(vid_list_file, 'r') as file_ptr:
            self.list_of_examples = file_ptr.read().split('\n')[:-1]
        random.shuffle(self.list_of_examples)

    def next_batch(self, batch_size):
        batch = self.list_of_examples[self.index:self.index + batch_size]
        self.index += batch_size

        batch_rgb = []
        batch_kp = []
        batch_target = []

        for vid in batch:
            # Load RGB features from corresponding .npy file.
            rgb_features = np.load(self.features_path + vid.split('.')[0] + '.npy')
            # Load keypoint features from corresponding .npy file.
            kp_features = np.load(self.kp_features_path + vid.split('.')[0] + '.npy')
            # Load ground truth labels.
            with open(self.gt_path + vid[:-4] + '.txt', 'r') as file_ptr:
                content = file_ptr.read().split('\n')[:-1]

            # Determine common sequence length.
            rgb_len = np.shape(rgb_features)[1]
            kp_len = np.shape(kp_features)[1]
            gt_len = len(content)
            seq_length = min(rgb_len, kp_len, gt_len)

            # Build target labels based on ground truth.
            classes = np.zeros(seq_length)
            for i in range(seq_length):
                classes[i] = self.actions_dict[content[i]]

            # Trim both modalities to the common sequence length and then sample.
            batch_rgb.append(rgb_features[:, :seq_length][:, ::self.sample_rate])
            batch_kp.append(kp_features[:, :seq_length][:, ::self.sample_rate])
            batch_target.append(classes[::self.sample_rate])

        lengths = list(map(len, batch_target))
        max_len = max(lengths)

        batch_rgb_tensor = torch.zeros(len(batch_rgb), np.shape(batch_rgb[0])[0], max_len, dtype=torch.float)
        batch_kp_tensor = torch.zeros(len(batch_kp), np.shape(batch_kp[0])[0], max_len, dtype=torch.float)
        batch_target_tensor = torch.ones(len(batch_rgb), max_len, dtype=torch.long) * (-100)
        mask = torch.zeros(len(batch_rgb), self.num_classes, max_len, dtype=torch.float)

        for i in range(len(batch_rgb)):
            seq_len = np.shape(batch_rgb[i])[1]
            batch_rgb_tensor[i, :, :seq_len] = torch.from_numpy(batch_rgb[i])
            batch_kp_tensor[i, :, :seq_len] = torch.from_numpy(batch_kp[i])
            batch_target_tensor[i, :len(batch_target[i])] = torch.from_numpy(batch_target[i])
            mask[i, :, :len(batch_target[i])] = torch.ones(self.num_classes, len(batch_target[i]))

        return batch_rgb_tensor, batch_kp_tensor, batch_target_tensor, mask
