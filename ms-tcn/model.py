# !/usr/bin/python2.7

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import copy
import numpy as np
from tqdm import tqdm


class MultiStageModel(nn.Module):
    def __init__(self, num_stages, num_layers, num_f_maps, dim, num_classes):
        super(MultiStageModel, self).__init__()
        self.stage1 = SingleStageModel(num_layers, num_f_maps, dim, num_classes)
        self.stages = nn.ModuleList([copy.deepcopy(SingleStageModel(num_layers, num_f_maps, num_classes, num_classes)) for s in range(num_stages-1)])

    def forward(self, x, mask):
        out = self.stage1(x, mask)
        outputs = out.unsqueeze(0)
        for s in self.stages:
            out = s(F.softmax(out, dim=1) * mask[:, 0:1, :], mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        return outputs


class SingleStageModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(SingleStageModel, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, mask):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out, mask)
        out = self.conv_out(out) * mask[:, 0:1, :]
        return out


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x, mask):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out) * mask[:, 0:1, :]


class Trainer:
   
    def __init__(self, num_blocks, num_layers, num_f_maps, dim, num_classes):
        self.model = MultiStageModel(num_blocks, num_layers, num_f_maps, dim, num_classes)
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.mse = nn.MSELoss(reduction='none')
        self.num_classes = num_classes

    def train(self, save_dir, batch_gen, num_epochs, batch_size, learning_rate, device):
        self.model.train()
        self.model.to(device)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
   
        for epoch in tqdm(range(num_epochs)):
            epoch_loss = 0
            correct = 0
            total = 0
   
            while batch_gen.has_next():
                batch_input, batch_target, mask = batch_gen.next_batch(batch_size)
                batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device)
                optimizer.zero_grad()
                predictions = self.model(batch_input, mask)
                loss = 0
   
                for p in predictions:
                    loss += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), batch_target.view(-1))
                    loss += 0.15*torch.mean(torch.clamp(self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0, max=16)*mask[:, :, 1:])

                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(predictions[-1].data, 1)
                correct += ((predicted == batch_target).float()*mask[:, 0, :].squeeze(1)).sum().item()
                total += torch.sum(mask[:, 0, :]).item()

            batch_gen.reset()
            torch.save(self.model.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".model")
            torch.save(optimizer.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".opt")
      
            print("[epoch %d]: epoch loss = %f,   acc = %f" % (epoch + 1, epoch_loss / len(batch_gen.list_of_examples), float(correct)/total))

    # def predict(self, model_dir, results_dir, features_path, vid_list_file, epoch, actions_dict, device, sample_rate):
    #     self.model.eval()
    #     with torch.no_grad():
    #         self.model.to(device)
    #         self.model.load_state_dict(torch.load(model_dir + "/epoch-" + str(epoch) + ".model", weights_only=True))
    #         file_ptr = open(vid_list_file, 'r')
    #         list_of_vids = file_ptr.read().split('\n')[:-1]
    #         file_ptr.close()
    #         for vid in tqdm(list_of_vids):
    #             # print vid
    #             features = np.load(features_path + vid.split('.')[0] + '.npy')
    #             features = features[:, ::sample_rate]
    #             input_x = torch.tensor(features, dtype=torch.float)
    #             input_x.unsqueeze_(0)
    #             input_x = input_x.to(device)
    #             predictions = self.model(input_x, torch.ones(input_x.size(), device=device))
    #             _, predicted = torch.max(predictions[-1].data, 1)
    #             predicted = predicted.squeeze()
    #             recognition = []
    #             for i in range(len(predicted)):
    #                 # recognition = np.concatenate((recognition, [list(actions_dict.keys()[actions_dict.values().index(predicted[i].item())])]*sample_rate))
    #                 recognition = np.concatenate(
    #                     (recognition, [list(actions_dict.keys())[list(actions_dict.values()).index(predicted[i].item())]] * sample_rate)
    #                 )

    #             f_name = vid.split('/')[-1].split('.')[0]
    #             f_ptr = open(results_dir + "/" + f_name, "w")
    #             f_ptr.write("### Frame level recognition: ###\n")
    #             f_ptr.write(' '.join(recognition))
    #             f_ptr.close()

    def predict(self, model_dir, results_dir, features_path, gt_path, vid_list_file, epoch, actions_dict, device, sample_rate):
        self.model.eval()
        total_correct = 0
        total_frames = 0
        
        with torch.no_grad():
            self.model.to(device)
            self.model.load_state_dict(torch.load(model_dir + "/epoch-" + str(epoch) + ".model", weights_only=True))
        
            with open(vid_list_file, 'r') as file_ptr:
                list_of_vids = file_ptr.read().split('\n')[:-1]

            # print(list_of_vids)  # For, testing
            
            for vid in tqdm(list_of_vids):
                # print(vid)  # For, testing
                # print(features_path + vid.split('.')[0] + '.npy')

                # Loading features and apply sampling
                features = np.load(features_path + vid.split('.')[0] + '.npy')

                # print(features.shape)  # For, testing

                features = features[:, ::sample_rate]

                # print(features.shape)  # For, testing

                input_x = torch.tensor(features, dtype=torch.float).unsqueeze(0).to(device)
                
                # Forward pass through the model
                predictions = self.model(input_x, torch.ones(input_x.size(), device=device))
                _, predicted = torch.max(predictions[-1].data, 1)
                predicted = predicted.squeeze().cpu().numpy()  # Converting to NumPy array

                # Loading the ground truth labels for this video
                gt_file = gt_path + vid[:-4] + '.txt'

                # print(gt_file)  # For, testing

                with open(gt_file, 'r') as f:
                    gt_labels_str = f.read().split('\n')[:-1]
                
                # Converting string labels to numeric labels and apply the same sampling as features
                gt_labels = np.array([actions_dict[label] for label in gt_labels_str])[::sample_rate]

                # print(gt_labels)  # For, testing
                # print(predicted)
                # print(len(gt_labels), len(predicted))
                
                # Calculating accuracy for this video
                gt_labels_1 = gt_labels[:len(predicted)]

                correct = (predicted == gt_labels_1).sum()
                total = len(gt_labels_1)
                total_correct += correct
                total_frames += total

                # Writing frame-level recognition file (unchanged)
                recognition = []
                for i in range(len(predicted)):
                    recognition = np.concatenate(
                        (recognition, [list(actions_dict.keys())[list(actions_dict.values()).index(predicted[i])]] * sample_rate)
                    )
                
                f_name = vid.split('/')[-1].split('.')[0]
                
                with open(results_dir + "/" + f_name, "w") as f_ptr:
                    f_ptr.write("### Frame level recognition: ###\n")
                    f_ptr.write(' '.join(recognition))
        
        # Computing and print overall accuracy
        accuracy = total_correct / total_frames
        print("Overall prediction accuracy: {:.2f}%".format(accuracy * 100))

    # def predict(self, model_dir, batch_gen, results_dir, features_path, gt_path, epoch, actions_dict, device, sample_rate):
    #     self.model.eval()
    #     total_correct = 0
    #     total_frames = 0

    #     self.model.to(device)
    #     self.model.load_state_dict(torch.load(model_dir + "/epoch-" + str(epoch) + ".model", weights_only=True))
        
    #     with torch.no_grad():

    #         while batch_gen.has_next():
    #             # Getting the next batch (assuming batch size of 1 for prediction)
    #             batch_input, batch_target, mask = batch_gen.next_batch(1)
    #             batch_input = batch_input.to(device)
    #             batch_target = batch_target.to(device)
    #             mask = mask.to(device)
                
    #             # Forward pass with the provided mask
    #             predictions = self.model(batch_input, mask)
    #             _, predicted = torch.max(predictions[-1].data, 1)
                
    #             # Using the mask (same as in training) for calculating accuracy
    #             correct = ((predicted == batch_target).float() * mask[:, 0, :].squeeze(1)).sum().item()
    #             total = torch.sum(mask[:, 0, :]).item()
    #             total_correct += correct
    #             total_frames += total

    #             # Writing recognition output for this video
    #             # (Assuming that the batch generator preserves the video order and name)
    #             # Here, we extract the video name from the batch generator’s example list.
    #             vid = batch_gen.list_of_examples[batch_gen.index - 1]
    #             recognition = []
                
    #             for i in range(predicted.size(1)):
    #                 action_name = list(actions_dict.keys())[list(actions_dict.values()).index(predicted[0, i].item())]
    #                 recognition = np.concatenate((recognition, [action_name] * sample_rate))
                
    #             f_name = vid.split('/')[-1].split('.')[0]
                
    #             with open(results_dir + "/" + f_name, "w") as f_ptr:
    #                 f_ptr.write("### Frame level recognition: ###\n")
    #                 f_ptr.write(' '.join(recognition))
                    
    #         batch_gen.reset()

    #     accuracy = total_correct / total_frames
    #     print("Overall prediction accuracy: {:.2f}%".format(accuracy * 100))
