#!/usr/bin/python2.7
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import copy
import numpy as np
from tqdm import tqdm
import os



#############################################
# Fusion Block: Concatenation + 1x1 Convolution
#############################################
class FusionBlock(nn.Module):
    def __init__(self, num_f_maps):
        super(FusionBlock, self).__init__()
        # Fuse two streams (2*num_f_maps channels) and project back to num_f_maps
        self.fusion_conv = nn.Conv1d(2 * num_f_maps, num_f_maps, 1)
    
    def forward(self, rgb_feat, kp_feat):
        fused = torch.cat([rgb_feat, kp_feat], dim=1)
        fused = self.fusion_conv(fused)
        return fused

#############################################
# Dilated Residual Layer
#############################################
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

#############################################
# Two-Stream Single Stage Model
#############################################
class TwoStreamSingleStageModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, rgb_dim, kp_dim, num_classes, fusion_point=5):
        """
        fusion_point: index after which the two modalities are fused.
        """
        super(TwoStreamSingleStageModel, self).__init__()
        self.fusion_point = fusion_point

        # Separate initial processing for RGB and keypoint streams.
        self.rgb_conv = nn.Conv1d(rgb_dim, num_f_maps, 1)
        self.kp_conv  = nn.Conv1d(kp_dim, num_f_maps, 1)

        # Independent dilated residual layers before fusion.
        self.rgb_layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps))
                                         for i in range(num_layers)])
        self.kp_layers  = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps))
                                         for i in range(num_layers)])

        # Fusion block to combine the modalities.
        self.fusion_block = FusionBlock(num_f_maps)

        # Post-fusion processing using the remaining layers.
        self.post_fusion_layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps))
                                                 for i in range(fusion_point, num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)
    
    def forward(self, rgb, kp, mask):
        # Process each stream separately.
        rgb_feat = self.rgb_conv(rgb)
        kp_feat = self.kp_conv(kp)
        for i in range(self.fusion_point):
            rgb_feat = self.rgb_layers[i](rgb_feat, mask)
            kp_feat  = self.kp_layers[i](kp_feat, mask)
        
        # Fuse the two streams.
        fused = self.fusion_block(rgb_feat, kp_feat)
        
        # Process the fused features further.
        for layer in self.post_fusion_layers:
            fused = layer(fused, mask)
        
        out = self.conv_out(fused) * mask[:, 0:1, :]
        return out

#############################################
# Standard Single Stage Model (Refinement Stage)
#############################################
class SingleStageModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(SingleStageModel, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps))
                                     for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, mask):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out, mask)
        out = self.conv_out(out) * mask[:, 0:1, :]
        return out

#############################################
# Multi-Stage Model
#############################################
class MultiStageModel(nn.Module):
    def __init__(self, num_stages, num_layers, num_f_maps, rgb_dim, kp_dim, num_classes, fusion_point=13):
        super(MultiStageModel, self).__init__()
        # Stage 1: Two-stream model with intermediate fusion.
        self.stage1 = TwoStreamSingleStageModel(num_layers, num_f_maps, rgb_dim, kp_dim, num_classes, fusion_point)
        # Subsequent stages: standard single-stage refinement models.
        self.stages = nn.ModuleList([copy.deepcopy(SingleStageModel(num_layers, num_f_maps, num_classes, num_classes))
                                      for _ in range(num_stages-1)])
    
    def forward(self, rgb, kp, mask):
        out = self.stage1(rgb, kp, mask)
        outputs = out.unsqueeze(0)
        for s in self.stages:
            # Refinement: compute softmax on the original p (shape: batch x num_classes x T)
            out = s(F.softmax(out, dim=1) * mask[:, 0:1, :], mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        return outputs

#############################################
# Trainer
#############################################
class Trainer:
    def __init__(self, num_stages, num_layers, num_f_maps, rgb_dim, kp_dim, num_classes):
        self.model = MultiStageModel(num_stages, num_layers, num_f_maps, rgb_dim, kp_dim, num_classes)
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.mse = nn.MSELoss(reduction='none')
        self.num_classes = num_classes

    def train(self, save_dir, batch_gen, num_epochs, batch_size, learning_rate, device):
        self.model.train()
        self.model.to(device)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        best_acc = 0.0  # Initialize best accuracy tracker
        for epoch in tqdm(range(num_epochs)):
            epoch_loss = 0
            correct = 0
            total = 0
            while batch_gen.has_next():
                # Get both RGB and keypoint inputs from the batch generator.
                batch_rgb, batch_kp, batch_target, mask = batch_gen.next_batch(batch_size)
                batch_rgb = batch_rgb.to(device)
                batch_kp = batch_kp.to(device)
                batch_target = batch_target.to(device)
                mask = mask.to(device)
                optimizer.zero_grad()
                predictions = self.model(batch_rgb, batch_kp, mask)
                loss = 0
                for p in predictions:
                    # For cross entropy, transpose p to shape (batch, T, num_classes)
                    loss += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), batch_target.view(-1))
                    # For temporal smoothness loss, use the original p (shape: batch, num_classes, T)
                    loss += 0.15 * torch.mean(torch.clamp(
                        self.mse(F.log_softmax(p[:, :, 1:], dim=1),
                                 F.log_softmax(p.detach()[:, :, :-1], dim=1)),
                        min=0, max=16) * mask[:, :, 1:])
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()
                _, predicted = torch.max(predictions[-1].data, 1)
                correct += ((predicted == batch_target).float() * mask[:, 0, :].squeeze(1)).sum().item()
                total += torch.sum(mask[:, 0, :]).item()
            batch_gen.reset()
            # Save model after each epoch as usual.
            torch.save(self.model.state_dict(), os.path.join(save_dir, f"epoch-{epoch+1}.model"))
            torch.save(optimizer.state_dict(), os.path.join(save_dir, f"epoch-{epoch+1}.opt"))
            epoch_acc = float(correct) / total
            print("[epoch %d]: epoch loss = %f,   acc = %f" % (epoch+1, epoch_loss/len(batch_gen.list_of_examples), epoch_acc))
            # If this epoch has the best accuracy so far, save it as the best model.
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(self.model.state_dict(), os.path.join(save_dir, "best.model"))
                torch.save(optimizer.state_dict(), os.path.join(save_dir, "best.opt"))
                print("Best model updated at epoch %d with acc = %f" % (epoch+1, best_acc))

    def predict(self, model_dir, results_dir, features_path, kp_features_path, vid_list_file, epoch, actions_dict, device, sample_rate, gt_path):
        self.model.eval()
        total_correct = 0
        total_count = 0
        with torch.no_grad():
            self.model.to(device)
            # Load the best model instead of using epoch number.
            self.model.load_state_dict(torch.load(os.path.join(model_dir, "best.model"), map_location=device, weights_only=True))
            with open(vid_list_file, 'r') as file_ptr:
                list_of_vids = file_ptr.read().split('\n')[:-1]
            for vid in tqdm(list_of_vids):
                # Load RGB features.
                rgb_features = np.load(os.path.join(features_path, vid.split('.')[0] + '.npy'))
                # Load keypoint features.
                kp_features = np.load(os.path.join(kp_features_path, vid.split('.')[0] + '.npy'))
                # --- Trim both modalities to a common sequence length ---
                common_length = min(rgb_features.shape[1], kp_features.shape[1])
                rgb_features = rgb_features[:, :common_length]
                kp_features = kp_features[:, :common_length]
                # Apply sampling.
                rgb_features = rgb_features[:, ::sample_rate]
                kp_features = kp_features[:, ::sample_rate]
                # Create tensors.
                input_rgb = torch.tensor(rgb_features, dtype=torch.float).unsqueeze(0).to(device)
                input_kp = torch.tensor(kp_features, dtype=torch.float).unsqueeze(0).to(device)
                # Create mask with the same time length as input_rgb.
                mask = torch.ones((input_rgb.size(0), self.num_classes, input_rgb.size(2)), device=device)
                predictions = self.model(input_rgb, input_kp, mask)
                _, predicted = torch.max(predictions[-1].data, 1)
                predicted = predicted.squeeze()
                
                # Load ground-truth labels.
                with open(os.path.join(gt_path, vid[:-4] + '.txt'), 'r') as file_ptr:
                    content = file_ptr.read().split('\n')[:-1]
                # Determine ground truth length matching the trimmed features.
                gt_seq_length = min(common_length, len(content))
                # Create target labels array and apply the same sampling.
                target = np.zeros(gt_seq_length)
                for i in range(gt_seq_length):
                    target[i] = actions_dict[content[i]]
                target = target[::sample_rate]
                # Convert to tensor.
                target_tensor = torch.tensor(target, dtype=torch.long).to(device)
                
                # Calculate accuracy for this video.
                correct = (predicted[:len(target_tensor)] == target_tensor).float().sum().item()
                total = len(target_tensor)
                total_correct += correct
                total_count += total
                
                recognition = []
                for i in range(len(predicted)):
                    recognition = np.concatenate(
                        (recognition, [list(actions_dict.keys())[list(actions_dict.values()).index(predicted[i].item())]] * sample_rate)
                    )
                f_name = vid.split('/')[-1].split('.')[0]
                with open(os.path.join(results_dir, f_name), "w") as f_ptr:
                    f_ptr.write("### Frame level recognition: ###\n")
                    f_ptr.write(' '.join(recognition))
            print("Overall prediction accuracy: {:.4f}".format(total_correct / total_count))
