import torch
from model import Trainer
from batch_gen import BatchGenerator
import os
import random
import sys



def split_generation_utility(dataset, split, split_ratio, features_path):
    from utils import generate_split_files

    generate_split_files(dataset, split, split_ratio, features_path)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 1538574472
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    # Configuration Variables
    action = "predict"
    dataset = "gtea"
    split = "1"
    split_ratio = 0.8

    num_stages = 4
    num_layers = 10
    num_f_maps = 64
    features_dim = 2048
    bz = 1
    lr = 0.0005
    num_epochs = 50

    # Using the full temporal resolution @ 15fps
    sample_rate = 1
    # sample input features @ 15fps instead of 30 fps
    # for 50salads, and up-sample the output to 30 fps

    if dataset == "50salads":
        sample_rate = 2

    vid_list_file = f"./data/{dataset}/splits/train.split{split}.bundle"
    vid_list_file_tst = f"./data/{dataset}/splits/test.split{split}.bundle"
    features_path = f"./data/{dataset}/merged_extracted_frame_features/"
    gt_path = f"./data/{dataset}/action_labels/"
    mapping_file = f"./data/{dataset}/mapping.txt"

    split_generation_utility(dataset, split, split_ratio, features_path)

    model_dir = f"./models/{dataset}/split_{split}"
    results_dir = f"./results/{dataset}/split_{split}"

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Loading action mappings
    with open(mapping_file, 'r') as file_ptr:
        actions = file_ptr.read().split('\n')[:-1]

    actions_dict = {a.split()[1]: int(a.split()[0]) for a in actions}
    num_classes = len(actions_dict)

    trainer = Trainer(num_stages, num_layers, num_f_maps, features_dim, num_classes)

    if action == "train":
        batch_gen = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate)
        batch_gen.read_data(vid_list_file)
        
        trainer.train(model_dir, batch_gen, num_epochs=num_epochs, batch_size=bz, learning_rate=lr, device=device)

    if action == "predict":
        trainer.predict(model_dir, results_dir, features_path, vid_list_file_tst, num_epochs, actions_dict, device, sample_rate)



if __name__ == "__main__":
    main()