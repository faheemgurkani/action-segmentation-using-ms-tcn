import os
import sys



def generate_split_files(dataset, split, ratio, frame_features_root):
    # Listing video sequences as given in the task
    video_sequences = os.listdir(frame_features_root)

    # Shuffling or split the video sequences for train/test as needed
    train_sequences = video_sequences[:int(len(video_sequences) * ratio)]  # 80% for training
    test_sequences = video_sequences[int(len(video_sequences) * ratio):]  # 20% for testing

    # Defining the directory paths
    train_file_path = f"./data/{dataset}/splits/train.split{split}.bundle"
    test_file_path = f"./data/{dataset}/splits/test.split{split}.bundle"

    # Function to write the sequences to the respective files
    def write_sequences_to_file(file_path, sequences):
        
        # Checking if the file exists and remove it if it does
        if os.path.exists(file_path):
            # print(f"File {file_path} already exists. Removing it...")
            os.remove(file_path)  # Removing the existing file

        # Now, creating and write the new file
        with open(file_path, 'w') as f:
                f.write("\n".join(sequences) + "\n")

    # Making sure the parent directories exist
    os.makedirs(os.path.dirname(train_file_path), exist_ok=True)
    os.makedirs(os.path.dirname(test_file_path), exist_ok=True)

    # Writing the train and test sequences to respective files
    write_sequences_to_file(train_file_path, train_sequences)
    write_sequences_to_file(test_file_path, test_sequences)

    # print(f"Train and test split files have been generated:")
    # print(f"Train file: {train_file_path}")
    # print(f"Test file: {test_file_path}")



# Ensuring the script is only run when executed directly, not when imported
if __name__ == "__main__":
    dataset = "gtea"  
    split = 1  
    frame_features_root = f"./data/{dataset}/frames/"
    
    generate_split_files(dataset, split, 0.8, frame_features_root)  # Added split ratio for clarity
