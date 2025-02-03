import os
import sys
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import cv2
'''
t-SNE Basics
t-SNE is a dimensionality reduction technique that maps high-dimensional data into a two-dimensional or three-dimensional space for visualization. It is particularly useful for visualizing clusters or groupings in the data.

X and Y Axes
The X-axis and Y-axis in a t-SNE plot are arbitrary and do not correspond to any specific features or variables in the original data. Instead, they represent the transformed low-dimensional coordinates of the data points.
The axes should be interpreted in relative terms, focusing on the structure of the data rather than absolute values.
What Does This Plot Show?
Clusters and Groupings:

Each point represents a data sample.
Points that are close together in the plot are similar in the high-dimensional feature space, while points farther apart are less similar.
Color-Coded Labels:

The colors represent distinct classes or categories (e.g., "background," "take," "open," etc.).
The plot shows how well the classes separate in the low-dimensional space.
Insights:

If clusters for different classes are well-separated, it suggests that the features used can effectively distinguish between these classes.
Overlapping clusters might indicate some ambiguity or similarity between classes in the feature space.
 
'''
def tsne(args):
    # Check if the output directory exists, if not create it
    output_dir = args.out_path
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Process input file (video or numpy array)
    if args.input.endswith("mp4") or args.input.endswith("avi"):
        imgs_list = []
        video = cv2.VideoCapture(args.input)
        while True:
            ret, frame = video.read()
            if ret:
                imgs_list.append(np.expand_dims(frame, axis=0))
            else:
                break
        feature = np.concatenate(imgs_list, axis=0)
        feature = feature.reshape((feature.shape[0], -1)).T
    else:
        with open(args.input, "rb") as input_file:
            feature = np.load(input_file)

    # Read labels from the file
    with open(args.label, "r") as label_file:
        labels = label_file.read().split('\n')[:-1]

    # Perform t-SNE dimensionality reduction
    tsne = TSNE(n_components=2)
    tsne_obj = tsne.fit_transform(feature.T)

    # Create a DataFrame for plotting
    tsne_df = pd.DataFrame({'X': tsne_obj[:, 0],
                            'Y': tsne_obj[:, 1],
                            'label': labels})

    # Create the scatterplot
    img = sns.scatterplot(x="X", y="Y", hue="label", data=tsne_df)
    img.legend(ncol=4, fontsize=5)
    
    # Save the plot to the specified output directory
    img.figure.savefig(os.path.join(output_dir, "t-SNE_visualize.png"), bbox_inches='tight', dpi=500)
    plt.close()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input',
                        type=str,
                        default='input.npy',
                        help='Input feature file to visualize (video or .npy)')
    parser.add_argument('-l', '--label',
                        type=str,
                        default='video.txt',
                        help='Label file corresponding to the features')
    parser.add_argument('-o', '--out_path',
                        type=str,
                        default='./output',  # Default to current directory
                        help='Directory to save the t-SNE plot')

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    tsne(args)
