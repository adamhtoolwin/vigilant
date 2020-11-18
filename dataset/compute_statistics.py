import numpy as np
import cv2
import pandas as pd

import argparse
from tqdm import tqdm

"""
Calculate the mean and std dev of the given dataset
"""


def get_means(img: np.ndarray):
    """
    Gets the mean of the image.
    :param img: ndarray
    :return: tuple of channelwise means
    """

    red = np.reshape(img[:, :, 0], -1)
    green = np.reshape(img[:, :, 1], -1)
    blue = np.reshape(img[:, :, 2], -1)

    red_mean = np.mean(red)
    green_mean = np.mean(green)
    blue_mean = np.mean(blue)

    return red_mean, green_mean, blue_mean


def get_std_dev(img):
    img = cv2.imread(path)
    red = np.reshape(img[:, :, 0], -1)
    green = np.reshape(img[:, :, 1], -1)
    blue = np.reshape(img[:, :, 2], -1)

    red_std = np.std(red)
    green_std = np.std(green)
    blue_std = np.std(blue)

    return red_std, green_std, blue_std


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-csv", "--csv_path", required=True, help="CSV file path.")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    print("Detected images: ", len(df))

    red_means = []
    green_means = []
    blue_means = []

    red_stds = []
    green_stds = []
    blue_stds = []

    pbar = tqdm(range(len(df)))
    for index in pbar:
        path = df.loc[index].path
        # pbar.set_description("Processing index" % path)

        image = cv2.imread(path)
        
        means = get_means(image)
        stds = get_std_dev(image)
        
        red_means.append(means[0])
        green_means.append(means[1])
        blue_means.append(means[2])

        red_stds.append(stds[0])
        green_stds.append(stds[1])
        blue_stds.append(stds[2])

    red_mean = np.sum(red_means) / len(df)
    green_mean = np.sum(green_means) / len(df)
    blue_mean = np.sum(blue_means) / len(df)

    red_std = np.sum(red_stds) / len(df)
    green_std = np.sum(green_stds) / len(df)
    blue_std = np.sum(blue_stds) / len(df)

    print("Red mean: ", red_mean / 255)
    print("Green mean: ", green_mean / 255)
    print("Blue mean: ", blue_mean / 255)

    print("Red std: ", red_std / 255)
    print("Green std: ", green_std / 255)
    print("Blue std: ", blue_std / 255)

        


