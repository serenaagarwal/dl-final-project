import os
import tifffile as tif
import numpy as np
import matplotlib.pyplot as plt #to plot images to see if they loaded in properly


def load_tif_data(folder_path):
    image_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.tif')])
    images = [tif.imread(os.path.join(folder_path, f)) for f in image_files]
    return images, image_files






