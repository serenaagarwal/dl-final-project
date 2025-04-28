# # import os
# # import tifffile as tif
# # import numpy as np
# # import matplotlib.pyplot as plt #to plot images to see if they loaded in properly


# # def load_tif_data(folder_path):
# #     image_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.tif')])
# #     images = [tif.imread(os.path.join(folder_path, f)) for f in image_files]
# #     return images, image_files

import rasterio #good option per this post: https://stackoverflow.com/questions/68020419/working-with-tiff-images-in-python-for-deep-learning
import os
import numpy as np
from skimage.segmentation import find_boundaries
from scipy.ndimage import distance_transform_edt
from augmentations import elastic_deformation, random_rotate_flip
    
def load_image_mask_pair(root_dir, sample_id, time_idx) -> tuple[np.ndarray, np.ndarray]:
    #loads one image mask pair
    #expected directory structure:
    #     root_dir/
    #     -> 01/                  
    #     ------> t000.tif
    #     ->01_ST/SEG/           
    #     ------> man_seg000.tif

    #paths for images, masks (this is the same for all datasets I think?, not just pancreatic)
    image_path = os.path.join(root_dir, sample_id, f"t{time_idx:03d}.tif")
    mask_path = os.path.join(root_dir, f"{sample_id}_ST", "SEG", f"man_seg{time_idx:03d}.tif") 

    with rasterio.open(image_path) as src:
        image = src.read(1)
    with rasterio.open(mask_path) as src:
        mask = src.read(1)
        
    return image, mask

def preprocess_data(root_dir, subsamples, timepoints) -> tuple[np.ndarray, np.ndarray]:
    #timepoints is the range of time indices to include, see the sample call
    images, masks = [], []
    
    for sample in subsamples:
        for time in timepoints:
            img, mask = load_image_mask_pair(root_dir, sample, time)
            
            img = img.astype(np.float32) / (img.max() or 1)  #normalize
            binary_mask = (mask > 0).astype(np.uint8)
            one_hot_mask = np.stack([1 - binary_mask, binary_mask], axis=-1)
            
            images.append(img[..., np.newaxis])
            masks.append(one_hot_mask)
    return np.stack(images), np.stack(masks)

################## sample usage ################## 

def get_batch(images, masks, batch_size = 4,augment = False):
    n = len(images)
    idxs = np.arange(n)
    while True:
        np.random.shuffle(idxs)
        for start in range(0, n, batch_size):
            batch_idx = idxs[start:start+batch_size]
            Xb, Yb = [], []
            for i in batch_idx:
                img = images[i].copy()
                msk = masks[i].copy()
                if augment:
                    itmd = msk[...,1]
                    img, itmd = elastic_deformation(img, itmd)
                    img, itmd = random_rotate_flip(img, itmd)
                    msk = np.stack([1 - itmd, itmd], axis=-1)
                Xb.append(img)
                Yb.append(msk)
            yield np.stack(Xb), np.stack(Yb)

###### sample call ######
root_dir = './data2/pancreatic_cells'
sample_number = ['01']
subsamples = list(range(0, 300, 10)) #change last argument to play with subsample


if __name__ == '__main__':

    X, Y = preprocess_data(root_dir, sample_number, subsamples)
    print('# samples:', len(X), 'input:', X.shape[1:], 'mask:', Y.shape[1:])

    idx = np.arange(len(X))
    np.random.shuffle(idx)
    pivot = int(0.8 * len(idx))
    train_i, test_i = idx[:pivot], idx[pivot:]
    X_train, Y_train = X[train_i], Y[train_i]
    X_test, Y_test = X[test_i], Y[test_i]
    print('training on', len(X_train), 'samples, testing on', len(X_test))

    train = get_batch(X_train, Y_train, 4, True)
    test = get_batch(X_test, Y_test, 4, False)

    xb, yb = next(train)
    print('bx shape:', xb.shape, 'yshape:', yb.shape)










