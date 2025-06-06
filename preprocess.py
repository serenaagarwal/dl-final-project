import rasterio #good option per this post: https://stackoverflow.com/questions/68020419/working-with-tiff-images-in-python-for-deep-learning
#it does produce some error message but it's okay to ignore
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

    try:
        with rasterio.open(image_path) as src:
            image = src.read(1)
        with rasterio.open(mask_path) as src:
            mask = src.read(1)
        return image, mask
    except Exception as e:
        print(f"Error loading {image_path} or {mask_path}: {e}")
        return None, None

def preprocess_data(root_dir, samples, timepoints) -> tuple[np.ndarray, np.ndarray]:
    #timepoints is the range of time indices to include, see the sample call
    images, masks = [], []
    
    for sample in samples:
        for time in timepoints:
            img, mask = load_image_mask_pair(root_dir, sample, time)
            if img is None or mask is None:
                continue
            
            # Enhance contrast 
            p_low, p_high = np.percentile(img, [2, 98])
            img_norm = np.clip((img - p_low) / (p_high - p_low + 1e-6), 0, 1)
            img_norm = img_norm.astype(np.float32)
            
            
            binary_mask = (mask > 0).astype(np.float32)
            
            images.append(img_norm[..., np.newaxis])
            masks.append(binary_mask[..., np.newaxis])
    
    if not images:
        raise ValueError("No valid images loaded.")
        
    return np.array(images), np.array(masks)


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