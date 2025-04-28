import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

def elastic_deformation(image, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # generates random displacement fields, applys Gaussian smoothing, and maps original pixels to
    # bicubic interpolation, nearest-neighbor interpolation (refer to Unet paper)
    if image.ndim == 3:
        image = image.squeeze(axis=-1)
    
    shape = image.shape
    rng = np.random.RandomState()

    displacement = rng.rand(2, *shape) * 2 - 1  # values are between 1, 1 for random displacements
    displacement = gaussian_filter(displacement, sigma=(30, 30, 0), mode='constant') * 1000 #apply gaussian filter
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    coordinates = [y + displacement[0],x + displacement[1]]
    
    deformed_image = map_coordinates(image, coordinates, order=3).reshape(shape) #chose cubic for image?
    deformed_mask = map_coordinates(mask, coordinates, order=0).reshape(shape)
    
    return deformed_image[..., np.newaxis], deformed_mask #added dim removed

def random_rotate_flip(image, mask) -> tuple[np.ndarray, np.ndarray]:
    k = np.random.choice([0, 2])
    i = np.rot90(image, k, axes=(0,1))
    m = np.rot90(mask, k, axes=(0,1))
    if np.random.rand() > 0.5:
        i = np.fliplr(i)
        m = np.fliplr(m)
    return i, m