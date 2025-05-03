import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import rasterio
from sklearn.model_selection import train_test_split
import plots as plot
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, y_pred.dtype)
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    union = tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f)
    return (2. * intersection + smooth) / (union + smooth)

def weighted_binary_crossentropy(y_true, y_pred, pos_weight=10.0):
    y_true = tf.cast(y_true, y_pred.dtype)
    
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
    
    weight_vector = y_true * pos_weight + (1 - y_true)
    
    bce = y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred)
    weighted_bce = -tf.reduce_mean(weight_vector * bce)
    
    return weighted_bce

def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

def tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3, smooth=1e-6):
    y_true = tf.cast(y_true, y_pred.dtype)
    y_true_flat = tf.reshape(y_true, [-1])
    y_pred_flat = tf.reshape(y_pred, [-1])
    
    # true positives, false negatives and false positives
    tp = tf.reduce_sum(y_true_flat * y_pred_flat)
    fn = tf.reduce_sum(y_true_flat * (1 - y_pred_flat))
    fp = tf.reduce_sum((1 - y_true_flat) * y_pred_flat)
    
    tversky = (tp + smooth) / (tp + alpha * fn + beta * fp + smooth)
    return 1 - tversky

def combined_loss(y_true, y_pred, alpha=0.5, beta=0.5, pos_weight=10.0):
    """Combined loss function with weighted BCE and Tversky loss"""
    weighted_bce = weighted_binary_crossentropy(y_true, y_pred, pos_weight)
    tversky = tversky_loss(y_true, y_pred)
    return alpha * weighted_bce + beta * tversky

# Data preprocessing functions
def load_image_mask_pair(root_dir, sample_id, time_idx):
    """Load an image and its corresponding mask"""
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

def preprocess_data(root_dir, samples, timepoints):
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
            
            # Add channel dimension
            images.append(img_norm[..., np.newaxis])
            masks.append(binary_mask[..., np.newaxis])
    
    if not images:
        raise ValueError("No valid images loaded.")
        
    return np.array(images), np.array(masks)

def main():
    root_dir = './data/pancreatic_cells'
    samples = ['01', '02']
    timepoints = list(range(0, 300, 5))  # Take every 5th frame
    
    print("Loading and preprocessing data...")
    X, Y = preprocess_data(root_dir, samples, timepoints)
    print(f"Loaded {X.shape[0]} samples with shape {X.shape[1:]}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42, shuffle=True
    )
    print(f"Test set: {X_test.shape[0]} samples")
    
    y_test_binary = y_test[..., 0] if y_test.ndim > 3 else y_test
    
    print("Loading model...")
    def custom_loss(y_true, y_pred):
        return combined_loss(y_true, y_pred, pos_weight=10.0)  # pos_weight from original
    
    try:
        model = tf.keras.models.load_model(
            'best_cell_segmentation_model.h5',
            custom_objects={
                'dice_coefficient': dice_coefficient,
                'combined_loss': combined_loss,
                'dice_loss': dice_loss,
                'tversky_loss': tversky_loss,
                'weighted_binary_crossentropy': weighted_binary_crossentropy,
                'custom_loss': custom_loss  # Add the custom_loss function
            }
        )
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
    
    print("Generating predictions...")
    predictions = model.predict(X_test)
    
    # Find best threshold
    thresholds = np.linspace(0.1, 0.9, 9)
    dice_scores = []
    
    for threshold in thresholds:
        binary_preds = (predictions > threshold).astype(np.float32)
        dice = np.mean([dice_coef_np(y_test[i, ..., 0], binary_preds[i, ..., 0]) for i in range(len(y_test))])
        dice_scores.append(dice)
        print(f"Threshold {threshold:.1f} - Dice: {dice:.4f}")
    
    best_idx = np.argmax(dice_scores)
    best_threshold = thresholds[best_idx]
    print(f"Using optimal threshold: {best_threshold:.2f}")
    
    binary_predictions = (predictions > best_threshold).astype(np.uint8)
    
    # directory for results
    save_dir = "visualization_results"
    os.makedirs(save_dir, exist_ok=True)
    
    # examples
    num_examples = min(10, len(X_test))
    for i in range(num_examples):
        plt.figure(figsize=(15, 5))
        
        # Original image
        plt.subplot(1, 3, 1)
        plt.title("Original Image")
        plt.imshow(X_test[i, ..., 0], cmap='gray')
        plt.axis('off')
        
        # Ground truth mask
        plt.subplot(1, 3, 2)
        plt.title("Ground Truth Mask")
        plt.imshow(y_test[i, ..., 0], cmap='viridis')
        plt.axis('off')
        
        # Predicted mask
        plt.subplot(1, 3, 3)
        plt.title("Predicted Mask")
        plt.imshow(binary_predictions[i, ..., 0], cmap='viridis')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'prediction_{i}.png'))
        plt.close()
    
    ### additional plots
    plot.plot_precision_recall(  y_test[...,0], predictions[...,0] )
    plot.plot_roc_curve(       y_test[...,0], predictions[...,0] )
    best_tau = plot.plot_dice_vs_threshold( y_test[...,0], predictions[...,0] )
    plot.save_triplet( X_test[0,...,0], y_test[0,...,0], (predictions[0,...,0]>best_tau).astype(np.uint8) )

def dice_coef_np(y_true, y_pred, smooth=1e-6):
    #just for evaluation purposes because the orig method causes Cannot convert 2.0 to EagerTensor of dtype uint8
    # and i dont want to touch it 
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

if __name__ == '__main__':
    main()