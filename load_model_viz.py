import os
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from preprocess import preprocess_data
from sklearn.model_selection import train_test_split

# Configure this to match your original data loading
root_dir = './data2/pancreatic_cells'
samples = ['01', '02']
subsamples = list(range(0, 300, 5))

def visualize_results():
    print("Loading data...")
    # Load data the same way as in training
    X, Y = preprocess_data(root_dir, samples, subsamples)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, shuffle=True)
    
    if y_test.shape[-1] > 1:
        y_test_binary = y_test[..., 1]  # Get just the foreground channel and remove extra dimension
    else:
        y_test_binary = y_test[..., 0]
    
    print("Loading model...")
    try:
        model = tf.keras.models.load_model('unet_pancreatic', 
                                          custom_objects={
                                              'dice_coefficient': dice_coefficient,
                                              'bce_dice_loss': bce_dice_loss
                                          })
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    print("Generating predictions...")
    # Generate predictions
    predictions = model.predict(X_test)
    binary_predictions = (predictions > 0.5).astype(np.uint8)
    
    save_dir = "visualization_results"
    os.makedirs(save_dir, exist_ok=True)
    
    # Visualize a few examples
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
        plt.imshow(y_test_binary[i], cmap='viridis')
        plt.axis('off')
        
        # Predicted mask
        plt.subplot(1, 3, 3)
        plt.title("Predicted Mask")
        plt.imshow(binary_predictions[i, ..., 0], cmap='viridis')
        plt.axis('off')
        
        plt.savefig(os.path.join(save_dir, f"comparison_{i}.png"), bbox_inches='tight')
        plt.close()
        
        # save individual masks as PNGs
        cv2.imwrite(os.path.join(save_dir, f"pred_mask_{i}.png"), binary_predictions[i, ..., 0] * 255)
    
    print(f"Saved {num_examples} visualization images to '{save_dir}'")
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    y_true = y_test_binary.flatten()
    y_pred = binary_predictions.flatten()
    
    print("\nSegmentation Metrics:")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"F1 Score: {f1_score(y_true, y_pred, zero_division=0):.4f}")

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, y_pred.dtype)
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    union = tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f)
    return (2. * intersection + smooth) / (union + smooth)

def dice_loss(y_true, y_pred):
    """Calculate Dice loss = 1 - dice coefficient"""
    return 1 - dice_coefficient(y_true, y_pred)

def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return bce + dice

if __name__ == "__main__":
    visualize_results()