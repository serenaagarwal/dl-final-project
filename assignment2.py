import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam  
from sklearn.model_selection import train_test_split
from preprocess import preprocess_data
from unet2 import unet_model
import matplotlib.pyplot as plt

# random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

#################### Metrics and Loss Functions #####################

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    # we no longer use this for loss - just model evaluation and determining which is best 
    y_true = tf.cast(y_true, y_pred.dtype)
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    union = tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f)
    return (2. * intersection + smooth) / (union + smooth)

def weighted_binary_crossentropy(y_true, y_pred, pos_weight=10.0):
    """higher weight for positive class (white - cells)"""
    y_true = tf.cast(y_true, y_pred.dtype)
    
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
    
    weight_vector = y_true * pos_weight + (1 - y_true)
    
    bce = y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred)
    weighted_bce = -tf.reduce_mean(weight_vector * bce)
    
    return weighted_bce

# def dice_loss(y_true, y_pred):
#     return 1 - dice_coefficient(y_true, y_pred)

def tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3, smooth=1e-6):
    """version of Dice loss that allows different weights for false positives and false negatives"""
    y_true = tf.cast(y_true, y_pred.dtype)
    y_true_flat = tf.reshape(y_true, [-1])
    y_pred_flat = tf.reshape(y_pred, [-1])
    
    # true positives, false negatives and false positives
    tp = tf.reduce_sum(y_true_flat * y_pred_flat)
    fn = tf.reduce_sum(y_true_flat * (1 - y_pred_flat))
    fp = tf.reduce_sum((1 - y_true_flat) * y_pred_flat)
    
    # this is called the Tversky loss https://medium.com/@saba99/tversky-loss-902f5f8cc35f
    tversky = (tp + smooth) / (tp + alpha * fn + beta * fp + smooth)
    return 1 - tversky

def combined_loss(y_true, y_pred, alpha=0.5, beta=0.5, pos_weight=10.0):
    weighted_bce = weighted_binary_crossentropy(y_true, y_pred, pos_weight)
    tversky = tversky_loss(y_true, y_pred)
    return alpha * weighted_bce + beta * tversky

# def bce_dice_loss(y_true, y_pred):
#     """combination of BCE and Dice loss"""
#     bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
#     dice = dice_loss(y_true, y_pred)
#     return bce + dice

# -------- Training and Evaluation --------

def evaluate_baseline_vs_model(y_true, predictions, threshold=0.5):
    """Compare model performance against baselines"""
    binary_preds = (predictions > threshold).astype(np.float32)
    
    dice_model = np.mean([dice_coef_np(y_true[i], binary_preds[i]) for i in range(len(y_true))])
    
    # All black baseline
    all_black = np.zeros_like(y_true)
    dice_black = np.mean([dice_coef_np(y_true[i], all_black[i]) for i in range(len(y_true))])
    
    # class distribution in ground truth
    pos_ratio = np.mean(y_true)
    
    print(f"\nEvaluation Results:")
    print(f"Model Dice score: {dice_model:.4f}")
    print(f"All-black baseline Dice score: {dice_black:.4f}")
    print(f"Positive class ratio in ground truth: {pos_ratio:.4f}")
    
    if dice_model <= dice_black:
        print("WARNING: Model performs worse than or equal to all-black baseline!")
    else:
        improvement = (dice_model - dice_black) / (max(dice_black, 1e-6))
        print(f"Model improves over all-black baseline by {improvement:.2f}x")
    
    return dice_model, dice_black

def dice_coef_np(y_true, y_pred, smooth=1e-6):
    #i lied i use this a lot 
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def find_optimal_threshold(y_true, y_pred):
    """threshold for converting probability maps to binary masks"""
    thresholds = np.linspace(0.1, 0.9, 9)
    dice_scores = []
    
    for threshold in thresholds:
        binary_preds = (y_pred > threshold).astype(np.float32)
        dice = np.mean([dice_coef_np(y_true[i], binary_preds[i]) for i in range(len(y_true))])
        dice_scores.append(dice)
        print(f"Threshold {threshold:.1f} - Dice: {dice:.4f}")
    
    best_idx = np.argmax(dice_scores)
    best_threshold = thresholds[best_idx]
    best_dice = dice_scores[best_idx]
    
    print(f"Optimal threshold: {best_threshold:.2f} with Dice score: {best_dice:.4f}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, dice_scores, 'o-')
    plt.axvline(x=best_threshold, color='r', linestyle='--')
    plt.title('Dice Score vs Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('Dice Score')
    plt.grid(True)
    plt.savefig('threshold_optimization.png')
    plt.close()
    
    return best_threshold

# ################################################################

def main():
    # 
    root_dir = './data2/pancreatic_cells'
    samples = ['01', '02']
    timepoints = list(range(0, 300, 5))  # Take every 5th frame
    batch_size = 8
    epochs = 50
    learning_rate = 1e-4
    
    print("Loading and preprocessing data...")
    X, Y = preprocess_data(root_dir, samples, timepoints)
    print(f"Loaded {X.shape[0]} samples with shape {X.shape[1:]}")
    

    if Y.shape[-1] > 1:
        Y = Y[..., 1:2]  
        
    positive_ratio = np.mean(Y)
    print(f"Class balance - Positive class: {positive_ratio:.4f}, Negative class: {1-positive_ratio:.4f}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42, shuffle=True
    )
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Calc class weight based on imbalance
    pos_samples = np.sum(y_train > 0.5) 
    neg_samples = np.sum(y_train <= 0.5)
    pos_class_weight = neg_samples / (pos_samples + 1e-6)  # no division by zero
    pos_class_weight = min(max(pos_class_weight, 1.0), 30.0)  # bound weight range
    
    print(f"Class statistics - Positive: {pos_samples}, Negative: {neg_samples}")
    print(f"Positive class weight: {pos_class_weight:.2f}")
    
    input_shape = X_train.shape[1:]
    model = unet_model(input_shape)
    
    def custom_loss(y_true, y_pred):
        return combined_loss(y_true, y_pred, pos_weight=pos_class_weight)
    
    model.compile(
        optimizer=Adam(learning_rate),
        loss=custom_loss,
        metrics=[dice_coefficient, 'binary_accuracy', 
                tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    model.summary()
    
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_dice_coefficient', factor=0.5, patience=5, 
            min_lr=1e-6, verbose=1, mode='max'
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_dice_coefficient', patience=10, 
            restore_best_weights=True, mode='max', verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath='best_cell_segmentation_model.h5',
            monitor='val_dice_coefficient',
            mode='max',
            save_best_only=True,
            verbose=1
        )
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    model.save('cell_segmentation_model', save_format='tf')
    print("Model saved successfully!")
    
    print("Making predictions...")
    predictions = model.predict(X_test)
    
    best_threshold = find_optimal_threshold(y_test, predictions)
    
    binary_predictions = (predictions > best_threshold).astype(np.float32)
    
    evaluate_baseline_vs_model(y_test, binary_predictions)
    
    print("Model training and eval complete!")

if __name__ == '__main__':
    main()