import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import cv2
import glob
from unet import UNet 

np.random.seed(42)
tf.random.set_seed(42)

def load_dataset(images_path, masks_path):
    image_files = sorted(glob.glob(os.path.join(images_path, "*.jpg")))
    
    images = []
    masks = []
    
    for image_file in image_files:
        base_name = os.path.basename(image_file).split('.')[0]
        
        mask_file = os.path.join(masks_path, f"{base_name}.png")
        
        if os.path.exists(mask_file):
            img = cv2.imread(image_file)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
            
            mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)  # Read as grayscale
            
            images.append(img)
            masks.append(mask)
    
    print(f"Loaded {len(images)} image-mask pairs")
    return np.array(images), np.array(masks)

def preprocess_data(images, masks, img_size=(256, 256)):
    processed_images = []
    processed_masks = []
    
    for i in range(len(images)):
        img = cv2.resize(images[i], img_size)
        img = img / 255.0
        
        mask = cv2.resize(masks[i], img_size)
        mask = (mask > 0).astype(np.float32)
        
        processed_images.append(img)
        processed_masks.append(mask)
    
    processed_images = np.array(processed_images, dtype=np.float32)
    processed_masks = np.array(processed_masks, dtype=np.float32)
    
    if len(processed_masks.shape) == 3:
        processed_masks = np.expand_dims(processed_masks, axis=-1)
    
    return processed_images, processed_masks

def dice_coefficient(y_true, y_pred, smooth=1e-6):
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


def main():
    train_images_path = "BloodCellData/train/original"
    train_masks_path = "BloodCellData/train/mask"

    test_images_path = "BloodCellData/test/original"
    test_masks_path = "BloodCellData/test/mask"
    
    img_size = (256, 256)
    batch_size = 8  
    epochs = 20
    learning_rate = 1e-4
    
    print("Loading training dataset...")
    train_images, train_masks = load_dataset(train_images_path, train_masks_path)
    
    print("Loading testing dataset...")
    test_images, test_masks = load_dataset(test_images_path, test_masks_path)
    
    print("Preprocessing training data...")
    train_images, train_masks = preprocess_data(train_images, train_masks, img_size)
    
    print("Preprocessing testing data...")
    test_images, test_masks = preprocess_data(test_images, test_masks, img_size)
    
    train_pos_ratio = np.mean(train_masks)
    test_pos_ratio = np.mean(test_masks)
    print(f"Training data - Positive pixel ratio: {train_pos_ratio:.4f}")
    print(f"Testing data - Positive pixel ratio: {test_pos_ratio:.4f}")
    
    input_shape = train_images[0].shape
    num_channels = input_shape[-1]
    
    print("Creating model...")
    model = UNet(num_channels=num_channels, num_classes=1)
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),  #decide how to change this if we want to use BCE + Dice loss or the weighted thing in the orig paper
        metrics=[dice_coefficient, 'binary_accuracy']
    )

    
    print("Training model...")
    model.fit(
        train_images, train_masks,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(test_images, test_masks)
    )
    
    
    model.save('final_unet_model', save_format='tf')

    #trying to save segmentations
    predictions = model.predict(test_images)
    predictions = (predictions > 0.5).astype(np.uint8) #binarizing

    save_dir = "predicted masks"
    os.makedirs(save_dir, exist_ok=True)

    for i, pred_mask in enumerate(predictions): 
        pred_mask = np.squeeze(pred_mask)
        save_path = os.path.join(save_dir, f"mask_{i:03d}.png")
        cv2.imwrite(save_path, pred_mask * 255)
    print(f"Saved {len(predictions)} predicted masks to '{save_dir}")

    print("Model saved successfully!")

if __name__ == "__main__":
    main()