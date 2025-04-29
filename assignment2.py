import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from preprocess import preprocess_data
from unet import UNet

np.random.seed(42)
tf.random.set_seed(42)

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
    root_dir      = './data2/pancreatic_cells'
    samples       = ['01', '02']
    subsamples    = list(range(0, 300, 5)) # last arg means we take every 5th frame (60 samples for this dataset)
    batch_size    = 8
    epochs        = 20
    learning_rate = 1e-4

    X, Y = preprocess_data(root_dir, samples, subsamples)
    X_train, X_test, y_train_onehot, y_test_onehot = train_test_split(X, Y, test_size=0.2, random_state=42, shuffle=True)

    y_train = y_train_onehot[..., 1:2]
    y_test  = y_test_onehot[..., 1:2]

    train_pos_ratio = np.mean(y_train)
    test_pos_ratio  = np.mean(y_test)

    input_shape  = X_train.shape[1:]
    model = UNet(num_channels=input_shape[-1], num_classes=1)
    model.compile(optimizer=Adam(learning_rate), loss=bce_dice_loss, metrics=[dice_coefficient, 'binary_accuracy'])
    model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=batch_size, epochs=epochs)
    model.save('unet_pancreatic', save_format='tf')
    print("model saved!!!!)

if __name__ == '__main__':
    main()
