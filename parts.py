import tensorflow as tf
import numpy as np

class DoubleConv(tf.keras.layers.Layer):
    #convolution->batch normalization->relu * 2
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = tf.keras.models.Sequential(
            tf.keras.layers.Conv2D(in_channels, mid_channels, kernel_size=3, padding=1) #can adjust kernel and padding to our dataset
            tf.keras.layers.BatchNormalization(mid_channels)
            tf.keras.activations.relu()
            tf.keras.layers.Conv2D(mid_channels, out_channels, kernel_size=3, padding=1)
            tf.keras.layers.BatchNormalization(out_channels)
            tf.keras.layers.activations.relu()
        )

    def call(self, x):
        return self.double_conv(x)
    
class Downscale(tf.keras.layers.Layer):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = tf.keras.models.Sequential(
            tf.keras.layers.MaxPool2D(2)
            DoubleConv(in_channels, out_channels)
        )

        def call(self, x):
            return self.maxpool_conv(x)

class Upscale(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels): #I am making the default bilinear for upsampling but I don't totally understand it. I will leave it for now but may change.
        super().__init__()
        self.up = 
        
