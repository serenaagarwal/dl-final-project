import tensorflow as tf
import numpy as np

class DoubleConv(tf.keras.layers.Layer):
    #convolution->batch normalization->relu * 2
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = tf.keras.Sequential([
            tf.keras.layers.Conv2D(mid_channels, kernel_size=3, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Conv2D(out_channels, kernel_size=3, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dropout(0.2)
        ])

    def call(self, x):
        return self.double_conv(x)
    
class Downscale(tf.keras.layers.Layer):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = tf.keras.Sequential([
            tf.keras.layers.MaxPool2D(2),
            DoubleConv(in_channels, out_channels)
        ])
        
    def call(self, x):
        return self.maxpool_conv(x)

class Upscale(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels): #I am making the default bilinear for upsampling because I think it's best for our data. Subject to change. -SA
        super().__init__()
        self.up = tf.keras.layers.UpSampling2D((2, 2), interpolation='bilinear')
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
    
    def call(self, x1, x2): #x1 is the matching 2d matrix on the other side, which is being concatenated to x2. - SA
        x1 = self.up(x1)
        #compare sizes between x1 and x2:
        diffX = x2.shape[2] - x1.shape[2] #compare widths
        diffY = x2.shape[1] - x1.shape[1] #compare heights

        #if x1 and x2 aren't the same, adds padding to x1 before concatenation. This part is pretty different from medium post - SA 
        if diffY != 0 or diffX != 0:
            x1 = tf.pad(x1, [
                [0, 0],
                [diffY // 2, diffY - diffY // 2],
                [diffX // 2, diffX - diffX // 2],
                [0, 0]
            ])
        
        x = tf.concat([x2, x1], axis=-1)
        return self.conv(x)
    
class OutConv(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = tf.keras.layers.Conv2D(out_channels, kernel_size=1)

    def call(self, x):
        return self.conv(x)

