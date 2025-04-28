import tensorflow as tf
import numpy as np
from parts import *

class UNet(tf.keras.Model):
    def __init__(self, num_channels=3, num_classes = 1):
        super(UNet, self).__init__()
        #definitely possible to reduce number of layers and simplify if computation is too expensive -SA
        self.num_channels = num_channels
        self.num_classes = num_classes

        self.input_conv = DoubleConv(num_channels, 64)
        self.downscale1 = Downscale(64, 128)
        self.downscale2 = Downscale(128, 256)
        self.downscale3 = Downscale(256, 512)
        self.downscale4 = Downscale(512, 512)
        self.upscale1 = Upscale(1024, 256)
        self.upscale2 = Upscale(512, 128)
        self.upscale3 = Upscale(256, 64)
        self.upscale4 = Upscale(128, 64)
        self.output_conv = OutConv(64, num_classes)

    def call(self, x):
        x1 = self.input_conv(x)
        x2 = self.downscale1(x1)
        x3 = self.downscale2(x2)
        x4 = self.downscale3(x3)
        x5 = self.downscale4(x4)
        x = self.upscale1(x5, x4)
        x = self.upscale2(x, x3)
        x = self.upscale3(x, x2)
        x = self.upscale4(x, x1)
        logits = self.output_conv(x)
        outputs = tf.keras.activations.sigmoid(logits)
        return outputs





