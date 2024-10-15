import tensorflow as tf
from tensorflow.keras.layers import (
     GlobalAveragePooling2D, Dense
)

class ChannelAttention(tf.keras.layers.Layer):
    def __init__(self, filters, ratio=8, **kwargs):
        """
        Channel Attention Mechanism.
        
        :param filters: Number of filters in the input tensor.
        :param ratio: Reduction ratio for the attention mechanism.
        """
        super(ChannelAttention, self).__init__(**kwargs)
        self.filters = filters
        self.ratio = ratio

        # Global average pooling
        self.global_avg_pool = GlobalAveragePooling2D()

        # Fully connected layers to calculate attention weights
        self.fc1 = Dense(filters // ratio, activation='relu')
        self.fc2 = Dense(filters, activation='sigmoid')