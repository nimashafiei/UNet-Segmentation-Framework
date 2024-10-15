import tensorflow as tf
from tensorflow.keras import layers

class ChannelAttBridge(layers.Layer):
    def __init__(self, channel_sizes, attention_type='fc', **kwargs):
        """
        Initialize the Channel Attention Bridge (CAB).
        
        :param channel_sizes: List of channel sizes for each feature map from the encoder.
        :param attention_type: Type of attention to apply ('fc' for fully connected, 'conv' for convolutional).
        """
        super(ChannelAttBridge, self).__init__(**kwargs)
        self.channel_sizes = channel_sizes
        self.attention_type = attention_type

        # Create an average pooling layer
        self.avg_pool = layers.GlobalAveragePooling2D(keepdims=True)

        # Create attention layers for each feature map
        self.att_layers = [
            layers.Dense(c, activation='sigmoid') if attention_type == 'fc'
            else layers.Conv1D(c, 1, activation='sigmoid')
            for c in channel_sizes
        ]

    def call(self, encoder_outputs):
        """
        Forward pass to apply attention to the encoder outputs.
        
        :param encoder_outputs: List of feature maps from the encoder.
        :return: List of attended feature maps.
        """
        # Apply global average pooling to each encoder output
        pooled_outputs = [self.avg_pool(feature_map) for feature_map in encoder_outputs]
        concatenated_pooled = layers.Concatenate(axis=-1)(pooled_outputs)

        # Apply attention to the pooled output
        attention_weights = [att_layer(concatenated_pooled) for att_layer in self.att_layers]

        # Reshape the attention weights and apply them to the original encoder outputs
        reshaped_attentions = [
            layers.Reshape((1, 1, c))(att) for att, c in zip(attention_weights, self.channel_sizes)
        ]

        attended_outputs = [
            layers.Multiply()([att, feature_map]) for att, feature_map in zip(reshaped_attentions, encoder_outputs)
        ]

        return attended_outputs
