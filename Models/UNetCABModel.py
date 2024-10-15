import tensorflow as tf
from tensorflow.keras import layers, Model
from ChannelAttBridge import ChannelAttBridge

class UNetCABModel:
    def __init__(self, input_size=(512, 512, 3), initial_filters=64):
        """
        Initialize the U-Net with Channel Attention Bridge (CAB).
        
        :param input_size: Input shape of the model.
        :param initial_filters: Number of filters for the first convolution layer.
        """
        self.input_size = input_size
        self.initial_filters = initial_filters

    def conv_block(self, inputs, filters, dropout_rate=0, batch_norm=True):
        """
        Create a convolutional block consisting of Conv2D layers, BatchNorm, and Dropout.
        
        :param inputs: Input tensor to the block.
        :param filters: Number of filters for the Conv2D layers.
        :param dropout_rate: Dropout rate.
        :param batch_norm: Whether to use BatchNormalization.
        :return: Output tensor after the convolutional block.
        """
        conv = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(inputs)
        if batch_norm:
            conv = layers.BatchNormalization()(conv)
        conv = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(conv)
        if dropout_rate > 0:
            conv = layers.SpatialDropout2D(dropout_rate)(conv)
        return conv

    def upsample_concat(self, upsampled_input, skip_connection):
        """
        Upsample the input and concatenate it with the corresponding encoder feature map.
        
        :param upsampled_input: Input tensor to upsample.
        :param skip_connection: Skip connection from the encoder path.
        :return: Concatenated tensor after upsampling and concatenation.
        """
        upsampled = layers.UpSampling2D(size=(2, 2))(upsampled_input)
        return layers.Concatenate()([upsampled, skip_connection])

    def build_model(self):
        """
        Build the U-Net model with Channel Attention Bridge.
        
        :return: Keras Model object.
        """
        inputs = layers.Input(self.input_size)
        width = self.initial_filters

        # Encoder path
        conv1 = self.conv_block(inputs, width)
        pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = self.conv_block(pool1, width * 2)
        pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = self.conv_block(pool2, width * 4)
        pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = self.conv_block(pool3, width * 8)
        pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = self.conv_block(pool4, width * 16)

        # Apply Channel Attention Bridge
        channel_sizes = [width, width * 2, width * 4, width * 8, width * 16]
        bridge = ChannelAttBridge(c_list=channel_sizes, attention_type='fc')
        conv1, conv2, conv3, conv4, conv5 = bridge([conv1, conv2, conv3, conv4, conv5])

        # Decoder path
        up6 = self.upsample_concat(conv5, conv4)
        conv6 = self.conv_block(up6, width * 8, dropout_rate=0.35)

        up7 = self.upsample_concat(conv6, conv3)
        conv7 = self.conv_block(up7, width * 4, dropout_rate=0.35)

        up8 = self.upsample_concat(conv7, conv2)
        conv8 = self.conv_block(up8, width * 2, dropout_rate=0.35)

        up9 = self.upsample_concat(conv8, conv1)
        conv9 = self.conv_block(up9, width, dropout_rate=0.35)

        # Output layer
        output = layers.Conv2D(1, (1, 1), activation='sigmoid')(conv9)

        # Build the model
        return Model(inputs=inputs, outputs=output)
