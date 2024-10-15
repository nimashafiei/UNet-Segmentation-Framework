import tensorflow as tf
from tensorflow.keras import layers, Model
from PVMLayer import PVMLayer
from ChannelAttBridge import ChannelAttBridge

class UNetPVMChannelAttModel:
    def __init__(self, img_height, img_width, width=64):
        """
        Initialize the U-Net model with PVM layers and Channel Attention Bridge (CAB).
        
        :param img_height: Height of the input images.
        :param img_width: Width of the input images.
        :param width: Number of filters for the first convolution layer.
        """
        self.img_height = img_height
        self.img_width = img_width
        self.width = width

    def conv_block(self, inputs, filters, dropout_rate=0, batch_norm=True):
        """
        Convolutional block with Conv2D, BatchNorm, and Dropout.
        
        :param inputs: Input tensor to the block.
        :param filters: Number of filters for the Conv2D layers.
        :param dropout_rate: Dropout rate.
        :param batch_norm: Whether to apply batch normalization.
        :return: Output tensor after the convolution block.
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
        Build the U-Net model with PVM layers and Channel Attention Bridge.
        
        :return: Compiled Keras Model.
        """
        inputs = layers.Input((self.img_height, self.img_width, 3))

        # Encoder
        conv1 = self.conv_block(inputs, self.width)
        conv1 = PVMLayer(self.width, self.width)(conv1)  # PVM layer
        pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = self.conv_block(pool1, self.width * 2)
        conv2 = PVMLayer(self.width * 2, self.width * 2)(conv2)
        pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = self.conv_block(pool2, self.width * 4)
        conv3 = PVMLayer(self.width * 4, self.width * 4)(conv3)
        pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = self.conv_block(pool3, self.width * 8)
        conv4 = PVMLayer(self.width * 8, self.width * 8)(conv4)
        pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = self.conv_block(pool4, self.width * 16)
        conv5 = PVMLayer(self.width * 16, self.width * 16)(conv5)

        # Channel Attention Bridge
        att_bridge = ChannelAttBridge([self.width, self.width * 2, self.width * 4, self.width * 8, self.width * 16])
        conv1, conv2, conv3, conv4, conv5 = att_bridge([conv1, conv2, conv3, conv4, conv5])

        # Decoder
        up6 = self.upsample_concat(conv5, conv4)
        conv6 = PVMLayer(self.width * 8, self.width * 8)(up6)

        up7 = self.upsample_concat(conv6, conv3)
        conv7 = PVMLayer(self.width * 4, self.width * 4)(up7)

        up8 = self.upsample_concat(conv7, conv2)
        conv8 = PVMLayer(self.width * 2, self.width * 2)(up8)

        up9 = self.upsample_concat(conv8, conv1)
        conv9 = PVMLayer(self.width, self.width)(up9)

        # Output
        output = layers.Conv2D(1, (1, 1), activation='sigmoid')(conv9)

        model = Model(inputs=inputs, outputs=output)
        return model
