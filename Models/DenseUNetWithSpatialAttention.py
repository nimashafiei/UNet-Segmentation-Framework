import tensorflow as tf
from tensorflow.keras import layers as L
from tensorflow.keras.models import Model

class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, filters, dilation_rate=1, use_batch_norm=True, activation='relu'):
        """
        Basic convolution block with optional batch normalization and activation.
        
        :param filters: Number of filters.
        :param dilation_rate: Dilation rate for the Conv2D layer.
        :param use_batch_norm: Whether to use batch normalization.
        :param activation: Activation function to use.
        """
        super(ConvBlock, self).__init__()
        self.conv = L.Conv2D(filters, (3, 3), padding="same", dilation_rate=dilation_rate)
        self.batch_norm = L.BatchNormalization() if use_batch_norm else None
        self.activation = L.Activation(activation) if activation else None

    def call(self, inputs):
        x = self.conv(inputs)
        if self.batch_norm:
            x = self.batch_norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class DenseBlock(tf.keras.layers.Layer):
    def __init__(self, filters, growth_rate=4):
        """
        Dense block with concatenated convolutions for feature reuse.
        
        :param filters: Number of filters.
        :param growth_rate: Growth rate for the internal dense connections.
        """
        super(DenseBlock, self).__init__()
        self.filters = filters
        self.growth_rate = growth_rate

    def call(self, inputs):
        x1 = self._conv_block(inputs, self.filters)
        x2 = self._conv_block(x1, self.filters // self.growth_rate)
        x3 = self._conv_block(L.concatenate([inputs, x2]), self.filters // self.growth_rate)
        x4 = self._conv_block(L.concatenate([inputs, x2, x3]), self.filters // self.growth_rate)
        return L.concatenate([inputs, x2, x3, x4])

    def _conv_block(self, inputs, filters):
        x = L.Conv2D(filters, (3, 3), padding="same")(inputs)
        x = L.BatchNormalization()(x)
        x = L.Activation('relu')(x)
        return x

class SpatialAttentionBlock(tf.keras.layers.Layer):
    def __init__(self, filters):
        """
        Spatial attention block that enhances feature maps using attention mechanisms.
        
        :param filters: Number of filters.
        """
        super(SpatialAttentionBlock, self).__init__()
        self.filters = filters

    def call(self, inputs):
        conv5 = L.Conv2D(self.filters, 3, padding="same", dilation_rate=4)(inputs)
        conv5 = L.Dropout(0.5)(conv5)
        conv5 = L.BatchNormalization()(conv5)

        avg_pool = L.Lambda(lambda x: tf.reduce_mean(x, axis=3, keepdims=True))(conv5)
        max_pool = L.Lambda(lambda x: tf.reduce_max(x, axis=3, keepdims=True))(conv5)

        concat = L.Concatenate(axis=3)([avg_pool, max_pool])
        cbam_feature = L.Conv2D(self.filters, 3, padding="same", activation='sigmoid')(concat)
        conv5 = L.multiply([conv5, cbam_feature])
        
        conv5 = L.Conv2D(self.filters, 3, padding="same", dilation_rate=2)(conv5)
        conv5 = L.Dropout(0.5)(conv5)
        return L.BatchNormalization()(conv5)

class DenseUNetWithSpatialAttention:
    def __init__(self, input_size=(512, 512, 3), output_channels=1, start_neurons=8, learning_rate=1e-3):
        """
        Dense U-Net with Spatial Attention initialization.
        
        :param input_size: Input image size.
        :param output_channels: Number of output channels.
        :param start_neurons: Number of initial filters.
        :param learning_rate: Learning rate for the model.
        """
        self.input_size = input_size
        self.output_channels = output_channels
        self.start_neurons = start_neurons
        self.learning_rate = learning_rate

    def build_model(self):
        """
        Build the Dense U-Net with Spatial Attention model.
        
        :return: Keras Model object.
        """
        inputs = L.Input(self.input_size)

        # Encoder path with dense blocks and spatial attention
        conv1 = self._encoder_block(inputs, self.start_neurons * 1)
        pool1 = L.MaxPooling2D((2, 2))(conv1)

        conv2 = self._encoder_block(pool1, self.start_neurons * 2)
        pool2 = L.MaxPooling2D((2, 2))(conv2)

        conv3 = self._encoder_block(pool2, self.start_neurons * 4)
        pool3 = L.MaxPooling2D((2, 2))(conv3)

        # Bottleneck with attention
        conv5 = self._bottleneck_block(pool3, self.start_neurons * 5)

        # Decoder path with upsampling and spatial attention
        uconv3 = self._decoder_block(conv5, conv3, self.start_neurons * 4)
        uconv2 = self._decoder_block(uconv3, conv2, self.start_neurons * 2)
        uconv1 = self._decoder_block(uconv2, conv1, self.start_neurons * 1)

        # Output layer
        output_layer = L.Conv2D(self.output_channels, (3, 3), padding="same", activation='sigmoid')(uconv1)
        model = Model(inputs, output_layer)
        return model

    def _encoder_block(self, inputs, filters):
        """
        Encoder block with dense block and spatial attention.
        
        :param inputs: Input tensor.
        :param filters: Number of filters for the block.
        :return: Output tensor after the block.
        """
        dense_block = DenseBlock(filters)(inputs)
        return SpatialAttentionBlock(filters)(dense_block)

    def _bottleneck_block(self, inputs, filters):
        """
        Bottleneck block with spatial attention.
        
        :param inputs: Input tensor.
        :param filters: Number of filters for the block.
        :return: Output tensor after the block.
        """
        conv5 = L.Conv2D(filters, (3, 3), padding="same", activation='relu')(inputs)
        return SpatialAttentionBlock(filters)(conv5)

    def _decoder_block(self, inputs, skip_connection, filters):
        """
        Decoder block with upsampling, concatenation, and spatial attention.
        
        :param inputs: Input tensor for upsampling.
        :param skip_connection: Skip connection from the encoder.
        :param filters: Number of filters for the block.
        :return: Output tensor after the block.
        """
        deconv = L.Conv2DTranspose(filters, (3, 3), strides=(2, 2), padding="same")(inputs)
        uconv = L.Concatenate()([deconv, skip_connection])
        uconv = ConvBlock(filters)(uconv)
        uconv = DenseBlock(filters)(uconv)
        return SpatialAttentionBlock(filters)(uconv)
