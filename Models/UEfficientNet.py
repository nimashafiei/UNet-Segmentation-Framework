import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, BatchNormalization, 
    Dropout, concatenate, Add, LeakyReLU, Resizing
)
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.models import Model

class ConvolutionBlock(tf.keras.layers.Layer):
    def __init__(self, filters, size, strides=(1, 1), padding='same', activation=True):
        """
        A convolution block with BatchNormalization and LeakyReLU activation.
        
        :param filters: Number of filters in the Conv2D layer.
        :param size: Kernel size for the Conv2D layer.
        :param strides: Strides for the Conv2D layer.
        :param padding: Padding type ('same' or 'valid').
        :param activation: Boolean indicating whether to apply activation.
        """
        super(ConvolutionBlock, self).__init__()
        self.conv = Conv2D(filters, size, strides=strides, padding=padding)
        self.batch_norm = BatchNormalization()
        self.activation = LeakyReLU(alpha=0.1) if activation else None

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.batch_norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, filters):
        """
        A residual block with convolution layers and a skip connection.
        
        :param filters: Number of filters for the convolution layers.
        """
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvolutionBlock(filters, (3, 3))
        self.conv2 = ConvolutionBlock(filters, (3, 3), activation=False)

    def call(self, inputs):
        skip_connection = BatchNormalization()(inputs)
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = Add()([x, skip_connection])  # Skip connection
        return x

class UEfficientNet(tf.keras.Model):
    def __init__(self, input_shape=(512, 512, 3), dropout_rate=0.3, start_neurons=32):
        """
        EfficientNet-based U-Net model.
        
        :param input_shape: Shape of the input image.
        :param dropout_rate: Dropout rate.
        :param start_neurons: Number of initial filters.
        """
        super(UEfficientNet, self).__init__()
        self.backbone = EfficientNetB4(weights='imagenet', include_top=False, input_shape=input_shape)
        self.start_neurons = start_neurons
        self.dropout_rate = dropout_rate

        # Encoder Layers
        self.conv4 = self.backbone.get_layer(index=342).output
        self.conv3 = self.backbone.get_layer(index=154).output
        self.conv2 = self.backbone.get_layer(index=92).output
        self.conv1 = self.backbone.get_layer(index=30).output

        # Middle Layers
        self.middle_conv = Conv2D(start_neurons * 32, (3, 3), padding="same")
        self.middle_residual1 = ResidualBlock(start_neurons * 32)
        self.middle_residual2 = ResidualBlock(start_neurons * 32)
        self.middle_activation = LeakyReLU(alpha=0.1)

        # Decoder Layers
        self.decoder = []

        for i, filters in enumerate([16, 8, 4, 2, 1]):
            self.decoder.append({
                "deconv": Conv2DTranspose(start_neurons * filters, (3, 3), strides=(2, 2), padding="same"),
                "conv": Conv2D(start_neurons * filters, (3, 3), padding="same"),
                "residual1": ResidualBlock(start_neurons * filters),
                "residual2": ResidualBlock(start_neurons * filters),
                "dropout": Dropout(self.dropout_rate if i < 2 else 0.1),
                "activation": LeakyReLU(alpha=0.1)
            })

        self.final_output = Conv2D(1, (1, 1), padding="same", activation="sigmoid")

    def build_graph(self):
        """
        Build the EfficientNet-based U-Net graph.
        
        :return: Keras Model object.
        """
        inputs = self.backbone.input

        # Encoder
        conv4 = LeakyReLU(alpha=0.1)(self.conv4)
        pool4 = MaxPooling2D((2, 2))(conv4)
        pool4 = Dropout(self.dropout_rate)(pool4)

        # Middle
        middle = self.middle_conv(pool4)
        middle = self.middle_residual1(middle)
        middle = self.middle_residual2(middle)
        middle = self.middle_activation(middle)

        # Decoder
        uconv4 = self._decoder_step(middle, conv4, self.decoder[0])

        conv3_resized = Resizing(int(uconv4.shape[1]), int(uconv4.shape[2]))(self.conv3)
        uconv3 = self._decoder_step(uconv4, conv3_resized, self.decoder[1])

        conv2_resized = Resizing(int(uconv3.shape[1]), int(uconv3.shape[2]))(self.conv2)
        uconv2 = self._decoder_step(uconv3, conv2_resized, self.decoder[2])

        conv1_resized = Resizing(int(uconv2.shape[1]), int(uconv2.shape[2]))(self.conv1)
        uconv1 = self._decoder_step(uconv2, conv1_resized, self.decoder[3])

        # Final Layer
        uconv0 = self._decoder_step(uconv1, uconv1, self.decoder[4])

        # Ensure final output has shape (512, 512, 1)
        uconv0 = Resizing(512, 512)(uconv0)
        output = self.final_output(uconv0)

        return Model(inputs=inputs, outputs=output)

    def _decoder_step(self, upsample_input, skip_connection, decoder_layer):
        """
        Perform one decoding step, which includes upsampling, concatenation with skip connections, residual blocks, and dropout.
        
        :param upsample_input: Input tensor for the upsampling.
        :param skip_connection: Skip connection from the encoder.
        :param decoder_layer: Dictionary with the components for the decoder layer.
        :return: Decoded tensor.
        """
        upsample = decoder_layer["deconv"](upsample_input)
        concat = concatenate([upsample, skip_connection])
        concat = decoder_layer["dropout"](concat)
        concat = decoder_layer["conv"](concat)
        concat = decoder_layer["residual1"](concat)
        concat = decoder_layer["residual2"](concat)
        return decoder_layer["activation"](concat)
