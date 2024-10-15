import tensorflow as tf
from tensorflow.keras import layers, models

class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, filters, filter_size, dropout_rate=0.0, batch_norm=True):
        """
        Convolutional block with optional Batch Normalization and Dropout.
        
        :param filters: Number of filters in Conv2D layers.
        :param filter_size: Size of the convolutional filter.
        :param dropout_rate: Dropout rate.
        :param batch_norm: Whether to apply Batch Normalization.
        """
        super(ConvBlock, self).__init__()
        self.batch_norm = batch_norm
        self.conv1 = layers.Conv2D(filters, filter_size, padding="same", activation=None)
        self.conv2 = layers.Conv2D(filters, filter_size, padding="same", activation=None)
        self.dropout = layers.Dropout(dropout_rate) if dropout_rate > 0 else None
        self.batch_norm1 = layers.BatchNormalization() if batch_norm else None
        self.batch_norm2 = layers.BatchNormalization() if batch_norm else None

    def call(self, inputs):
        x = self.conv1(inputs)
        if self.batch_norm1:
            x = self.batch_norm1(x)
        x = layers.ReLU()(x)
        x = self.conv2(x)
        if self.batch_norm2:
            x = self.batch_norm2(x)
        x = layers.ReLU()(x)
        if self.dropout:
            x = self.dropout(x)
        return x

class GatingSignal(tf.keras.layers.Layer):
    def __init__(self, filters, batch_norm=True):
        """
        Gating signal for attention mechanism.
        
        :param filters: Number of filters for Conv2D.
        :param batch_norm: Whether to apply Batch Normalization.
        """
        super(GatingSignal, self).__init__()
        self.conv = layers.Conv2D(filters, (1, 1), padding="same", activation=None)
        self.batch_norm = layers.BatchNormalization() if batch_norm else None

    def call(self, inputs):
        x = self.conv(inputs)
        if self.batch_norm:
            x = self.batch_norm(x)
        return layers.ReLU()(x)

class AttentionGate(tf.keras.layers.Layer):
    def __init__(self, filters):
        """
        Attention gate mechanism.
        
        :param filters: Number of filters for Conv2D.
        """
        super(AttentionGate, self).__init__()
        self.conv_f = layers.Conv2D(filters, (1, 1), padding="same")
        self.conv_g = layers.Conv2D(filters, (1, 1), padding="same")
        self.psi = layers.Conv2D(1, (1, 1), padding="same")
        self.batch_norm = layers.BatchNormalization()

    def call(self, x, g):
        f = self.conv_f(x)
        g = self.conv_g(g)
        f_g_comb = layers.add([f, g])
        f_g_comb = layers.ReLU()(f_g_comb)
        psi = self.psi(f_g_comb)
        psi = self.batch_norm(psi)
        psi = layers.Activation('sigmoid')(psi)
        return layers.multiply([x, psi])

class AttentionUWNet:
    def __init__(self, input_shape, num_classes=1, dropout_rate=0.0, batch_norm=True):
        """
        Attention UW-Net Model Initialization.
        
        :param input_shape: Shape of the input image.
        :param num_classes: Number of output classes.
        :param dropout_rate: Dropout rate.
        :param batch_norm: Whether to apply Batch Normalization.
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        self.filter_num = 32  # Basic number of filters
        self.filter_size = 3  # Size of convolution filters
        self.up_samp_size = 2  # Size of upsampling

    def build_model(self):
        """
        Build the Attention UW-Net model.
        
        :return: Keras Model object.
        """
        inputs = layers.Input(self.input_shape)

        # Encoder path
        conv_128, pool_64 = self._encoder_step(inputs, self.filter_num)
        conv_64, pool_32 = self._encoder_step(pool_64, 2 * self.filter_num)
        conv_32, pool_16 = self._encoder_step(pool_32, 4 * self.filter_num)
        conv_16, pool_8 = self._encoder_step(pool_16, 8 * self.filter_num)
        conv_8 = ConvBlock(16 * self.filter_num, self.filter_size, self.dropout_rate, self.batch_norm)(pool_8)

        # Decoder path with attention gates
        up_conv_16 = self._decoder_step(conv_8, conv_16, 8 * self.filter_num)
        up_conv_32 = self._decoder_step(up_conv_16, conv_32, 4 * self.filter_num)
        up_conv_64 = self._decoder_step(up_conv_32, conv_64, 2 * self.filter_num)
        up_conv_128 = self._decoder_step(up_conv_64, conv_128, self.filter_num)

        # Output layer
        conv_final = layers.Conv2D(self.num_classes, (1, 1))(up_conv_128)
        conv_final = layers.BatchNormalization(axis=3)(conv_final)
        conv_final = layers.Activation('sigmoid')(conv_final)

        # Build model
        model = models.Model(inputs=inputs, outputs=conv_final, name="Attention_UWNet")
        return model

    def _encoder_step(self, inputs, filters):
        """
        Apply a convolution block and max pooling for the encoder path.
        
        :param inputs: Input tensor.
        :param filters: Number of filters for the convolution block.
        :return: Tuple of (conv_output, pooled_output).
        """
        conv = ConvBlock(filters, self.filter_size, self.dropout_rate, self.batch_norm)(inputs)
        pool = layers.MaxPooling2D(pool_size=(2, 2))(conv)
        return conv, pool

    def _decoder_step(self, inputs, skip_connection, filters):
        """
        Apply upsampling, attention gate, concatenation, and a convolution block for the decoder path.
        
        :param inputs: Input tensor for upsampling.
        :param skip_connection: Skip connection from the encoder.
        :param filters: Number of filters for the convolution block.
        :return: Output tensor after the decoder step.
        """
        gating_signal = GatingSignal(filters, self.batch_norm)(inputs)
        attention = AttentionGate(filters)(skip_connection, gating_signal)
        upsampled = layers.UpSampling2D(size=(self.up_samp_size, self.up_samp_size))(inputs)
        concatenated = layers.concatenate([upsampled, attention], axis=3)
        return ConvBlock(filters, self.filter_size, self.dropout_rate, self.batch_norm)(concatenated)
