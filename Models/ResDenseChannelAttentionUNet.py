import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, MaxPooling2D, UpSampling2D,
    Concatenate, Multiply
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

class ResidualDenseBlock(tf.keras.layers.Layer):
    def __init__(self, filters, num_layers=3, **kwargs):
        """
        Residual Dense Block (ResDense).
        
        :param filters: Number of filters in each Conv2D layer.
        :param num_layers: Number of dense layers within the block.
        """
        super(ResidualDenseBlock, self).__init__(**kwargs)
        self.filters = filters
        self.num_layers = num_layers
        self.conv_layers = [Conv2D(filters, (3, 3), activation='relu', padding='same') for _ in range(num_layers)]
        self.batch_norm_layers = [BatchNormalization() for _ in range(num_layers)]

    def call(self, inputs):
        """
        Forward pass for Residual Dense Block.
        
        :param inputs: Input tensor.
        :return: Output tensor after applying the residual dense block.
        """
        x = inputs
        skip_connection = inputs
        for i in range(self.num_layers):
            x = self.conv_layers[i](x)
            x = self.batch_norm_layers[i](x)
        
        # Concatenate skip connection (residual connection)
        x = Concatenate()([x, skip_connection])
        return x

    def call(self, inputs):
        """
        Forward pass for Channel Attention.
        
        :param inputs: Input tensor.
        :return: Output tensor with attention applied.
        """
        avg_pooled = self.global_avg_pool(inputs)
        avg_pooled = tf.expand_dims(tf.expand_dims(avg_pooled, 1), 1)  # Add spatial dimensions
        fc1_output = self.fc1(avg_pooled)
        att_weights = self.fc2(fc1_output)
        return Multiply()([inputs, att_weights])

class ResDenseChannelAttentionUNet:
    def __init__(self, img_height, img_width, base_filters):
        """
        U-Net with Residual Dense Block and Channel Attention.
        
        :param img_height: Height of the input image.
        :param img_width: Width of the input image.
        :param base_filters: Number of filters for the first Conv2D layer.
        """
        self.img_height = img_height
        self.img_width = img_width
        self.base_filters = base_filters

    def build_model(self):
        """
        Build the U-Net model with Residual Dense Block and Channel Attention.
        
        :return: Compiled Keras Model.
        """
        inputs = Input((self.img_height, self.img_width, 3))
        width = self.base_filters

        # Encoder path with ResDense and Channel Attention
        conv1 = self._encoder_block(inputs, width)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = self._encoder_block(pool1, width * 2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = self._encoder_block(pool2, width * 4)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = self._encoder_block(pool3, width * 8)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = self._encoder_block(pool4, width * 16)

        # Decoder path
        up6 = self._decoder_block(conv5, conv4, width * 8)
        up7 = self._decoder_block(up6, conv3, width * 4)
        up8 = self._decoder_block(up7, conv2, width * 2)
        up9 = self._decoder_block(up8, conv1, width)

        # Output layer
        outputs = Conv2D(1, (1, 1), activation='sigmoid')(up9)

        # Create the model
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=1e-5), loss=self.dice_loss, metrics=[self.dice_coefficient])

        return model

    def _encoder_block(self, inputs, filters):
        """
        Create an encoder block with Residual Dense Block and Channel Attention.
        
        :param inputs: Input tensor.
        :param filters: Number of filters for the block.
        :return: Output tensor after the encoder block.
        """
        # Apply Residual Dense Block
        x = ResidualDenseBlock(filters)(inputs)

        # Apply Channel Attention
        x = ChannelAttention(filters)(x)

        return x

    def _decoder_block(self, upsampled_input, skip_connection, filters):
        """
        Create a decoder block with upsampling and concatenation.
        
        :param upsampled_input: Input tensor to upsample.
        :param skip_connection: Skip connection from the corresponding encoder block.
        :param filters: Number of filters for the block.
        :return: Output tensor after the decoder block.
        """
        upsampled = UpSampling2D(size=(2, 2))(upsampled_input)
        concatenated = Concatenate()([upsampled, skip_connection])

        # Apply Residual Dense Block
        x = ResidualDenseBlock(filters)(concatenated)

        return x

