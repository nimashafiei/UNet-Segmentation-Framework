import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, MaxPooling2D, UpSampling2D, ReLU, Dense, GlobalAveragePooling2D,
    Activation, Reshape, Lambda, concatenate, add, multiply
)
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

class ChannelAttentionBlock(tf.keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        """
        Channel Attention Block.
        
        :param filters: Number of filters to apply in the attention mechanism.
        """
        super(ChannelAttentionBlock, self).__init__(**kwargs)
        self.filters = filters

    def build(self, input_shape):
        self.conv1 = Conv2D(self.filters, (3, 3), padding="same", dilation_rate=(3, 3))
        self.conv2 = Conv2D(self.filters, (5, 5), padding="same")
        self.batch_norm = BatchNormalization()
        self.global_avg_pool = GlobalAveragePooling2D()
        self.fc1 = Dense(units=self.filters)
        self.fc2 = Dense(units=self.filters)
    
    def call(self, inputs):
        # First convolution branch
        conv1 = self.conv1(inputs)
        conv1 = self.batch_norm(conv1)
        relu1 = ReLU()(conv1)

        # Second convolution branch
        conv2 = self.conv2(inputs)
        conv2 = self.batch_norm(conv2)
        relu2 = ReLU()(conv2)

        # Concatenate the outputs of both convolution branches
        concat = concatenate([relu1, relu2])

        # Global average pooling
        pooled = self.global_avg_pool(concat)
        dense = self.fc1(pooled)
        dense = self.batch_norm(dense)
        dense = ReLU()(dense)
        dense = self.fc2(dense)
        sigmoid_out = Activation('sigmoid')(dense)

        # Reshape the sigmoid output to match input dimensions
        scale = Reshape((1, 1, self.filters))(sigmoid_out)
        scale = Lambda(lambda x: K.repeat_elements(x, K.int_shape(relu1)[1], axis=1))(scale)
        scale = Lambda(lambda x: K.repeat_elements(x, K.int_shape(relu1)[2], axis=2))(scale)

        # Apply attention
        attention1 = multiply([relu1, scale])
        attention2 = multiply([relu2, 1 - scale])

        # Concatenate the results of attention
        result = concatenate([attention1, attention2])

        return Conv2D(self.filters, (1, 1), padding="same")(result)


class SpatialAttentionBlock(tf.keras.layers.Layer):
    def __init__(self, filters, size, **kwargs):
        """
        Spatial Attention Block.
        
        :param filters: Number of filters.
        :param size: Kernel size for the convolution.
        """
        super(SpatialAttentionBlock, self).__init__(**kwargs)
        self.filters = filters
        self.size = size

    def build(self, input_shape):
        self.conv1 = Conv2D(self.filters, (3, 3), padding="same")
        self.conv2 = Conv2D(self.filters, (1, 1), padding="same")
        self.conv3 = Conv2D(1, (1, 1), padding='same')
        self.final_conv = Conv2D(self.filters, self.size, padding='same')
        self.batch_norm = BatchNormalization()

    def call(self, inputs, channel_data):
        # Convolutions for spatial attention
        conv1 = self.conv1(inputs)
        conv1 = self.batch_norm(conv1)
        relu1 = ReLU()(conv1)

        conv2 = self.conv2(relu1)
        conv2 = self.batch_norm(conv2)
        relu2 = ReLU()(conv2)

        # Add channel and spatial data
        spatial_attention = add([channel_data, relu2])
        spatial_attention = ReLU()(spatial_attention)

        # Apply final convolutions
        spatial_attention = self.conv3(spatial_attention)
        spatial_attention = Activation('sigmoid')(spatial_attention)

        # Expand spatial attention to match input dimensions
        scale = Lambda(lambda x: K.repeat_elements(x, rep=self.filters, axis=-1))(spatial_attention)

        # Apply spatial attention
        scaled_channel_data = multiply([scale, channel_data])
        scaled_spatial_data = multiply([1 - scale, relu2])

        concat = concatenate([scaled_channel_data, scaled_spatial_data])
        result = self.final_conv(concat)
        return self.batch_norm(result)


class HybridAttentionAugmentationModule(tf.keras.layers.Layer):
    def __init__(self, filters, size, **kwargs):
        """
        Hybrid Attention Augmentation Module (HAAM).
        
        :param filters: Number of filters.
        :param size: Kernel size for the final convolution.
        """
        super(HybridAttentionAugmentationModule, self).__init__(**kwargs)
        self.filters = filters
        self.size = size

    def build(self, input_shape):
        self.channel_attention = ChannelAttentionBlock(filters=self.filters)
        self.spatial_attention = SpatialAttentionBlock(filters=self.filters, size=self.size)

    def call(self, inputs):
        channel_data = self.channel_attention(inputs)
        output = self.spatial_attention(inputs, channel_data)
        return output


class AAUNet:
    def __init__(self, input_shape=(512, 512, 3), num_classes=1):
        """
        AAU-Net Model Initialization.
        
        :param input_shape: Shape of the input image.
        :param num_classes: Number of output classes.
        """
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build_model(self):
        """
        Build the AAU-Net model with HAAM.
        
        :return: Compiled Keras Model.
        """
        inputs = Input(shape=self.input_shape)

        # Initial Conv Layer
        x = Conv2D(32, (3, 3), padding="same", activation="relu")(inputs)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Apply HAAM block
        x = HybridAttentionAugmentationModule(filters=64, size=(3, 3))(x)

        # Further Conv Layers
        x = Conv2D(64, (3, 3), padding="same", activation="relu")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Upsample to restore spatial dimensions
        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(32, (3, 3), padding="same", activation="relu")(x)

        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(16, (3, 3), padding="same", activation="relu")(x)

        # Final Output Layer
        outputs = Conv2D(self.num_classes, (1, 1), padding="same", activation="sigmoid")(x)

        # Create the model
        model = Model(inputs, outputs)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        return model
