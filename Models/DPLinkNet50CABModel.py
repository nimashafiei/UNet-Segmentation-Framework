from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
import tensorflow as tf

class DPLinkNetCABModel:
    def __init__(self, input_shape=(512, 512, 3), num_classes=1):
        """
        Initialize the DPLinkNet50 model with Channel Attention Bridge (CAB).
        
        :param input_shape: Shape of the input images.
        :param num_classes: Number of output classes (1 for binary segmentation).
        """
        self.input_shape = input_shape
        self.num_classes = num_classes

    def resize_tensor(self, tensor, target_size):
        """
        Resize the given tensor to the target size.
        
        :param tensor: Input tensor to resize.
        :param target_size: Desired output size.
        :return: Resized tensor.
        """
        return tf.image.resize(tensor, target_size, method='bilinear')

    def convolution_block(self, input_tensor, filters, kernel_size=(1, 1), padding="same", activation='relu'):
        """
        Define a basic convolutional block with a single convolutional layer and activation.
        
        :param input_tensor: Input tensor for the convolution.
        :param filters: Number of output filters for the convolution.
        :param kernel_size: Size of the convolution kernel.
        :param padding: Padding type.
        :param activation: Activation function.
        :return: Output tensor after the convolution block.
        """
        conv_layer = layers.Conv2D(filters, kernel_size=kernel_size, padding=padding)(input_tensor)
        return layers.Activation(activation)(conv_layer)

    def channel_attention(self, input_tensor, reduction_ratio=8):
        """
        Apply Channel Attention Bridge (CAB) to the input tensor.
        
        :param input_tensor: Input tensor to apply attention to.
        :param reduction_ratio: The reduction ratio for channel compression.
        :return: Output tensor after applying channel attention.
        """
        channels = input_tensor.shape[-1]
        avg_pool = layers.GlobalAveragePooling2D()(input_tensor)
        avg_pool = layers.Reshape((1, 1, channels))(avg_pool)

        dense1 = layers.Dense(channels // reduction_ratio, activation='relu')(avg_pool)
        dense2 = layers.Dense(channels, activation='sigmoid')(dense1)

        attention = layers.multiply([input_tensor, dense2])
        return attention

    def spatial_pyramid_pool(self, input_tensor):
        """
        Apply Spatial Pyramid Pooling (SPP) to the input tensor.
        
        :param input_tensor: Input tensor to apply SPP.
        :return: Output tensor after SPP.
        """
        h, w = input_tensor.shape[1:3]
        pool_sizes = [2, 3, 5]
        
        pooled_outputs = [
            layers.MaxPooling2D(pool_size=(size, size), strides=size, padding="same")(input_tensor)
            for size in pool_sizes
        ]

        convoluted_pools = [self.convolution_block(pool, 512) for pool in pooled_outputs]
        upscaled_pools = [layers.Lambda(lambda x: self.resize_tensor(x, (h, w)))(conv) for conv in convoluted_pools]

        concat_pools = layers.Concatenate()([*upscaled_pools, input_tensor])
        return self.convolution_block(concat_pools, 512)

    def hybrid_dilated_convolutions(self, input_tensor, filters):
        """
        Apply Hybrid Dilated Convolutions (HDC) to the input tensor.
        
        :param input_tensor: Input tensor for the block.
        :param filters: Number of filters for each dilated convolution.
        :return: Output tensor after applying HDC with residual connections.
        """
        dilation_rates = [1, 2, 4]
        
        dilated_convs = [
            layers.Conv2D(filters, kernel_size=(3, 3), dilation_rate=rate, padding="same", activation='relu')(input_tensor)
            for rate in dilation_rates
        ]

        return layers.Add()([input_tensor] + dilated_convs)

    def decoder_block(self, input_tensor, filters):
        """
        Define the decoder block using transposed convolutions for upsampling.
        
        :param input_tensor: Input tensor for the block.
        :param filters: Number of filters for the convolution layers.
        :return: Output tensor after the decoder block.
        """
        conv_1x1 = self.convolution_block(input_tensor, filters // 4)

        trans_conv = layers.Conv2DTranspose(filters // 4, kernel_size=(3, 3), strides=(2, 2), padding="same")(conv_1x1)
        trans_conv_norm = layers.BatchNormalization()(trans_conv)
        trans_conv_relu = layers.Activation('relu')(trans_conv_norm)

        return layers.BatchNormalization()(self.convolution_block(trans_conv_relu, filters))

    def build_model(self):
        """
        Build the complete DPLinkNet50 model with Channel Attention Bridge.
        
        :return: Compiled Keras Model.
        """
        # Encoder from ResNet50 backbone
        resnet_encoder = ResNet50(include_top=False, weights="imagenet", input_shape=self.input_shape)
        encoder_1 = resnet_encoder.get_layer("conv1_relu").output
        encoder_2 = resnet_encoder.get_layer("conv2_block3_out").output
        encoder_3 = resnet_encoder.get_layer("conv3_block4_out").output

        # Center block with Hybrid Dilated Convolutions and Spatial Pyramid Pooling
        center_block = self.hybrid_dilated_convolutions(encoder_3, 512)
        spp_block = self.spatial_pyramid_pool(center_block)

        # Decoder blocks with skip connections
        decoder_3 = self.decoder_block(spp_block, 256)
        decoder_3 = layers.Add()([decoder_3, encoder_2])

        decoder_2 = self.decoder_block(decoder_3, 64)
        decoder_2 = layers.Add()([decoder_2, encoder_1])

        decoder_1 = self.decoder_block(decoder_2, 64)

        # Apply Channel Attention Bridge (CAB) to the final decoder output
        final_decoder = self.channel_attention(decoder_1)

        # Output layers
        final_conv = self.convolution_block(final_decoder, 32, kernel_size=(3, 3))
        output_layer = layers.Conv2D(self.num_classes, kernel_size=(1, 1), padding="same")(final_conv)

        # Sigmoid activation for binary classification or softmax for multi-class
        final_activation = layers.Activation('sigmoid' if self.num_classes == 1 else 'softmax')(output_layer)

        return models.Model(inputs=resnet_encoder.input, outputs=final_activation)

