from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
import tensorflow as tf

class DPLinkNetModel:
    def __init__(self, input_shape=(512, 512, 3), num_classes=1):
        """
        Initialize the DPLinkNet50 model with Channel Attention Bridge.
        
        :param input_shape: Shape of the input images.
        :param num_classes: Number of output classes (1 for binary segmentation).
        """
        self.input_shape = input_shape
        self.num_classes = num_classes

    def resize_image_tensor(self, tensor, target_size):
        """
        Resize image tensors to a target size.
        
        :param tensor: Input tensor to resize.
        :param target_size: Target size to resize the tensor to.
        :return: Resized tensor.
        """
        return tf.image.resize(tensor, target_size, method='bilinear')

    def conv_block(self, input_tensor, num_filters, kernel_size=(1, 1), padding="same", activation='relu'):
        """
        Define a convolution block.
        
        :param input_tensor: Input tensor for the convolution block.
        :param num_filters: Number of filters in the convolution layer.
        :param kernel_size: Size of the convolution kernel.
        :param padding: Padding type for the convolution layer.
        :param activation: Activation function.
        :return: Output tensor after the convolution block.
        """
        conv = layers.Conv2D(num_filters, kernel_size=kernel_size, padding=padding)(input_tensor)
        return layers.Activation(activation)(conv)

    def spatial_pyramid_pooling(self, input_tensor):
        """
        Define the Spatial Pyramid Pooling block.
        
        :param input_tensor: Input tensor for the block.
        :return: Output tensor after pooling and concatenation.
        """
        h, w = input_tensor.shape[1:3]
        pool_sizes = [2, 3, 5]
        pooled_layers = [
            layers.MaxPooling2D(pool_size=(size, size), strides=size, padding="same")(input_tensor)
            for size in pool_sizes
        ]

        conv_pooled_layers = [self.conv_block(pool, 512) for pool in pooled_layers]
        upscaled_layers = [layers.Lambda(lambda x: self.resize_image_tensor(x, (h, w)))(conv) for conv in conv_pooled_layers]

        concat = layers.Concatenate()([*upscaled_layers, input_tensor])
        return self.conv_block(concat, 512)

    def hybrid_dilated_convolution(self, input_tensor, num_channels):
        """
        Define the Hybrid Dilated Convolution block.
        
        :param input_tensor: Input tensor for the block.
        :param num_channels: Number of channels in the convolution layers.
        :return: Output tensor after dilated convolutions and residual addition.
        """
        dilations = [1, 2, 4]
        conv_layers = [
            layers.Conv2D(num_channels, kernel_size=(3, 3), dilation_rate=d, padding="same", activation='relu')(input_tensor)
            for d in dilations
        ]
        return layers.Add()([input_tensor] + conv_layers)

    def decoder_block(self, input_tensor, num_filters):
        """
        Define the decoder block with transposed convolution.
        
        :param input_tensor: Input tensor for the block.
        :param num_filters: Number of filters in the transposed convolution layers.
        :return: Output tensor after upsampling and convolution.
        """
        conv_1x1 = self.conv_block(input_tensor, num_filters // 4)
        deconv = layers.Conv2DTranspose(num_filters // 4, kernel_size=(3, 3), strides=(2, 2), padding="same")(conv_1x1)
        deconv_norm = layers.BatchNormalization()(deconv)
        deconv_relu = layers.Activation('relu')(deconv_norm)
        return layers.BatchNormalization()(self.conv_block(deconv_relu, num_filters))

    def build_model(self):
        """
        Build the complete DPLinkNet50 model architecture.
        
        :return: Keras Model object.
        """
        resnet_encoder = ResNet50(include_top=False, weights="imagenet", input_shape=self.input_shape)
        encoder_1 = resnet_encoder.get_layer("conv1_relu").output
        encoder_2 = resnet_encoder.get_layer("conv2_block3_out").output
        encoder_3 = resnet_encoder.get_layer("conv3_block4_out").output

        center_block = self.hybrid_dilated_convolution(encoder_3, 512)
        spp_block = self.spatial_pyramid_pooling(center_block)

        decoder_3 = self.decoder_block(spp_block, 256)
        decoder_3 = layers.Add()([decoder_3, encoder_2])

        decoder_2 = self.decoder_block(decoder_3, 64)
        decoder_2 = layers.Add()([decoder_2, encoder_1])

        decoder_1 = self.decoder_block(decoder_2, 64)
        final_conv = self.conv_block(decoder_1, 32, kernel_size=(3, 3))
        output_layer = layers.Conv2D(self.num_classes, kernel_size=(1, 1), padding="same")(final_conv)

        final_activation = layers.Activation('sigmoid' if self.num_classes == 1 else 'softmax')(output_layer)
        return models.Model(inputs=resnet_encoder.input, outputs=final_activation)
