# UNet Segmentation Framework

This repository contains a comprehensive collection of advanced U-Net architectures designed for medical image segmentation tasks. These models integrate state-of-the-art features like Dense blocks, Spatial Attention, Channel Attention, and EfficientNet backbones to enhance segmentation performance for various medical datasets, including DICOM images.

## Models Included:
- **Dense U-Net with Spatial Attention**: Combines dense blocks and spatial attention mechanisms for robust feature extraction.
- **Efficient U-Net**: U-Net with EfficientNet as the backbone, improving efficiency and accuracy.
- **Attention UW-Net**: A U-Net variant that incorporates attention mechanisms in both the encoder and decoder.
- **UNet+++ (UNet++ Inspired)**: Dense connections in the decoder to aggregate features from multiple levels.
- **PVM + Channel Attention U-Net**: Uses Hybrid Dilated Convolutions with a Channel Attention Bridge to refine feature extraction.
- **ResDense Channel Attention U-Net**: Combines residual dense blocks with channel attention to focus on relevant features.
- **AAU-Net**: A U-Net variant with augmented hybrid attention mechanisms.

## Features:
- **Flexible Architectures**: Multiple U-Net variants that are customizable for different segmentation tasks.
- **Data Handling**: A custom `DataGenerator` class to manage loading, augmentation, and splitting of image and mask data.
- **Modular Codebase**: Easily extendable code to implement additional models and features.
- **DICOM Support**: Compatibility with DICOM images through the use of `pydicom`.
