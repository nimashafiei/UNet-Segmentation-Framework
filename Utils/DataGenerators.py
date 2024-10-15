from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

class DataGeneratorManager:
    def __init__(self, root_path, img_size=(512, 512), batch_size=4):
        """
        Initialize DataGeneratorManager to handle all data-related processes.
        
        :param root_path: The root path where 'Train', 'Validation', and 'Test' folders are located.
        :param img_size: Target image size (default: 512x512).
        :param batch_size: Batch size for generators (default: 4).
        """
        self.root_path = root_path
        self.img_size = img_size
        self.batch_size = batch_size
        self.datagen = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=30,
            fill_mode='nearest',
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=0.3
        )
    
    def create_image_mask_generators(self, path, image_subfolder, mask_subfolder):
        """
        Create the generators for images and masks.
        
        :param path: The directory containing images and masks.
        :param image_subfolder: The subfolder containing the images (e.g., 'img').
        :param mask_subfolder: The subfolder containing the masks (e.g., 'Mask').
        :return: Two generators, one for images and one for masks.
        """
        image_gen = self.datagen.flow_from_directory(
            os.path.join(path, image_subfolder),
            class_mode=None,
            color_mode='rgb',
            target_size=self.img_size,
            batch_size=self.batch_size,
            seed=1
        )

        mask_gen = self.datagen.flow_from_directory(
            os.path.join(path, mask_subfolder),
            class_mode=None,
            color_mode='grayscale',
            target_size=self.img_size,
            batch_size=self.batch_size,
            seed=1
        )

        return image_gen, mask_gen

    def zip_generators(self, image_gen, mask_gen):
        """
        Zip together image and mask generators.
        
        :param image_gen: Generator for images.
        :param mask_gen: Generator for masks.
        :return: A generator yielding images and corresponding masks.
        """
        while True:
            yield next(image_gen), next(mask_gen)
    
    def get_generators(self):
        """
        Returns the train, validation, and test generators and calculates the number of validation steps.
        
        :return: train_generator, val_generator, test_generator, validation_steps
        """
        train_path = os.path.join(self.root_path, 'Train')
        val_path = os.path.join(self.root_path, 'Validation')
        test_path = os.path.join(self.root_path, 'Test')

        train_image_gen, train_mask_gen = self.create_image_mask_generators(train_path, 'img', 'Mask')
        val_image_gen, val_mask_gen = self.create_image_mask_generators(val_path, 'img', 'Mask')
        test_image_gen, test_mask_gen = self.create_image_mask_generators(test_path, 'img', 'Mask')

        train_generator = self.zip_generators(train_image_gen, train_mask_gen)
        val_generator = self.zip_generators(val_image_gen, val_mask_gen)
        test_generator = self.zip_generators(test_image_gen, test_mask_gen)

        num_val_images = len(val_image_gen.filenames)
        validation_steps = num_val_images // self.batch_size + (num_val_images % self.batch_size > 0)

        return train_generator, val_generator, test_generator, validation_steps
