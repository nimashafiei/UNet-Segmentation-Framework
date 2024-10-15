import tensorflow as tf
from tensorflow.keras import backend as K

class SegmentationMetrics:
    @staticmethod
    def dice_loss(y_true, y_pred):
        """
        Compute Dice Loss, a measure of overlap between the true and predicted masks.
        """
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return 1 - (2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)

    @staticmethod
    def dice_coefficient(y_true, y_pred):
        """
        Compute Dice Coefficient, which measures how well the predicted masks match the true masks.
        """
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)

    @staticmethod
    def iou_metric(y_true, y_pred):
        """
        Compute the Intersection over Union (IoU) metric, also known as the Jaccard index.
        """
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
        return (intersection + 1) / (union + 1)

    @staticmethod
    def pixel_accuracy(y_true, y_pred):
        """
        Compute pixel accuracy, measuring how many pixels were correctly classified.
        """
        y_pred_rounded = K.round(y_pred)
        correct_pixels = K.sum(K.cast(K.equal(y_true, y_pred_rounded), K.floatx()))
        total_pixels = K.cast(K.prod(K.shape(y_true)), K.floatx())
        return correct_pixels / total_pixels

    @staticmethod
    def precision(y_true, y_pred):
        """
        Compute the precision of the model, which is the ratio of true positives to predicted positives.
        """
        y_pred = K.round(y_pred)
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        return true_positives / (predicted_positives + K.epsilon())

    @staticmethod
    def recall(y_true, y_pred):
        """
        Compute the recall of the model, which is the ratio of true positives to actual positives.
        """
        y_pred = K.round(y_pred)
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        return true_positives / (possible_positives + K.epsilon())

    @staticmethod
    def f1_score(y_true, y_pred):
        """
        Compute the F1 score, which is the harmonic mean of precision and recall.
        """
        p = SegmentationMetrics.precision(y_true, y_pred)
        r = SegmentationMetrics.recall(y_true, y_pred)
        return 2 * (p * r) / (p + r + K.epsilon())
