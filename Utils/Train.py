from tensorflow.keras.callbacks import ModelCheckpoint
from data_generators import create_generators
from models import DPLinkNet50
from metrics import dice_loss, pixel_accuracy, dice_coefficient, iou_metric, precision, recall, f1_score
from tensorflow.keras.optimizers import Adam

# Create generators
root_path = 'LIDC-Data/'
train_generator, val_generator, test_generator, num_val_images = create_generators(root_path)

# Model Configuration
input_shape = (512, 512, 3)
model = DPLinkNet50(input_shape=input_shape, num_classes=1)

# Compile Model
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss=dice_loss,
    metrics=[pixel_accuracy, dice_coefficient, iou_metric, precision, recall, f1_score]
)

# Model Summary
model.summary()

# Define Checkpoint
checkpoint = ModelCheckpoint(
    'best_DPLink50Ch_model.keras',
    monitor='val_dice_coefficient',
    save_best_only=True,
    mode='max',
    verbose=1
)

# Calculate validation steps
validation_steps = num_val_images // 4 + (num_val_images % 4 > 0)

# Train the Model
model.fit(
    train_generator,
    validation_data=val_generator,
    validation_steps=validation_steps,
    epochs=150,
    steps_per_epoch=250,
    verbose=1,
    callbacks=[checkpoint]
)
