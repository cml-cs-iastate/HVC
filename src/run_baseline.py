import os
import time

from tensorflow.keras import models, optimizers, callbacks, utils

import settings as settings
from Utils.plot import plot_loss_accuracy
from Utils.data_loader import generator_loader

total_history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}


# ----------------------------------------------------------------------------------------------------------------------
# Loading data

# Create Training, validation, and Testing generators
train_generator, val_generator, testing_generator = generator_loader(settings.training_path,
                                                                     settings.val_path,
                                                                     settings.testing_path,
                                                                     settings.image_width,
                                                                     settings.image_height,
                                                                     settings.batch_size)


# ----------------------------------------------------------------------------------------------------------------------
# Model

model = settings.neural_network(num_classes=settings.num_classes, dropout_rate=settings.drop_off)
print(model.summary())

# Plot the sub_model
if settings.plot_model:
    utils.plot_model(model, to_file=os.path.join(settings.output_directory, 'base_model.png'))

# Compile the model
sgd = optimizers.SGD(learning_rate=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

# Checkpoint
checkpoint = callbacks.ModelCheckpoint(settings.base_model_name, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')


# ----------------------------------------------------------------------------------------------------------------------
# Training

start_time_training = time.time()

# Train the model
history = model.fit(train_generator,
                    steps_per_epoch=train_generator.samples // train_generator.batch_size,
                    validation_data=val_generator,
                    validation_steps=val_generator.samples // val_generator.batch_size,
                    epochs=settings.epochs,
                    callbacks=[checkpoint],
                    verbose=1)

print(f'Training is over, the best model is saved to disk. \nThe training time is {time.time() - start_time_training} seconds')

# Plot the Training results
plot_loss_accuracy(history.history, settings.output_directory)


# ----------------------------------------------------------------------------------------------------------------------
# Testing

new_model = models.load_model(settings.base_model_name)

# Evaluate the model with Testing data
if testing_generator:
    results = new_model.evaluate(testing_generator)
else:
    results = new_model.evaluate(val_generator)
print(f'Testing is over, the classification accuracy: {results[1]} , and loss: {results[0]}')
