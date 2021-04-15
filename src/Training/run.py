import os
import time
import numpy as np
import tensorflow as tf

from tensorflow.keras import models, optimizers, backend as K, callbacks
from tensorflow.keras.utils import plot_model

import src.settings as settings
from src.Utils.plot import plot_loss_accuracy
from src.Utils.data_loader import generator_loader, multiple_outputs
from src.Utils.loadHVCModel import load_hvc_model
from src.Training.activation_and_loss import custom_loss
from src.Utils.utils import get_layer_names


# ----------------------------------------------------------------------------------------------------------------------
# Loading data

# Create Training, validation, and Testing generators
train_generator, val_generator, testing_generator = generator_loader(settings.training_path,
                                                                     settings.val_path,
                                                                     settings.testing_path,
                                                                     settings.image_width,
                                                                     settings.image_height,
                                                                     settings.batch_size)

total_history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}


# ----------------------------------------------------------------------------------------------------------------------
# Training

start_time_training = time.time()

# Iterate for each sub_model backward from the last sub_model to the first sub_model
for sub_model_num in range(settings.total_sub_models_num, 0, -1):

    print(f'\nTraining the sub_model {sub_model_num} is started!\n')

    sub_model_name = os.path.join(settings.sub_model_directory, 'sub_model_' + str(sub_model_num) + '.h5')

    # Create the sub_model
    sub_model, vcl, inputs, vcl_next = settings.neural_network(num_classes=settings.num_classes,
                                                               dropout_rate=settings.drop_off,
                                                               num_vc=settings.num_vc,
                                                               layer_to_vcl=sub_model_num,
                                                               p_h=settings.p_h,
                                                               p_w=settings.p_w,
                                                               total_sub_models_num=settings.total_sub_models_num)
    print(sub_model.summary())
    if settings.plot_model:
        plot_model(sub_model, to_file=os.path.join(settings.sub_model_directory, 'sub_model_' + str(sub_model_num) + '.png'))

    # Load the base model
    base_model = models.load_model(settings.base_model_name)

    # Get the layers name of the sub_model and base_model
    sub_model_layers = get_layer_names(sub_model)
    base_model_layers = get_layer_names(base_model)

    # Copy the weights of the base model to the convolutional layers of the sub_model and set the trainable to False
    for i, layer in enumerate(sub_model.layers):
        if layer.name in base_model_layers:
            layer.set_weights(base_model.get_layer(layer.name).get_weights())
            layer.trainable = False

    # Copy the weights of the two additional convolutional and the VCL from the next sub_mode to the current sub_model
    if sub_model_num != settings.total_sub_models_num:
        next_sub_model_name = os.path.join(settings.sub_model_directory, 'sub_model_' + str(sub_model_num + 1) + '.h5')

        next_sub_mode = load_hvc_model(model_path=next_sub_model_name,
                                       neural_network=settings.neural_network,
                                       num_classes=settings.num_classes,
                                       drop_off=settings.drop_off,
                                       total_sub_models_num=settings.total_sub_models_num,
                                       num_vc=settings.num_vc,
                                       p_h=settings.p_h,
                                       p_w=settings.p_w)

        sub_model.get_layer('additional_layer_1_next').set_weights(next_sub_mode.get_layer('additional_layer_1').get_weights())
        sub_model.get_layer('additional_layer_2_next').set_weights(next_sub_mode.get_layer('additional_layer_2').get_weights())
        sub_model.get_layer('vcl_1').set_weights(next_sub_mode.get_layer('vcl').get_weights())
        sub_model.get_layer('additional_layer_1_next').trainable = False
        sub_model.get_layer('additional_layer_2_next').trainable = False
        sub_model.get_layer('vcl_1').trainable = False

    if sub_model_num != settings.total_sub_models_num:
        gap = (settings.image_height // sub_model.get_layer('vcl').distances.shape[1]) * settings.p_h
        n_gap = (settings.image_height // sub_model.get_layer('vcl_1').distances.shape[1]) * settings.p_h

    def compile_model():
        sgd = optimizers.SGD(learning_rate=0.001, decay=1e-6, momentum=0.9, nesterov=True)
        if sub_model_num == settings.total_sub_models_num:
            sub_model.compile(optimizer=sgd,
                              loss=custom_loss(vc_distances=vcl[3],
                                               vc_weights=vcl[1][0],
                                               batch_size=tf.constant(settings.batch_size),
                                               num_classes=tf.constant(settings.num_classes),
                                               inputs_images=inputs,
                                               p_h=settings.p_h,
                                               p_w=settings.p_w),
                              metrics=['accuracy'])

        else:
            sub_model.compile(optimizer=sgd,
                              loss={'sub_model_predictions': custom_loss(vc_distances=vcl[3],
                                                                         vc_weights=vcl[1][0],
                                                                         batch_size=tf.constant(settings.batch_size),
                                                                         num_classes=tf.constant(settings.num_classes),
                                                                         inputs_images=inputs,
                                                                         next_coordinates=vcl_next[4],
                                                                         current_coordinates=vcl[4],
                                                                         vc_per_class=settings.vc_per_class,
                                                                         vcl_activations=vcl[0],
                                                                         gap=gap,
                                                                         n_gap=n_gap,
                                                                         p_h=settings.p_h,
                                                                         p_w=settings.p_w),
                                    'sub_model_predictions2': 'categorical_crossentropy'},
                              loss_weights={'sub_model_predictions': 1.0,
                                            'sub_model_predictions2': 0.0},
                              metrics={'sub_model_predictions': ['accuracy']})

    # Checkpoint
    checkpoint_sub_model = callbacks.ModelCheckpoint(sub_model_name, monitor='val_loss', verbose=1, save_best_only=True, mode='min')


    def fit_model(num_epoch_):
        if sub_model_num == settings.total_sub_models_num:
            training_history = sub_model.fit(train_generator,
                                             steps_per_epoch=train_generator.samples // train_generator.batch_size,
                                             validation_data=val_generator,
                                             validation_steps=val_generator.samples // val_generator.batch_size,
                                             epochs=num_epoch_,
                                             callbacks=[checkpoint_sub_model],
                                             verbose=1)
            for key_, value_ in total_history.items():
                total_history[key_] += training_history.history[key_]
        else:
            training_history = sub_model.fit(multiple_outputs(train_generator),
                                             steps_per_epoch=train_generator.samples // train_generator.batch_size,
                                             validation_data=multiple_outputs(val_generator),
                                             validation_steps=val_generator.samples // val_generator.batch_size,
                                             epochs=num_epoch_,
                                             callbacks=[checkpoint_sub_model],
                                             verbose=1)

            total_history['loss'] += training_history.history['loss']
            total_history['val_loss'] += training_history.history['val_loss']
            total_history['accuracy'] += training_history.history['sub_model_predictions_accuracy']
            total_history['val_accuracy'] += training_history.history['val_sub_model_predictions_accuracy']


    # Train the sub_model
    iteration = 0
    trainable_w = False
    while iteration < settings.total_iteration:
        if not trainable_w and iteration > int(settings.epochs/2):
            sub_model.get_layer('sub_model_predictions').trainable = True
            trainable_w = True

        compile_model()
        fit_model(settings.training_epochs)
        iteration += settings.training_epochs

    sub_model.save(settings.sub_model_directory, 'Last_sub_model_' + str(sub_model_num) + '.h5')

    print("Training the sub_model is over and the best sub_model saved to disk!")

    # Clear the session
    K.clear_session()

print('Training is over. The Training time = %s seconds' % (time.time() - start_time_training))

# Plot the results
plot_loss_accuracy(total_history, settings.output_directory, model_acc_file='HVC_model_accuracy.png', model_loss_file='HVC_model_loss.png')
