"""
File name: model_training.py
Author: Jinesh Mehta
File Description: This file contains the functions to train the model.
"""
import os
from constants import RESNET_MODEL_PATH, RESNET_FER_IMG_HEIGHT, RESNET_FER_IMG_WIDTH,\
    BASE_MODEL_NAME, BASE_MODEL_INITIAL_WEIGHTS, NUM_CLASSES, \
    EPOCHS_TOP_LAYERS, EPOCHS_ALL_LAYERS, BATCH_SIZE, RESNET_FER_MEAN,\
    FER_TRAIN_DATA_PATH, FER_EVAL_DATA_PATH, LOGS_DIRECTORY_PRE_LAYERS,\
    LOGS_DIRECTORY_ALL_LAYERS, MODEL_LEARNING_RATE, MODEL_MOMENTUM, \
    MODEL_DECAY, MODEL_LOSS, CHECKPOINT_DIRECTORY

import numpy as np
import pandas as pd
from keras.optimizers import SGD
from skimage.transform import resize
from keras import backend as K
from keras.utils import to_categorical
from keras_vggface.vggface import VGGFace
from keras.models import Model
from keras.layers import Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping, Callback


def model_creation_from_vgg_face():
    """ Creates a model from the VGGFace model.

    Returns:
        _type_: Model
    """
    base_model = VGGFace(
        model=BASE_MODEL_NAME,
        include_top=False,
        weights=BASE_MODEL_INITIAL_WEIGHTS,
        input_shape=(RESNET_FER_IMG_HEIGHT, RESNET_FER_IMG_WIDTH, 3))
    x = base_model.output
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return base_model, model


def perform_data_preparation():
    """ Performs data preparation for the model.
    """

    def preprocess_input(x):
        x -= RESNET_FER_MEAN
        return x

    def get_data(dataset_location):
        data = pd.read_csv(dataset_location)
        pixels = data['pixels'].tolist()
        images = np.empty(
            (len(data), RESNET_FER_IMG_HEIGHT, RESNET_FER_IMG_WIDTH, 3))
        i = 0
        for pixel_sequence in pixels:
            single_image = [float(pixel)
                            for pixel in pixel_sequence.split(' ')]
            single_image = np.asarray(single_image).reshape(
                48, 48)
            single_image = resize(
                single_image, (RESNET_FER_IMG_HEIGHT, RESNET_FER_IMG_WIDTH), order=3, mode='constant')
            ret = np.empty((RESNET_FER_IMG_HEIGHT, RESNET_FER_IMG_WIDTH, 3))
            ret[:, :, 0] = single_image
            ret[:, :, 1] = single_image
            ret[:, :, 2] = single_image
            images[i, :, :, :] = ret
            i += 1
        images = preprocess_input(images)
        labels = to_categorical(data['emotion'])
        return images, labels

    train_data_x, train_data_y = get_data(FER_TRAIN_DATA_PATH)
    val_data = get_data(FER_EVAL_DATA_PATH)

    train_datagen = ImageDataGenerator(
        rotation_range=10,
        shear_range=10,
        zoom_range=0.1,
        fill_mode='reflect',
        horizontal_flip=True)

    train_generator = train_datagen.flow(
        train_data_x,
        train_data_y,
        batch_size=BATCH_SIZE)

    return train_generator, train_data_x, val_data


def training_pre_layers(base_model, model, train_generator, train_data_x, val_data):
    """ Trains the model with the pre layers.

    Args:
        base_model (_type_): base model
        model (_type_): model
        train_generator (_type_): train generator
        train_data_x (_type_): train data x
        val_data (_type_): val data

    Returns:
        _type_: Model
    """
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.999,
                       epsilon=1e-08, decay=0.0),
        loss=MODEL_LOSS,
        metrics=['accuracy'])

    tensorboard_top_layers = TensorBoard(
        log_dir=LOGS_DIRECTORY_PRE_LAYERS,
        histogram_freq=0,
        write_graph=True,
        write_grads=False,
        write_images=True)

    model.fit(
        generator=train_generator,
        steps_per_epoch=len(train_data_x) // BATCH_SIZE,
        epochs=EPOCHS_TOP_LAYERS,
        validation_data=val_data,
        callbacks=[tensorboard_top_layers])
    return model


class ModelCheckpoint(Callback):
    """ModelCheckpoint callback modified to save the best model based on the validation loss instead of the training loss.

    Args:
        Callback (_type_): callback
    """

    def __init__(self, filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1):
        """Initializes the ModelCheckpoint callback.

        Args:
            filepath (_type_): filepath
            monitor (str, optional): monitor. Defaults to 'val_loss'.
            verbose (int, optional): verbose. Defaults to 0.
            save_best_only (bool, optional): save the best. Defaults to False.
            save_weights_only (bool, optional): save weights. Defaults to False.
            mode (str, optional): mode. Defaults to 'auto'.
            period (int, optional): period. Defaults to 1.
        """
        super(ModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            mode = 'auto'
        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        """On epoch end.

        Args:
            epoch (_type_): epoch
            logs (_type_, optional): logs location. Defaults to None.
        """
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    pass
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,' ' saving model to %s' % (
                                epoch + 1, self.monitor, self.best, current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve' %
                                  (epoch + 1, self.monitor))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' %
                          (epoch + 1, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)


def fine_tune_from_resnet(model, train_generator, train_data_x, val_data):
    """Fine tunes the model from the resnet.

    Args:
        model (_type_): model
        train_generator (_type_): train generator
        train_data_x (_type_): train data x
        val_data (_type_): val data

    Returns:
        _type_: Model
    """
    for layer in model.layers:
        layer.trainable = True
    model.compile(
        optimizer=SGD(learning_rate=MODEL_LEARNING_RATE, momentum=MODEL_MOMENTUM,
                      decay=MODEL_DECAY, nesterov=True),
        loss=MODEL_LOSS,
        metrics=['accuracy'])

    tensorboard_all_layers = TensorBoard(
        log_dir=LOGS_DIRECTORY_ALL_LAYERS,
        histogram_freq=0,
        write_graph=True,
        write_grads=False,
        write_images=True)

    def scheduler(epoch):
        updated_lr = K.get_value(model.optimizer.lr) * 0.5
        if (epoch % 3 == 0) and (epoch != 0):
            K.set_value(model.optimizer.lr, updated_lr)
            print(K.get_value(model.optimizer.lr))
        return K.get_value(model.optimizer.lr)

    reduce_lr = LearningRateScheduler(scheduler)

    reduce_lr_plateau = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        mode='auto',
        min_lr=1e-8)

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='auto')

    check_point = ModelCheckpoint(
        filepath=os.path.join(CHECKPOINT_DIRECTORY,
                              'ResNet-50_{epoch:02d}_{val_loss:.2f}.h5'),
        monitor='val_loss',
        save_best_only=True,
        mode='auto',
        period=1)

    model.fit(
        generator=train_generator,
        steps_per_epoch=len(train_data_x) // BATCH_SIZE,
        epochs=EPOCHS_ALL_LAYERS,
        validation_data=val_data,
        callbacks=[tensorboard_all_layers, reduce_lr, reduce_lr_plateau, early_stop, check_point])
    return model


def save_model(model, model_path):
    """saves the model.

    Args:
        model (_type_): model
        model_path (_type_): model path
    """
    model.save(model_path)


def main():
    """main function.
    """
    base_model, model = model_creation_from_vgg_face()
    train_generator, train_data_x, val_data = perform_data_preparation()
    updated_model = training_pre_layers(
        base_model, model, train_generator, train_data_x, val_data)
    trained_model = fine_tune_from_resnet(
        updated_model, train_generator, train_data_x, val_data)
    save_model(trained_model, RESNET_MODEL_PATH)


if __name__ == '__main__':
    main()
