"""
File Name: constants.py
Author: Jinesh Mehta
File Description: This file contains all the constants used in the project.

"""
import os


EMOTIONS = ['Anger', 'Disgust', 'Fear',
            'Happiness', 'Sadness', 'Surprise', 'Neutral']
MODEL_DIRECTORY = 'Models'
EMOJI_DIRECTORY = 'Emojis'
DATASET_DIRECTORY = 'Dataset'
LOGS_DIRECTORY = 'logs'
CHECKPOINT_DIRECTORY = 'checkpoint'
LOGS_DIRECTORY_PRE_LAYERS = os.path.join(LOGS_DIRECTORY, 'pre_layers')
LOGS_DIRECTORY_ALL_LAYERS = os.path.join(LOGS_DIRECTORY, 'all_layers')
HAAR_CASCADE_PATH = os.path.join(
    MODEL_DIRECTORY, 'haarcascade_frontalface_alt.xml')
RESNET_MODEL_PATH = os.path.join(MODEL_DIRECTORY, 'resnet50.h5')
RESNET_FER_MEAN = 128.8006
RESNET_FER_STD = 64.6497
RESNET_FER_IMG_WIDTH = 197
RESNET_FER_IMG_HEIGHT = 197

# BASE MODEL
BASE_MODEL_NAME = 'resnet50'
BASE_MODEL_INITIAL_WEIGHTS = 'vggface'
# Parameters
NUM_CLASSES = 7
EPOCHS_TOP_LAYERS = 5
EPOCHS_ALL_LAYERS = 100
BATCH_SIZE = 128
FER_TRAIN_DATA_PATH = os.path.join(DATASET_DIRECTORY, 'fer2013_train.csv')
FER_EVAL_DATA_PATH = os.path.join(DATASET_DIRECTORY, 'fer2013_eval.csv')

MODEL_LEARNING_RATE = 1e-4
MODEL_MOMENTUM = 0.9
MODEL_DECAY = 0.0
MODEL_LOSS = 'categorical_crossentropy'
