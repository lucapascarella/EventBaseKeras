import argparse
import math
import os

import DataGenerator
import utils
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, List
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Dense, Input
from keras.layers import Reshape, Add
from keras.metrics import categorical_accuracy
from keras.regularizers import l1_l2, l2, l1
from keras.callbacks import ModelCheckpoint
from keras.layers import Activation
from keras.utils.generic_utils import get_custom_objects
from tensorflow.python.keras import backend as kernel
from keras.models import load_model


# custom activation function for keeping adversarial pixel values between 0.0 and 1.0
def clip(x):
    return kernel.clip(x, 0.0, 1.0)


# custom loss function for non-targeted misclassification
def negative_categorical_crossentropy(y_true, y_pred):
    return 0.0 - kernel.categorical_crossentropy(y_true, y_pred)


# add custom objects to dictionary
get_custom_objects().update({'clip': Activation(clip)})
get_custom_objects().update({'negative_categorical_crossentropy': negative_categorical_crossentropy})


def load_custom_model(model_path: str, batch_size: int) -> Model:
    k_mse = tf.Variable(batch_size, trainable=False, name='k_mse', dtype=tf.int32)
    target_model = load_model(model_path, custom_objects={'custom_mse': utils.hard_mining_mse(k_mse), 'steering_loss': utils.steering_loss, 'pred_std': utils.pred_std})
    return target_model


# function for generating an adversarial example given a base image, adversarial class target, classifier, and regularization type
def generate_adversary(model_path: str, batch_size: int, img: np.array, target: float, regularization: keras.regularizers, loss_function: str) -> Tuple[np.array, np.array]:
    img_shape = (200, 200, 3)

    # input for base image
    image = Input(shape=img_shape, name='image')
    # unit input for adversarial noise
    one = Input(shape=(1,), name='unity')

    # layer for learning adversarial noise to apply to image
    noise = Dense(math.prod(img_shape), activation=None, use_bias=False, kernel_initializer='random_normal', kernel_regularizer=regularization, name='adversarial_noise')(one)

    # reshape noise in shape of image
    noise = Reshape(img_shape, name='reshape')(noise)

    # add noise to image
    net = Add(name='add')([noise, image])
    # clip values to be within 0.0 and 1.0
    net = Activation('clip', name='clip_values')(net)

    # feed adversarial image to trained MNIST classifier
    target_model = load_custom_model(model_path, batch_size)
    outputs = target_model(net)

    adversarial_model = Model(inputs=[image, one], outputs=outputs)
    # freeze trained MNIST classifier layers
    adversarial_model.layers[-1].trainable = False

    adversarial_model.compile(optimizer='nadam', loss=loss_function, metrics=[categorical_accuracy])
    adversarial_model.summary()

    # target adversarial classification
    target_vector = np.zeros(1)
    # target_vector[target] = 1.
    target_vector[0] = target

    # callback for saving weights with the smallest loss
    tmp_adversarial_filename = "./adversarial_weights.h5"
    os.remove(tmp_adversarial_filename)
    checkpoint = ModelCheckpoint(tmp_adversarial_filename, monitor='loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', save_freq=1)
    # train adversarial image
    adversarial_model.fit(x={'image': img, 'unity': np.ones(shape=(1, 1))}, y=target_vector.reshape(1, -1), epochs=10000, verbose=0, callbacks=[checkpoint])
    # restore best weights
    adversarial_model.load_weights(tmp_adversarial_filename)

    # quantize adversarial noise
    weights = adversarial_model.get_weights()
    quantized_weights = np.round(weights[0].reshape(img_shape) * 255.) / 255.

    # add trained weights to original image and clip values to produce adversarial image
    adversarial_img = np.clip(img.reshape(img_shape) + quantized_weights, 0., 1.)

    return adversarial_img, quantized_weights


def show_sub_figs(title: str, images: List[Tuple[np.array, np.array]]):
    fig = plt.figure()
    plt.title(title)
    plt.axis('off')
    for i, img in enumerate(images):
        fig.add_subplot(2, len(images), i + 1)
        plt.imshow(img[0])
        plt.axis('off')
        fig.add_subplot(2, len(images), i + 4)
        plt.imshow(img[1])
        plt.axis('off')
    plt.show()


def _main(flags: argparse):
    img_shape = flags.img_height, flags.img_width, flags.img_depth
    batch_size = flags.batch_size
    model_path = flags.model_path

    # First, prepare the input dataset
    image_loader = DataGenerator.CustomSequence(flags.test_dir, flags.frame_mode, False, img_shape, batch_size, False, False, flags.dvs_repeat)

    # Second, re-load the trained model
    target_model = load_custom_model(model_path, batch_size)

    # Select image to create an adversarial example from
    x_batch, gt_steer_batch = image_loader.__getitem__(0)
    img_idx = 105
    img = x_batch[img_idx:img_idx + 1]
    gt_steer = gt_steer_batch[img_idx:img_idx + 1]
    plt.imshow(img.reshape(img_shape), vmin=0., vmax=1.)
    plt.show()

    prediction = target_model.predict(img)[0]
    print('Expected {}, predicted: {}'.format(gt_steer[0], prediction))

    # applying random noise does not fool the classifier
    quantized_noise = np.round(np.random.normal(loc=0.0, scale=0.1, size=(img_shape[0], img_shape[1], 1)) * 255.) / 255.
    quantized_noise = np.repeat(quantized_noise, 3, axis=2)

    noisy_img = np.clip(img + quantized_noise, 0., 1.)
    # noisy_img = img
    plt.imshow(noisy_img.reshape(img_shape), vmin=0., vmax=1.)
    plt.show()
    noisy_prediction = target_model.predict(noisy_img)[0]
    print('Expected {}, predicted: {}'.format(gt_steer[0], noisy_prediction))

    # Non-targeted misclassification image
    regularizations = [l1(0.01), l2(0.01), l1_l2(l1=0.01, l2=0.01)]
    non_targeted = gt_steer
    generated_images = []
    for regularization in regularizations:
        generated_images.append(generate_adversary(model_path, batch_size, img, non_targeted, regularization, 'negative_categorical_crossentropy'))
        adversarial_prediction = target_model.predict(generated_images[-1][0].reshape((1, 200, 200, 3)))
        print('Expected {}, predicted: {}'.format(gt_steer, adversarial_prediction))
    show_sub_figs("Non targeted", generated_images)

    # Targeted misclassification image
    targeted = -gt_steer
    generated_images = []
    for regularization in regularizations:
        generated_images.append(generate_adversary(model_path, batch_size, img, targeted, regularization, 'categorical_crossentropy'))
        adversarial_prediction = target_model.predict(generated_images[-1][0].reshape((1, 200, 200, 3)))
        print('Expected {}, predicted: {}'.format(gt_steer, adversarial_prediction))
    show_sub_figs("Targeted", generated_images)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--test_dir", help="Folder containing testing experiments", type=str, default=None)
    parser.add_argument("-f", "--frame_mode", help="Load mode for images, either dvs, aps or aps_diff", type=str, default=None)
    parser.add_argument("-m", "--model_path", help="Load the model from a .model", type=str, default=None)
    parser.add_argument("-r", '--dvs_repeat', help="True repeats DVS diffs three times, False uses positive, negative, and diffs", type=bool, default=True)
    parser.add_argument("-b", "--batch_size", help="Batch size in training and evaluation", type=int, default=128)
    parser.add_argument("-iw", "--img_width", help="Target image width", type=int, default=200)
    parser.add_argument("-ih", "--img_height", help="Target image height", type=int, default=200)
    parser.add_argument("-id", "--img_depth", help="Target image depth", type=int, default=3)
    args = parser.parse_args()

    _main(args)
