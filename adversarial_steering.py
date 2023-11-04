import argparse
import json
import math
import os
from unipath import Path
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
# from keras.utils.generic_utils import get_custom_objects
from tensorflow.python.keras import backend as kernel
from keras.models import load_model

from img_utils import save_steering_degrees, save_image


# custom activation function for keeping adversarial pixel values between 0.0 and 1.0
def clip(x):
    return kernel.clip(x, 0.0, 1.0)


# custom loss function for non-targeted misclassification
def negative_categorical_crossentropy(y_true, y_pred):
    return 0.0 - kernel.categorical_crossentropy(y_true, y_pred)


# add custom objects to dictionary
keras.utils.get_custom_objects().update({'clip': Activation(clip)})
keras.utils.get_custom_objects().update({'negative_categorical_crossentropy': negative_categorical_crossentropy})


def load_custom_model(model_path: str, batch_size: int) -> Model:
    k_mse = tf.Variable(batch_size, trainable=False, name='k_mse', dtype=tf.int32)
    target_model = load_model(model_path,
                              custom_objects={'custom_mse': utils.hard_mining_mse(k_mse), 'steering_loss': utils.steering_loss, 'pred_std': utils.pred_std})
    return target_model


# function for generating an adversarial example given a base image, adversarial class target, classifier, and regularization type
def generate_adversary(model_path: str, batch_size: int, img: np.array, target: float, regularization: keras.regularizers, loss_function: str) -> Tuple[
    np.array, np.array]:
    img_shape = (200, 200, 3)

    # input for base image
    image = Input(shape=img_shape, name='image')
    # unit input for adversarial noise
    one = Input(shape=(1,), name='unity')

    # layer for learning adversarial noise to apply to image
    noise = Dense(math.prod(img_shape), activation=None, use_bias=False, kernel_initializer='random_normal', kernel_regularizer=regularization,
                  name='adversarial_noise')(one)

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
    # adversarial_model.summary()

    # target adversarial classification
    target_vector = np.asarray(target)
    # target_vector[target] = 1.
    # target_vector[0] = target

    # callback for saving weights with the smallest loss
    tmp_adversarial_filename = "./adversarial_weights.h5"
    if os.path.exists(tmp_adversarial_filename):
        os.remove(tmp_adversarial_filename)
    checkpoint = ModelCheckpoint(tmp_adversarial_filename, monitor='loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', save_freq=1)
    # train adversarial image
    tt = target_vector.reshape(1, -1)
    adversarial_model.fit(x={'image': img, 'unity': np.ones(shape=(1, 1))}, y=tt, epochs=100, verbose=0, callbacks=[checkpoint])
    # restore best weights
    adversarial_model.load_weights(tmp_adversarial_filename)

    # quantize adversarial noise
    weights = adversarial_model.get_weights()
    adv_weights = adversarial_model.get_layer('adversarial_noise').get_weights()[0].reshape(img_shape)
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
        plt.imshow(utils.normalize_nparray(img[0], 0, 1))
        plt.axis('off')
        fig.add_subplot(2, len(images), i + 1 + len(images))
        plt.imshow(utils.normalize_nparray(img[1], 0, 1))
        plt.axis('off')
    plt.show()


def get_image_by_index(batch_size: int, img_index: int) -> Tuple[int, int]:
    img_idx = img_index % batch_size
    batch_idx = img_index // batch_size
    return batch_idx, img_idx


def _main(flags: argparse):
    img_shape = flags.img_height, flags.img_width, flags.img_depth
    input_shape = 1, *img_shape
    batch_size = 32
    model_path = flags.model_path
    img_index = flags.img_index

    # First, prepare the input dataset
    image_loader = DataGenerator.CustomSequence(flags.test_dir, flags.frame_mode, False, img_shape, batch_size, False, False, flags.dvs_repeat)

    # Second, re-load the trained model
    target_model = load_custom_model(model_path, batch_size)

    # Select image to create an adversarial example from
    batch_idx, batch_img_idx = get_image_by_index(batch_size, img_index)
    x_batch, gt_steer_batch = image_loader.__getitem__(batch_idx)
    img = x_batch[batch_img_idx:batch_img_idx + 1]
    gt_steer = np.reshape(gt_steer_batch[batch_img_idx:batch_img_idx + 1], (1,))

    # Rescale image of 1./255
    img = utils.normalize_nparray(img.reshape(img_shape), 0, 1)

    # plt.imshow(img)
    # plt.show()

    # Reload normalizer
    with open(os.path.join(Path(os.path.realpath(flags.test_dir)).parent, 'scaler.json'), 'r') as f:
        scaler_dict = json.load(f)

        data_min = scaler_dict['mins']
        data_max = scaler_dict['maxs']

        # Range of the transformed data
        min_bound = -1.0
        max_bound = 1.0

        img_dir = "adversarial_images"
        if not os.path.exists(img_dir):
            os.mkdir(img_dir)
        img_dir = os.path.join(img_dir, "img_{}_batch_{:03d}_{:02d}_idx_{:04d}".format(flags.frame_mode, batch_idx, batch_img_idx, img_index))
        if not os.path.exists(img_dir):
            os.mkdir(img_dir)

        # Predict
        prediction = target_model.predict(img.reshape(input_shape))[0]

        # Undo transformation for ground-truth (only for steering)
        y_gt = utils.normalize_nparray(gt_steer[0], data_min, data_max, min_bound, max_bound)
        # Undo transformation for predictIons (only for steering)
        y_mp = utils.normalize_nparray(prediction[0], data_min, data_max, min_bound, max_bound)
        print('Actual prediction. Expected {:.1f}, predicted: {:.1f}'.format(y_gt, y_mp))

        # applying random noise does not fool the classifier
        quantized_noise = np.round(np.random.normal(loc=0.0, scale=0.01, size=(img_shape[0], img_shape[1], 1)) * 255.) / 255.
        quantized_noise = np.repeat(quantized_noise, 3, axis=2)

        noisy_img = np.clip(img + quantized_noise, 0., 1.)
        # plt.imshow(noisy_img.reshape(img_shape), vmin=0., vmax=1.)
        # plt.show()
        noisy_prediction = target_model.predict(noisy_img.reshape(input_shape))[0]
        # Undo transformation for predictIons (only for steering)
        y_mp = utils.normalize_nparray(noisy_prediction[0], data_min, data_max, min_bound, max_bound)
        print('Noisy prediction. Expected {:.1f}, predicted: {:.1f}'.format(y_gt, y_mp))
        save_steering_degrees(os.path.join(img_dir, "random_img.png"), utils.normalize_nparray(noisy_img, 0, 255), y_mp, y_gt, flags.frame_mode)
        save_image(os.path.join(img_dir, "random_noise.png"), utils.normalize_nparray(quantized_noise, 0, 255))

        # Non-targeted misclassification image
        regularizations = [l1(0.01), l2(0.01), l1_l2(l1=0.01, l2=0.01)]
        regularization_names = ["l1", "l2", "l1l2"]
        # non_targeted = -gt_steer
        generated_images = []

        for regularization, reg_name in zip(regularizations, regularization_names):
            generated_images.append(generate_adversary(model_path, batch_size, img.reshape(input_shape), -gt_steer[0], regularization, "mean_squared_error"))
            foolish_img = generated_images[-1][0]
            adversarial_prediction = target_model.predict(foolish_img.reshape((1, 200, 200, 3)))[0]
            # Undo transformation for predictIons (only for steering)
            y_mp = utils.normalize_nparray(adversarial_prediction[0], data_min, data_max, min_bound, max_bound)
            print('Foolish prediction with regularization: {}, Expected {:.1f}, predicted: {:.1f}'.format(reg_name, y_gt, y_mp))

            img_filename = os.path.join(img_dir, "adv_img_{}.png".format(reg_name))
            save_steering_degrees(img_filename, utils.normalize_nparray(foolish_img, 0, 255), y_mp, y_gt, flags.frame_mode)
            noise_filename = os.path.join(img_dir, "adv_noise_{}.png".format(reg_name))
            save_image(noise_filename, utils.normalize_nparray(generated_images[-1][1], 0, 255))

        # show_sub_figs("Non targeted", generated_images)

    # # Targeted misclassification image
    # targeted = -gt_steer
    # generated_images = []
    # for regularization in regularizations:
    #     generated_images.append(generate_adversary(model_path, batch_size, img, targeted, regularization, 'categorical_crossentropy'))
    #     adversarial_prediction = target_model.predict(generated_images[-1][0].reshape((1, 200, 200, 3)))
    #     print('Expected {}, predicted: {}'.format(gt_steer, adversarial_prediction))
    # show_sub_figs("Targeted", generated_images)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--test_dir", help="Folder containing testing experiments", type=str, default=None)
    parser.add_argument("-f", "--frame_mode", help="Load mode for images, either dvs, aps or aps_diff", type=str, default=None)
    parser.add_argument("-m", "--model_path", help="Load the model from a .model", type=str, default=None)
    parser.add_argument("-r", '--dvs_repeat', help="True repeats DVS diffs three times, False uses positive, negative, and diffs", type=utils.str2bool,
                        default=True)
    # parser.add_argument("-b", "--batch_size", help="Batch size in training and evaluation", type=int, default=128)
    parser.add_argument("-i", "--img_index", help="Image index", type=int, default=116)
    parser.add_argument("-iw", "--img_width", help="Target image width", type=int, default=200)
    parser.add_argument("-ih", "--img_height", help="Target image height", type=int, default=200)
    parser.add_argument("-id", "--img_depth", help="Target image depth", type=int, default=3)
    args = parser.parse_args()

    _main(args)
