import argparse
import json
import os

import DataGenerator
from unipath import Path
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras
from keras.models import Model
from keras import backend
from keras.callbacks import History
from keras.models import model_from_json


def build_model(model_architecture: str, weights_path: str) -> Model:
    # Load the model architecture from file
    with open(model_architecture, 'r') as json_file:
        model = model_from_json(json_file.read())
    # Load a saved checkpoint
    try:
        model.load_weights(weights_path)
        print("Loaded model from {}".format(weights_path))
    except ImportError as e:
        print("Impossible to find weight path. Returning untrained model")
    except ValueError as e:
        print("Impossible to find weight path. Returning untrained model")

    return model


def steering_loss(y_true, y_pred):
    return tf.reduce_mean(backend.square(y_pred - y_true))


def pred_std(y_true, y_pred):
    _, var = tf.nn.moments(y_pred, axes=[0])
    return tf.sqrt(var)


def hard_mining_mse(k):
    """
    Compute MSE for steering evaluation and hard-mining for the current batch.

    # Arguments
        k: number of samples for hard-mining.

    # Returns
        custom_mse: average MSE for the current batch.
    """

    def custom_mse(y_true, y_pred):
        # Steering loss
        l_steer = backend.square(y_pred - y_true)
        l_steer = tf.squeeze(l_steer, axis=-1)

        # Hard mining
        k_min = tf.minimum(k, tf.shape(l_steer)[0])
        _, indices = tf.nn.top_k(l_steer, k=k_min)
        max_l_steer = tf.gather(l_steer, indices)
        hard_l_steer = tf.divide(tf.reduce_sum(max_l_steer), tf.cast(k, tf.float32))

        return hard_l_steer

    return custom_mse


def plot_history(history: History) -> None:
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.plot(history.history['steering_loss'])
    plt.plot(history.history['val_steering_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test', 'Steering train', 'Steering test'], loc='upper left')

    plt.savefig("loss.png")
    # plt.show()


def _main(flags: argparse) -> None:
    channels_dict = {"grayscale": 1, "rgb": 3, "rgba": 4}
    img_height, img_width = flags.img_height, flags.img_width
    # img_channels = channels_dict["rgb"]
    batch_size = flags.batch_size

    # Generate training data with real-time augmentation
    if flags.frame_mode == "dvs":
        img_channels = channels_dict["rgb"]
    elif flags.frame_mode == "aps":
        img_channels = channels_dict["grayscale"]
    else:
        img_channels = channels_dict["rgb"]

    test_image_loader = DataGenerator.CustomSequence(flags.test_dir, flags.frame_mode, False, (img_height, img_width), batch_size, False)

    # Create a Keras model
    model = build_model(flags.model_architecture, flags.model_weights)
    # Compile model
    model.compile(loss='mse', optimizer='sgd')

    # Get predictions and ground
    steps = 30

    y_gt = np.zeros((steps, batch_size), dtype=backend.floatx())
    y_mp = np.zeros((steps, batch_size), dtype=backend.floatx())

    for step in range(steps):
        generator_output = test_image_loader.__getitem__(step)
        if isinstance(generator_output, tuple):
            if len(generator_output) == 2:
                x, gt_steer = generator_output
            elif len(generator_output) == 3:
                x, gt_steer, _ = generator_output
            else:
                raise ValueError('Output of generator should be a tuple `(x, y, sample_weight)` or `(x, y)`. Found: ' + str(generator_output))
        else:
            raise ValueError('Output not valid for current evaluation')

        # Append prediction and ground through
        y_gt[step] = gt_steer.flatten()
        y_mp[step] = model.predict_on_batch(x).flatten()

    # Flatten matrix
    y_gt = y_gt.flatten()
    y_mp = y_mp.flatten()

    # Steering boundaries seen in data
    with open(os.path.join(Path(os.path.realpath(flags.test_dir)).parent, 'scaler.json'), 'r') as f:
        scaler_dict = json.load(f)

        mins = np.array(scaler_dict['mins'])
        maxs = np.array(scaler_dict['maxs'])

        # Range of the transformed data
        min_bound = -1.0
        max_bound = 1.0

        # Undo transformation for ground-truth (only for steering)
        gt_std = (y_gt - min_bound) / (max_bound - min_bound)
        steer_gt = gt_std * (maxs - mins) + mins
        # steer_gt = np.expand_dims(gt_steer, axis=-1)

        # Undo transformation for predicitons (only for steering)
        pred_std = (y_mp - min_bound) / (max_bound - min_bound)
        pred_steer = pred_std * (maxs - mins) + mins
        # pred_steer = np.expand_dims(pred_steer, axis = -1)

        # Compute random and constant baselines for steerings
        # random_steerings = random_regression_baseline(steer_gt).ravel()
        # constant_steerings = constant_baseline(steer_gt).ravel()

        plt.plot(pred_steer)
        plt.plot(steer_gt)
        plt.title('Steering angle')
        plt.ylabel('Steering angle')
        plt.xlabel('Frame')
        plt.gca().set_ylim([-110, 110])
        plt.legend(['Prediction', 'Ground-truth'], loc='upper left')

        # plt.savefig("steering_prediction.png")
        plt.show()

        # with open(os.path.join(Path(os.path.realpath(flags.test_dir)).parent, 'pred.csv'), 'w') as f:
        #     pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--test_dir", help="Folder containing testing experiments", type=str, default=None)
    parser.add_argument("-w", "--model_weights", help="Load the model weights from a HDF5 checkpoint", type=str, default=None)
    parser.add_argument("-a", "--model_architecture", help="Load the model architecture from a JSON file", type=str, default=None)
    parser.add_argument("-r", "--random_seed", help="Set an initial random seed or leave it empty", type=int, default=18)
    parser.add_argument("-f", "--frame_mode", help="Load mode for images, either dvs, aps or aps_diff", type=str, default="dvs")
    parser.add_argument("-b", "--batch_size", help="Batch size in training and evaluation", type=int, default=64)
    parser.add_argument("-iw", "--img_width", help="Target image width", type=int, default=200)
    parser.add_argument("-ih", "--img_height", help="Target image height", type=int, default=200)
    args = parser.parse_args()

    y = np.zeros((3, 2), dtype=np.float32)

    a = np.array([1, 2], dtype=np.float32)
    b = np.array([3, 4], dtype=np.float32)

    s = np.stack((a, b), axis=1)
    # s = np.hstack((a,b))

    print(a.shape)
    print(y.shape)

    y[0] = a

    y = y.flatten()
    print(y)

    if args.test_dir is None:
        print("Missing --test_dir parameter")
        exit(-1)

    if args.model_weights is None:
        print("Missing --model_weights parameter")
        exit(-1)

    if args.model_architecture is None:
        print("Missing --model_architecture parameter")
        exit(-1)

    if args.frame_mode not in ["dvs", "aps", "aps_diff"]:
        print("A valid --frame_mode must be selected")
        exit(-1)

    # Set a predefined seed
    if args.random_seed:
        np.random.seed(args.random_seed)
        tf.random.set_seed(args.random_seed)

    _main(args)
