import argparse

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from keras import backend
from keras.callbacks import History


def steering_loss(y_true, y_pred):
    return tf.reduce_mean(backend.square(y_pred - y_true))


def pred_std(y_true, y_pred):
    _, var = tf.nn.moments(y_pred, axes=[0])
    return tf.sqrt(var)


def hard_mining_mse(k):
    """
    Compute Mean Squared Error (MSE) for steering evaluation and hard-mining for the current batch.

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


def normalize_nparray(data: np.array, min_bound: float, max_bound: float, min_data: float = None, max_data: float = None) -> np.array:
    # If not passed, find a local data min/max
    if min_data is None:
        min_data = data.min()
    if max_data is None:
        max_data = data.max()

    val_range = max_data - min_data
    tmp = (data - min_data) / val_range
    return tmp * (max_bound - min_bound) + min_bound


def plot_history(history: History, filename: str = None) -> None:
    # Add data
    legend = []
    if 'loss' in history.history:
        plt.plot(history.history['loss'])
        legend.append('Train')

    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'])
        legend.append('Validation')

    plt.legend(legend, loc='upper left')

    plt.gca().set_ylim([0.00001, 1])
    plt.yscale('log')
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')

    # Always save before show
    if filename:
        plt.savefig(filename)

    plt.show()


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
