import tensorflow as tf
import matplotlib.pyplot as plt

from keras import backend
from keras.callbacks import History


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

    plt.gca().set_ylim([0.0001, 1])
    plt.yscale('log')
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')

    # Always save before show
    if filename:
        plt.savefig(filename)

    plt.show()
