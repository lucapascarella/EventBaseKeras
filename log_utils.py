# import logz
import os

import numpy as np

from tensorflow import keras
from keras import backend
from keras import callbacks

MIN_LR = 0.00001


class MyCallback(callbacks.Callback):
    """
    Customized callback class.

    # Arguments
       filepath: Path to save model.
       period: Frequency in epochs with which model is saved.
       batch_size: Number of images per batch.
    """

    def __init__(self, filepath: str, batch_size: int, factor: float = 1.0):
        super().__init__()
        self.filepath = filepath
        self.batch_size = batch_size
        self.factor = factor
        self.min_lr = MIN_LR

        self.loss_file = open(os.path.join(filepath, "loss.csv"), "w")
        self.loss_file.write("steering_loss,val_steering_loss\n")

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}

        # Save training and validation losses
        self.loss_file.write("{},{}\n".format(logs.get('steering_loss'), logs.get('val_steering_loss')))
        self.loss_file.flush()

        # Reduce learning_rate in critical conditions
        # std_pred = logs.get('pred_std')
        if 'pred_std' in logs and logs['pred_std'] < 0.05:
            if hasattr(self.model.optimizer, 'learning_rate'):
                current_learning_rate = backend.get_value(self.model.optimizer.learning_rate)
                new_learning_rate = np.maximum(current_learning_rate * self.factor, self.min_lr)
                if not isinstance(new_learning_rate, (float, np.float32, np.float64)):
                    raise ValueError('The output of the "schedule" function should be float.')
                backend.set_value(self.model.optimizer.learning_rate, new_learning_rate)
                print("Reduced learning rate!\n")
            else:
                raise ValueError('Optimizer must have a "learning_rate" attribute.')

        # Hard mining
        sess = backend.get_session()
        mse_function = self.batch_size - (self.batch_size - 10) * (np.maximum(0.0, 1.0 - np.exp(-1.0 / 30.0 * (epoch - 30.0))))
        self.model.k_mse.load(int(np.round(mse_function)), sess)

    def on_train_end(self, logs=None):
        if logs is None:
            logs = {}
        print("Training ended")
