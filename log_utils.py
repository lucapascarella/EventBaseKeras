# import logz
import os

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import backend
from keras import callbacks

MIN_LR = 0.00001


class MyCallback(callbacks.Callback):

    def __init__(self, filepath: str, batch_size: int, factor: float = 1.0):
        super().__init__()
        self.filepath = filepath
        self.batch_size = batch_size
        self.factor = factor
        self.min_lr = MIN_LR

        # self.hl, = plt.plot([], [])
        # self.hl.yscale('log')
        # self.hl.title('Model loss')
        # self.hl.ylabel('Loss')
        # self.hl.xlabel('Epoch')
        # self.hl.legend(['Steering train', 'Steering val.'], loc='upper left')

        self.loss_file = open(os.path.join(filepath, "loss.csv"), "w")
        self.loss_file.write("epoch,steering_loss,val_steering_loss\n")

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}

        # Save training and validation losses
        self.loss_file.write("{},{},{}\n".format(epoch,logs.get('steering_loss'), logs.get('val_steering_loss')))
        self.loss_file.flush()

        # Reduce learning_rate in critical conditions
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
        self.model.k_mse.assign(int(np.round(mse_function)), sess)

        # self.hl.set_xdata(np.append(self.hl.get_xdata(), logs['steering_loss']))
        # self.hl.set_ydata(np.append(self.hl.get_ydata(), logs['val_steering_loss']))
        # plt.draw()

    def on_train_end(self, logs=None):
        if logs is None:
            logs = {}
        self.loss_file.close()
        print("Training ended")
