import os
import re
import json

import numpy as np

from unipath import Path
import tensorflow
from tensorflow import keras
from keras import backend
from keras import Sequential
from keras.utils import Sequence
from keras.layers import RandomRotation, RandomZoom, RandomTranslation, RandomBrightness
from typing import Tuple

import img_utils

import matplotlib.pyplot as plt


class MyInput:
    def __init__(self, frame_number: int, filename: str,
                 actual_steering: float, actual_torque: float, actual_engine_velocity: float, actual_vehicle_velocity: float,
                 future_steering: float, future_torque: float, future_engine_velocity: float, future_vehicle_velocity: float):
        self.frame_number = frame_number
        self.filename = filename

        self.actual_steering = actual_steering
        self.actual_torque = actual_torque
        self.actual_engine_velocity = actual_engine_velocity
        self.actual_vehicle_velocity = actual_vehicle_velocity

        self.future_steering = future_steering
        self.future_torque = future_torque
        self.future_engine_velocity = future_engine_velocity
        self.future_vehicle_velocity = future_vehicle_velocity

        self.actual_output = None  # Normalized actual steering between -1.0 and 1.0
        self.future_output = None  # Normalized future steering between -1.0 and 1.0


class CustomSequence(Sequence):
    """
        Class for runtime image loading
        It assumes that the input folder structure is:
        root_folder/
               folder_1/
                        dvs/ aps/ aps_diff/
                        sync_steering
               folder_2/
                        dvs/ aps/ aps_diff/
                        sync_steering
               .
               .
               folder_n/
                        dvs/ aps/ aps_diff/
                        sync_steering
        """

    def __init__(self, input_dir_data: str, frame_mode: str = 'dvs', is_training: bool = False, image_size: Tuple[int, int] = (224, 224), batch_size: int = 32,
                 shuffle: bool = True, seed: int = None, follow_links: bool = False, augment_data: bool = False):
        # Initialization
        self.is_training = is_training
        self.image_size = image_size
        self.crop_size = image_size
        self.img_channels = 3
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.follow_links = follow_links
        self.augment_data = augment_data

        # Check that the given frame is supported
        if frame_mode not in {'dvs', 'aps', 'aps_diff'}:
            raise ValueError('Invalid frame mode:', frame_mode, '; expected "dvs", "aps", or "aps_diff".')
        self.frame_mode = frame_mode

        self.scaler_json = os.path.join(Path(input_dir_data).parent, 'scaler.json')

        if self.frame_mode == 'dvs':
            percentile_filename = os.path.join(Path(input_dir_data).parent, 'percentiles.txt')
            try:
                self.event_percentiles = np.loadtxt(percentile_filename, usecols=0, skiprows=1)
            except IOError:
                raise IOError("Percentile file {} not found".format(percentile_filename))
        else:
            self.event_percentiles = None

        # Get a list of input data folders
        steering_datapoints = 0
        self.input_data = []
        for sub_dir_level_one in sorted(os.listdir(input_dir_data)):
            sub_dir_level_one = os.path.join(input_dir_data, sub_dir_level_one)
            if os.path.isdir(sub_dir_level_one):

                # Check if input data are organized properly as in the required sub folder structure
                steering_filename = os.path.join(sub_dir_level_one, 'sync_steering.txt')
                if {'dvs', 'aps', 'aps_diff'}.issubset(os.listdir(sub_dir_level_one)) and os.path.isfile(steering_filename):
                    steering = np.loadtxt(steering_filename, delimiter=',', skiprows=1)
                    steering_datapoints += steering.shape[0]

                    sub_dir_level_two = os.path.join(sub_dir_level_one, self.frame_mode)
                    unsorted_files = []
                    for file in os.listdir(sub_dir_level_two):
                        tmp = os.path.join(sub_dir_level_two, file)
                        if os.path.isfile(tmp) and tmp.endswith('.png'):
                            unsorted_files.append(tmp)

                    # Get an ordered list of frames
                    sorted_files = sorted(unsorted_files, key=lambda fil: int(re.search(r'_(\d+)\.png', fil).group(1)))

                    # Be sure no other files other than PNGs are in the given sub folder
                    if steering.shape[0] == len(sorted_files):
                        for index, filename in enumerate(sorted_files):
                            offset = 6  # This is more or less 1/3s in the future!
                            if index + offset < steering.shape[0]:
                                if self.is_training:
                                    # Filter those images whose velocity is under 23 km/h and 30% of images whose steering is under 5 (for training)
                                    if np.abs(steering[index][3]) >= 2.3e1 and (np.abs(steering[index][0]) >= 5.0 or np.random.random() <= 0.3):
                                        actual_steering, actual_torque, actual_engine, actual_vehicle = steering[index]
                                        future_steering, future_torque, future_engine, future_vehicle = steering[index + offset]
                                        self.input_data.append(MyInput(index, filename, actual_steering, actual_torque, actual_engine, actual_vehicle,
                                                                       future_steering, future_torque, future_engine, future_vehicle))
                                else:
                                    # Filter those images whose velocity is under 15 km/h (for evaluation)
                                    if np.abs(steering[index + offset][3]) >= 1.5e1:
                                        actual_steering, actual_torque, actual_engine, actual_vehicle = steering[index]
                                        future_steering, future_torque, future_engine, future_vehicle = steering[index + offset]
                                        self.input_data.append(MyInput(index, filename, actual_steering, actual_torque, actual_engine, actual_vehicle,
                                                                       future_steering, future_torque, future_engine, future_vehicle))

                    else:
                        raise ValueError("Mismatch between frames in {} and rows in {}".format(sub_dir_level_two, steering_filename))
                else:
                    raise ValueError("Invalid frame/folder structure: {}".format(sub_dir_level_one))

        # Normalize actual output/prediction
        np_actual_outputs = self._output_normalization(np.array([i.actual_steering for i in self.input_data], dtype=backend.floatx()))
        for idx, out in enumerate(np_actual_outputs):
            self.input_data[idx].actual_output = out

        # Normalize future output/prediction
        np_future_outputs = self._output_normalization(np.array([i.future_steering for i in self.input_data], dtype=backend.floatx()))
        for idx, out in enumerate(np_future_outputs):
            self.input_data[idx].future_output = out

        # Used to emulate ImageDataGenerator => random_transform(x) and standardize(x)
        self.data_augmentation = Sequential([
            # RandomFlip("horizontal_and_vertical")
            RandomRotation(0.02),
            RandomZoom(0.1, 0.05),
            RandomTranslation(0.01, 0.01),
            RandomBrightness(0.1),
            # RandomContrast(0.2),
        ])

        self.samples = len(self.input_data)
        print("Selected {} of {} frames for {}".format(self.samples, steering_datapoints, 'training' if is_training else 'validation'))
        self.indexes = np.arange(self.samples)
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def _recursive_list(self, sub_path):
        return sorted(os.walk(sub_path, followlinks=self.follow_links), key=lambda tpl: tpl[0])

    def _output_normalization(self, outputs: np.array) -> np.array:
        """
        Normalize input array between -1 and 1.

        # Arguments
            array: input array.
            directory:

        # Returns
            array: normalized array.
        """
        if self.is_training:
            # Get mean and std dev
            means = np.mean(outputs)
            stds = np.std(outputs)
            # 3sigma clipping
            outputs = np.clip(outputs, means - 3 * stds, means + 3 * stds)
            # Get min and max before scaling
            clip_min = np.min(outputs)
            clip_max = np.max(outputs)

            # Scaling all values
            outputs /= np.max(np.abs(outputs), axis=0)

            out_dict = {'means': means.tolist(),
                        'stds': stds.tolist(),
                        'mins': clip_min.tolist(),
                        'maxs': clip_max.tolist()}

            # Save dictionary for later testing
            with open(self.scaler_json, 'w') as f:
                json.dump(out_dict, f)

        else:
            # Read dictionary
            with open(self.scaler_json, 'r') as f:
                train_dict = json.load(f)

            # 3sigma clipping
            means = train_dict['means']
            stds = train_dict['stds']
            outputs = np.clip(outputs, means - 3 * stds, means + 3 * stds)

            # Scaling of all values
            mins = np.array(train_dict['mins'])
            maxs = np.array(train_dict['maxs'])

            # Range of the transformed data
            min_bound = -1.0
            max_bound = 1.0

            outputs = (outputs - mins) / (maxs - mins)
            outputs = outputs * (max_bound - min_bound) + min_bound

        return outputs

    def __getitem__(self, index):
        # get batch indexes from shuffled indexes
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        batch_x = np.zeros((self.batch_size, self.image_size[0], self.image_size[1], self.img_channels), dtype=backend.floatx())
        batch_y = np.zeros((self.batch_size, 1), dtype=backend.floatx())

        for i, batch_index in enumerate(batch_indexes):
            batch_x[i] = img_utils.load_img(self.input_data[batch_index].filename, self.frame_mode, self.event_percentiles, self.image_size, self.crop_size)
            batch_y[i] = self.input_data[batch_index].future_output
            # print("{:>3}) File {} {} {}".format(self.input_data[batch_index].frame_number, self.input_data[batch_index].filename,
            #                                     self.input_data[batch_index].actual_steering, self.input_data[batch_index].actual_output))

        if self.augment_data:
            # Emulate ImageDataGenerator => random_transform(x) and standardize(x)
            batch_x_aug = np.zeros((self.batch_size, self.image_size[0], self.image_size[1], self.img_channels), dtype=backend.floatx())
            plt.figure(figsize=(10, 10))
            for i in range(batch_x.shape[0]):
                # Does not work without passing the two seconds arguments
                batch_x_aug[i] = self.data_augmentation(batch_x[i], {"flip_horizontal": False, "flip_vertical": False})
                # # Debug, show augmented images
                # ax = plt.subplot(1, 2, 1)
                # plt.imshow(batch_x[i].astype("int"))
                # ax = plt.subplot(1, 2, 2)
                # plt.imshow(batch_x_aug[i].astype("int"))
                # plt.axis("off")
                # plt.show()
            return batch_x_aug, batch_y
        else:
            return batch_x, batch_y

    def __len__(self):
        # Denotes the number of batches per epoch
        return self.samples // self.batch_size

    def on_epoch_end(self):
        # Updates indexes after each epoch
        self.indexes = np.arange(self.samples)
        if self.shuffle:
            np.random.shuffle(self.indexes)
