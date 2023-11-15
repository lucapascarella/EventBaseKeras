import math
import os
import re
import json

import numpy as np
from keras_preprocessing.image import ImageDataGenerator

from unipath import Path
from keras import backend
from keras import Sequential
from keras.utils import Sequence
from keras.layers import RandomRotation, RandomZoom, RandomTranslation, RandomBrightness
from typing import Tuple, List
import concurrent.futures
import img_utils
from utils import normalize_nparray


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

    def __init__(self, input_dir_data: str, frame_mode: str = 'dvs', is_training: bool = False, image_shape: Tuple[int, int, int] = (224, 224, 3), batch_size: int = 32,
                 shuffle: bool = True, augment_data: bool = False, dvs_repeat_channel: bool = True):
        # Initialization
        self.is_training = is_training
        self.image_size = image_shape
        self.crop_size = image_shape
        self.dvs_repeat_ch = dvs_repeat_channel
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.follow_links = False
        self.augment_data = augment_data

        # Check that the given frame is supported
        if frame_mode not in {'dvs', 'aps', 'aps_diff', 'cmb', 'dbl'}:
            raise ValueError('Invalid frame mode:', frame_mode, '; expected "dvs", "aps", "aps_diff", "cmb", or "dbl"')
        self.frame_mode = frame_mode

        self.scaler_json = os.path.join(Path(input_dir_data).parent, 'scaler.json')

        if self.frame_mode in ['dvs', 'cmb', 'dbl']:
            self.datagen = ImageDataGenerator()
            if self.frame_mode == 'dbl' or self.frame_mode == 'cmb':
                self.datagen_aps = ImageDataGenerator(rescale=1. / 255)
            percentile_filename = os.path.join(Path(input_dir_data).parent, 'percentiles.txt')
            try:
                self.event_percentiles = np.loadtxt(percentile_filename, usecols=0, skiprows=1)
            except IOError:
                raise IOError("Percentile file {} not found".format(percentile_filename))
        elif self.frame_mode == 'aps':
            if self.is_training:
                self.datagen = ImageDataGenerator(rotation_range=0.2, rescale=1. / 255, width_shift_range=0.2, height_shift_range=0.2)
            else:
                self.datagen = ImageDataGenerator(rescale=1. / 255)
            self.event_percentiles = None
        else:  # 'aps_diff'
            if self.is_training:
                self.datagen = ImageDataGenerator(rotation_range=0.2, width_shift_range=0.2, height_shift_range=0.2)
            else:
                self.datagen = ImageDataGenerator()
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

                    frame_mode = self.frame_mode if self.frame_mode != 'dbl' else 'dvs'
                    frame_mode = frame_mode if frame_mode != 'cmb' else 'dvs'
                    sub_dir_level_two = os.path.join(sub_dir_level_one, frame_mode)
                    unsorted_files = []
                    for file in os.listdir(sub_dir_level_two):
                        tmp = os.path.join(sub_dir_level_two, file)
                        if os.path.isfile(tmp) and tmp.endswith('.png'):
                            unsorted_files.append(tmp)

                    # Get an ordered list of frames
                    sorted_files = sorted(unsorted_files, key=lambda fil: int(re.search(r'_(\d+)\.png', fil).group(1)))

                    # Be sure no other files other than PNGs are in the given sub folder
                    # In APS and DVS frame mode (steering.shape[0] - len(sorted_files) == 0); The number of image in the folder matches the steering datapoints
                    # In APS_DIFF (steering.shape[0] - len(sorted_files) == 1); To get the img-diff we removed an image, therefore, we have more steering points than images
                    if steering.shape[0] - len(sorted_files) < 2:
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
        mode = 'training' if self.is_training else 'validation'
        print("Selected {} of {} frames for {}".format(self.samples, steering_datapoints, mode))
        self.indexes = np.arange(self.samples)
        if self.shuffle:
            np.random.shuffle(self.indexes)

        indexes_csv = os.path.join(Path(input_dir_data).parent, "indexes_{}.csv".format(mode))
        with open(indexes_csv, 'w') as idx_file:
            idx_file.write("filename,index,actual_steering,future_steering,actual_output,future_output\n")
            for idx in self.indexes:
                myi = self.input_data[idx]
                idx_file.write("{},{},{},{},{},{}\n".format(myi.filename, idx, myi.actual_steering, myi.future_steering, myi.actual_output, myi.future_output))

    def _recursive_list(self, sub_path):
        return sorted(os.walk(sub_path, followlinks=self.follow_links), key=lambda tpl: tpl[0])

    def _output_normalization(self, outputs: np.array) -> np.array:
        if self.is_training:
            # Get mean and std dev
            means = np.mean(outputs)
            stds = np.std(outputs)
            # 3sigma clipping
            outputs = np.clip(outputs, means - 3 * stds, means + 3 * stds)
            # Get min and max before scaling
            data_min = np.min(outputs)
            data_max = np.max(outputs)

            out_dict = {'means': means.tolist(), 'stds': stds.tolist(), 'mins': data_min.tolist(), 'maxs': data_max.tolist()}

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
            data_min = train_dict['mins']
            data_max = train_dict['maxs']

        # Scaling all values
        return normalize_nparray(outputs, -1, 1, data_min, data_max)

    def read_batch(self, batch_x: np.array, batch_y: np.array, batch_offset: int, batch_indexes: List[int]):
        for i, batch_index in enumerate(batch_indexes):
            x = img_utils.load_img(self.input_data[batch_index].filename, self.frame_mode, self.event_percentiles, self.image_size, self.crop_size, self.dvs_repeat_ch)
            batch_x[batch_offset * len(batch_indexes) + i] = self.datagen.standardize(self.datagen.random_transform(x))
            batch_y[batch_offset * len(batch_indexes) + i] = self.input_data[batch_index].future_output

    def read_batch_dbl(self, batch_x_aps: np.array, batch_x_dvs: np.array, batch_y: np.array, batch_offset: int, batch_indexes: List[int]):
        for i, batch_index in enumerate(batch_indexes):
            filename = self.input_data[batch_index].filename
            x_dvs = img_utils.load_img(filename, 'dvs', self.event_percentiles, self.image_size, self.crop_size, self.dvs_repeat_ch)
            batch_x_aps[batch_offset * len(batch_indexes) + i] = self.datagen.standardize(self.datagen.random_transform(x_dvs))
            x_aps = img_utils.load_img(filename.replace('/dvs/', '/aps/'), 'aps', self.event_percentiles, self.image_size, self.crop_size, self.dvs_repeat_ch)
            batch_x_dvs[batch_offset * len(batch_indexes) + i] = self.datagen_aps.standardize(self.datagen.random_transform(x_aps))
            batch_y[batch_offset * len(batch_indexes) + i] = self.input_data[batch_index].future_output

    def read_batch_cmb(self, batch_x: np.array, batch_y: np.array, batch_offset: int, batch_indexes: List[int]):
        for i, batch_index in enumerate(batch_indexes):
            filename = self.input_data[batch_index].filename
            x_dvs = img_utils.load_img(filename, 'dvs', self.event_percentiles, self.image_size, self.crop_size, self.dvs_repeat_ch)
            x_dvs = self.datagen.standardize(self.datagen.random_transform(x_dvs))
            x_aps = img_utils.load_img(filename.replace('/dvs/', '/aps/'), 'aps', self.event_percentiles, self.image_size, self.crop_size, self.dvs_repeat_ch)
            x_aps = self.datagen_aps.standardize(self.datagen.random_transform(x_aps))
            x = np.zeros((x_dvs.shape[0], x_dvs.shape[1], x_dvs.shape[2]), dtype=backend.floatx())
            x[:, :, 0] = x_dvs[:, :, 0]
            x[:, :, 1] = x_aps[:, :, 1]
            x[:, :, 2] = x_dvs[:, :, 2]
            batch_x[batch_offset * len(batch_indexes) + i] = x
            batch_y[batch_offset * len(batch_indexes) + i] = self.input_data[batch_index].future_output

    def __getitem__(self, index):
        # get batch indexes from shuffled indexes
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        if self.frame_mode == 'dbl':
            batch_x_aps = np.zeros((self.batch_size, self.image_size[0], self.image_size[1], self.image_size[2]), dtype=backend.floatx())
            batch_x_dvs = np.zeros((self.batch_size, self.image_size[0], self.image_size[1], self.image_size[2]), dtype=backend.floatx())
            batch_y = np.zeros((self.batch_size, 1), dtype=backend.floatx())

            number_of_workers = 4
            chunks = self.split_in_chunks(batch_indexes, number_of_workers)
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(chunks)) as executor:
                jobs = {executor.submit(self.read_batch_dbl, batch_x_aps, batch_x_dvs, batch_y, idx, chunk): chunk for idx, chunk in enumerate(chunks)}
                for future in concurrent.futures.as_completed(jobs):
                    future.result()

            return [batch_x_aps, batch_x_dvs], batch_y
        elif self.frame_mode == 'cmb':
            batch_x = np.zeros((self.batch_size, self.image_size[0], self.image_size[1], self.image_size[2]), dtype=backend.floatx())
            batch_y = np.zeros((self.batch_size, 1), dtype=backend.floatx())

            number_of_workers = 4
            chunks = self.split_in_chunks(batch_indexes, number_of_workers)
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(chunks)) as executor:
                jobs = {executor.submit(self.read_batch_cmb, batch_x, batch_y, idx, chunk): chunk for idx, chunk in enumerate(chunks)}
                for future in concurrent.futures.as_completed(jobs):
                    future.result()
            return batch_x, batch_y
        else:
            batch_x = np.zeros((self.batch_size, self.image_size[0], self.image_size[1], self.image_size[2]), dtype=backend.floatx())
            batch_y = np.zeros((self.batch_size, 1), dtype=backend.floatx())

            number_of_workers = 4
            chunks = self.split_in_chunks(batch_indexes, number_of_workers)
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(chunks)) as executor:
                jobs = {executor.submit(self.read_batch, batch_x, batch_y, idx, chunk): chunk for idx, chunk in enumerate(chunks)}
                for future in concurrent.futures.as_completed(jobs):
                    future.result()

            if self.augment_data:
                # Emulate ImageDataGenerator => random_transform(x) and standardize(x)
                batch_x_aug = np.zeros((self.batch_size, self.image_size[0], self.image_size[1], self.image_size[2]), dtype=backend.floatx())
                # plt.figure(figsize=(10, 10))
                for i in range(batch_x.shape[0]):
                    # Does not work without passing the two seconds arguments
                    batch_x_aug[i] = self.data_augmentation(batch_x[i], {"flip_horizontal": False, "flip_vertical": False})
                return batch_x_aug, batch_y
            else:
                return batch_x, batch_y

    def __len__(self):
        # Denotes the number of batches per epoch
        return self.samples // self.batch_size

    def get_filename(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        filenames = []
        for i, batch_index in enumerate(batch_indexes):
            filenames.append(self.input_data[batch_index].filename)
        return filenames

    @staticmethod
    def split_in_chunks(full_list: List[int], num_workers: int) -> List[List[int]]:
        num_chunks = math.ceil(len(full_list) / num_workers)
        chunks = []
        for i in range(0, len(full_list), num_chunks):
            chunks.append(full_list[i:i + num_chunks])

        return chunks

    def on_epoch_end(self):
        # Updates indexes after each epoch
        self.indexes = np.arange(self.samples)
        if self.shuffle:
            np.random.shuffle(self.indexes)
