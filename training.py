import argparse
import os
from typing import Tuple

import utils
import log_utils
import DataGenerator

import numpy as np
import tensorflow as tf

from tensorflow import keras
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.applications import ResNet50, EfficientNetB6
from keras.layers import Dense, GlobalAveragePooling2D, Concatenate
from keras.models import model_from_json


def build_model(img_shape: Tuple[int, int, int], output_dim: int, model_architecture: str, weights_path: str, use_imagenet: bool, frame_mode: str) -> Model:
    if model_architecture:
        # Load the model architecture from file
        with open(model_architecture, 'r') as json_file:
            model = model_from_json(json_file.read())
    else:
        # None starts from scratch, 'imagenet' reuses imagenet pre-trained model
        weights = 'imagenet' if use_imagenet else None

        if frame_mode == 'dbl':
            # Combine two inputs, aps and dvs images
            img_input_aps = keras.Input(shape=img_shape)
            base_model_aps = ResNet50(input_tensor=img_input_aps, weights=weights, include_top=False)
            x_aps = GlobalAveragePooling2D()(base_model_aps.output)
            for layer in base_model_aps.layers:
                layer._name = layer.name + str("_aps")

            img_input_dvs = keras.Input(shape=img_shape)
            base_model_dvs = ResNet50(input_tensor=img_input_dvs, weights=weights, include_top=False)
            x_dvs = GlobalAveragePooling2D()(base_model_dvs.output)
            for layer in base_model_dvs.layers:
                layer._name = layer.name + str("_dvs")

            x = Concatenate(axis=1)([x_aps, x_dvs])
            x = Dense(1024, activation='relu')(x)

            # Steering prediction
            output = Dense(output_dim)(x)

            model = Model(inputs=[img_input_aps, img_input_dvs], outputs=[output])
            print(model.summary())
        else:
            # Use a single input at time, aps or dvs
            img_input = keras.Input(shape=img_shape)

            # base_model = ResNet50(input_tensor=img_input, weights=weights, include_top=False)
            base_model = EfficientNetB6(input_tensor=img_input, weights=weights, include_top=False)

            x = GlobalAveragePooling2D()(base_model.output)
            x = Dense(1024, activation='relu')(x)

            # Steering prediction
            output = Dense(output_dim)(x)

            model = Model(inputs=[img_input], outputs=[output])
            # print(model.summary())

        # Save the model's architecture in JSON file
        with open("model_architecture.json", "w") as f:
            f.write(model.to_json())

    if weights_path:
        # Load a saved checkpoint
        try:
            model.load_weights(weights_path)
            print("Loaded model from {}".format(weights_path))
        except ImportError or ValueError as e:
            raise ValueError("Impossible to load weights from file {} due to: {}".format(weights_path, e))

    return model


def train_model(model: Model, train_data_generator: DataGenerator.CustomSequence, val_data_generator: DataGenerator.CustomSequence,
                batch_size: int, learn_rate: float, initial_epoch: int, epochs: int, checkpoint_path: str):
    # Initialize number of samples for hard-mining
    model.k_mse = tf.Variable(batch_size, trainable=False, name='k_mse', dtype=tf.int32)

    # Configure training process
    optimizer = keras.optimizers.legacy.Adam(learning_rate=learn_rate, decay=1e-4)
    model.compile(loss=[utils.hard_mining_mse(model.k_mse)], optimizer=optimizer, metrics=[utils.steering_loss, utils.pred_std])

    # Save model with the lowest validation loss, use Tensorflow native ckpt to allow easy reload later
    # weights_path = os.path.join(checkpoint_path, 'weights_{epoch:03d}.h5')
    weights_path = os.path.join(checkpoint_path, "weights_{epoch:03d}.ckpt")
    write_best_model = ModelCheckpoint(filepath=weights_path, monitor='val_loss', save_best_only=True, save_weights_only=True)

    # Save training and validation losses
    loss_filename = os.path.join(checkpoint_path, "loss.csv")
    save_model_and_loss = log_utils.MyCallback(loss_filename, batch_size, 0.5)

    # Train model
    steps_per_epoch = np.minimum(int(np.ceil(train_data_generator.samples / batch_size) - 1), 5000)
    validation_steps = np.minimum(int(np.ceil(val_data_generator.samples / batch_size)) - 1, np.ceil(int(steps_per_epoch / 10)))

    print("Training: {}. Validation: {}. Batch: {}".format(train_data_generator.samples, val_data_generator.samples, batch_size))
    print("Training steps per epoch {}".format(steps_per_epoch))
    print("Validation steps per epoch {}".format(validation_steps))

    history = model.fit(train_data_generator, batch_size=batch_size,
                        epochs=epochs, steps_per_epoch=steps_per_epoch,
                        callbacks=[write_best_model, save_model_and_loss],
                        validation_data=val_data_generator,
                        validation_steps=validation_steps,
                        initial_epoch=initial_epoch,
                        workers=4)

    # Plot loss
    plot_filename = os.path.join(checkpoint_path, "loss.png")
    utils.plot_history(history, plot_filename)

    # Save the whole model
    model.save(os.path.join(checkpoint_path, "model.model"))


def _main(flags: argparse) -> None:
    frame_mode = flags.frame_mode
    # Always use --img_depth 3 (RGB channel format), even for grayscale that are transformed in RGB later
    img_shape = flags.img_height, flags.img_width, flags.img_depth
    batch_size = flags.batch_size
    learn_rate = flags.learning_rate
    initial_epoch = 0  # Optionally used to restart learning from checkpoint
    epochs = flags.epochs
    checkpoint_path = flags.checkpoints
    use_imagenet_pretrain = flags.use_pretrain
    use_augmentation = flags.use_augmentation
    dvs_repeat_channels = flags.dvs_repeat

    # Remove trailing os separator
    train_dir = flags.train_dir[:-1] if flags.train_dir.endswith(os.sep) else flags.train_dir
    val_dir = flags.val_dir[:-1] if flags.val_dir.endswith(os.sep) else flags.val_dir

    # Generate training data with real-time augmentation
    train_image_loader = DataGenerator.CustomSequence(train_dir, frame_mode, True, img_shape, batch_size, True, use_augmentation, dvs_repeat_channels)
    val_image_loader = DataGenerator.CustomSequence(val_dir, frame_mode, False, img_shape, batch_size, True, False, dvs_repeat_channels)

    # Create the checkpoint directory if it does not already exist
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    # Create the sub checkpoint directory if it does not already exist
    checkpoint_path = os.path.join(checkpoint_path, "epochs_{}_batch_{}_{}".format(epochs, batch_size, frame_mode))
    if use_imagenet_pretrain:
        checkpoint_path += "_imgnet"
    if use_augmentation:
        checkpoint_path += "_aug"
    if dvs_repeat_channels:
        checkpoint_path += "_dvsrpt"
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    # Create a Keras model
    model = build_model(img_shape, 1, flags.model_architecture, flags.model_weights, use_imagenet_pretrain, frame_mode)
    train_model(model, train_image_loader, val_image_loader, batch_size, learn_rate, initial_epoch, epochs, checkpoint_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train_dir", help="Folder containing training experiments", type=str, default=None)
    parser.add_argument("-v", "--val_dir", help="Folder containing validation experiments", type=str, default=None)
    parser.add_argument("-c", "--checkpoints", help="Folder in which save checkpoints", type=str, default="checkpoints")
    parser.add_argument("-w", "--model_weights", help="Load the model weights from a HDF5 checkpoint", type=str, default=None)
    parser.add_argument("-m", "--model_architecture", help="Load the model architecture from a JSON file", type=str, default=None)
    parser.add_argument("-s", "--random_seed", help="Set an initial random seed or leave it empty", type=int, default=18)
    parser.add_argument("-f", "--frame_mode", help="Load mode for images, either dvs, aps, aps_diff, cmb, dbl", type=str, default=None)
    parser.add_argument("-b", "--batch_size", help="Batch size in training and evaluation", type=int, default=64)
    parser.add_argument("-e", "--epochs", help="Number of epochs for training", type=int, default=30)
    parser.add_argument("-l", '--learning_rate', help="Initial learning rate for adam", type=float, default=1e-4)
    parser.add_argument("-p", '--use_pretrain', help="Load Imagenet pre-trained weights", type=utils.str2bool, default=True)
    parser.add_argument("-a", '--use_augmentation', help="Augment images while loading", type=utils.str2bool, default=False)
    parser.add_argument("-r", '--dvs_repeat', help="True repeats DVS diffs three times, False uses positive, negative, and diffs", type=utils.str2bool, default=True)
    parser.add_argument("-iw", "--img_width", help="Target image width", type=int, default=200)
    parser.add_argument("-ih", "--img_height", help="Target image height", type=int, default=200)
    parser.add_argument("-id", "--img_depth", help="Target image depth", type=int, default=3)
    args = parser.parse_args()

    if args.train_dir is None:
        print("Missing --train_dir parameter")
        exit(-1)

    if args.val_dir is None:
        print("Missing --val_dir parameter")
        exit(-1)

    if not os.path.exists(args.train_dir) or not os.path.isdir(args.train_dir):
        print("--train_dir {} is not a directory".format(args.train_dir))
        exit(-1)

    if not os.listdir(args.train_dir):
        print("--train_dir {} is empty".format(args.train_dir))
        exit(-1)

    if args.frame_mode is None or args.frame_mode not in ["dvs", "aps", "aps_diff", "cmb", "dbl"]:
        print("A valid --frame_mode must be selected")
        exit(-1)

    # Set a predefined seed
    if args.random_seed:
        np.random.seed(args.random_seed)
        tf.random.set_seed(args.random_seed)

    _main(args)
