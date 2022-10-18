import argparse
import os

import log_utils
import DataGenerator

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras import backend
from keras.callbacks import ModelCheckpoint, History
from keras.applications import ResNet50
from keras.layers import Dense, GlobalAveragePooling2D
import matplotlib.pyplot as plt


def build_model(img_width: int, img_height: int, img_channels: int, output_dim: int, random_init: bool = True, weights_path: str = None) -> Model:
    img_input = keras.Input(shape=(img_width, img_height, img_channels))
    print(img_input.shape)

    # None starts from scratch, 'imagenet' reuses imagenet pre-trained model
    weights = None if random_init else 'imagenet'

    base_model = ResNet50(input_tensor=img_input, weights=weights, include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    x = Dense(1024, activation='relu')(x)

    # Steering channel
    output = Dense(output_dim)(x)

    model = Model(inputs=[img_input], outputs=[output])
    # print(model.summary())

    if weights_path:
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

    plt.savefig("loss.png", dpi=plt.figure(figsize=(3, 6)).dpi)
    plt.show()


def train_model(model: Model, train_data_generator: DataGenerator.CustomSequence, val_data_generator: DataGenerator.CustomSequence,
                batch_size: int, learn_rate: float, initial_epoch: int, epochs: int, checkpoint_path: str):
    lr_scale_factor = 0.5

    # Create the experiment rootdir if not already there
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    # Initialize number of samples for hard-mining
    model.k_mse = tf.Variable(batch_size, trainable=False, name='k_mse', dtype=tf.int32)

    # Configure training process
    optimizer = keras.optimizers.Adam(learning_rate=learn_rate, decay=1e-4)
    model.compile(loss=[hard_mining_mse(model.k_mse)], optimizer=optimizer, metrics=[steering_loss, pred_std])

    # Save model with the lowest validation loss
    weights_path = os.path.join(checkpoint_path, 'weights_{epoch:03d}.h5')
    write_best_model = ModelCheckpoint(filepath=weights_path, monitor='val_steering_loss', save_best_only=True, save_weights_only=True)

    # Save training and validation losses
    save_model_and_loss = log_utils.MyCallback(checkpoint_path, batch_size, lr_scale_factor)

    # Train model
    steps_per_epoch = np.minimum(int(np.ceil(train_data_generator.samples / batch_size)), 2000)
    validation_steps = int(np.ceil(val_data_generator.samples / batch_size)) - 1

    print("Datasets size. Train: {}, validation: {}, batch: {}".format(train_data_generator.samples, val_data_generator.samples, batch_size))
    print("Step per epoch {}".format(steps_per_epoch))
    print("Validation steps {}".format(validation_steps))

    history = model.fit(train_data_generator, batch_size=batch_size,
                        epochs=epochs, steps_per_epoch=steps_per_epoch - 1,
                        callbacks=[write_best_model, save_model_and_loss],
                        validation_data=val_data_generator,
                        validation_steps=validation_steps,
                        initial_epoch=initial_epoch)

    plot_history(history)


def _main(flags: argparse) -> None:
    channels_dict = {"grayscale": 1, "rgb": 3, "rgba": 4}
    img_height, img_width = flags.img_height, flags.img_width
    # img_channels = channels_dict["rgb"]
    batch_size = flags.batch_size
    learn_rage = flags.learning_rate
    initial_epoch = 0  # Used to restart learning from checkpoint
    epochs = flags.epochs
    checkpoint_path = flags.checkpoints

    # Generate training data with real-time augmentation
    if flags.frame_mode == "dvs":
        img_channels = channels_dict["rgb"]
    elif flags.frame_mode == "aps":
        img_channels = channels_dict["grayscale"]
    else:
        img_channels = channels_dict["rgb"]

    train_image_loader = DataGenerator.CustomSequence(flags.train_dir, flags.frame_mode, True, (img_height, img_width), batch_size, True)
    val_image_loader = DataGenerator.CustomSequence(flags.test_dir, flags.frame_mode, False, (img_height, img_width), batch_size, True)

    # Create a Keras model
    model = build_model(img_height, img_width, img_channels, 1, False, flags.model_weights)

    # Save model's architecture in JSON file
    with open(os.path.join(checkpoint_path, "model.json"), "w") as f:
        f.write(model.to_json())

    train_model(model, train_image_loader, val_image_loader, batch_size, learn_rage, initial_epoch, epochs, checkpoint_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train_dir", help="Folder containing training experiments", type=str, default=None)
    parser.add_argument("-v", "--test_dir", help="Folder containing testing experiments", type=str, default=None)
    parser.add_argument("-c", "--checkpoints", help="Folder in which save checkpoints", type=str, default="checkpoints")
    parser.add_argument("-w", "--model_weights", help="HDF5 file containing the saved model weights", type=str, default=None)
    parser.add_argument("-r", "--random_seed", help="Set an initial random seed or leave it empty", type=int, default=18)
    parser.add_argument("-f", "--frame_mode", help="Load mode for images, either dvs, aps or aps_diff", type=str, default="dvs")
    parser.add_argument("-b", "--batch_size", help="Batch size in training and evaluation", type=int, default=32)
    parser.add_argument("-e", "--epochs", help="Number of epochs for training", type=int, default=50)
    parser.add_argument("-l", '--learning_rate', help="Initial learning rate for adam", type=float, default=1e-4)
    parser.add_argument("-iw", "--img_width", help="Target Image Width", type=int, default=200)
    parser.add_argument("-ih", "--img_height", help="Target Image Height", type=int, default=200)
    args = parser.parse_args()

    if args.train_dir is None:
        print("Missing --train_dir parameter")
        exit(-1)

    if args.test_dir is None:
        print("Missing --test_dir parameter")
        exit(-1)

    if not os.path.exists(args.train_dir) or not os.path.isdir(args.train_dir):
        print("--train_dir {} is not a directory".format(args.train_dir))
        exit(-1)

    if not os.listdir(args.train_dir):
        print("--train_dir {} is empty".format(args.train_dir))
        exit(-1)

    if args.frame_mode not in ["dvs", "aps", "aps_diff"]:
        print("A valid --frame_mode must be selected")
        exit(-1)

    # Set seed
    if args.random_seed:
        np.random.seed(args.random_seed)
        tf.random.set_seed(args.random_seed)

    _main(args)
