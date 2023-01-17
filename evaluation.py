import argparse
import json
import os
import utils
import DataGenerator
from unipath import Path
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.models import Model
from keras import backend
from keras.models import model_from_json, load_model

from img_utils import save_steering_degrees


def explained_variance_1d(ypred, y):
    """
    Var[ypred - y] / var[y].
    https://www.quora.com/What-is-the-meaning-proportion-of-variance-explained-in-linear-regression
    """
    assert y.ndim == 1 and ypred.ndim == 1
    vary = np.var(y)
    return np.nan if vary == 0 else 1 - np.var(y - ypred) / vary


def build_model(model_architecture: str, weights_path: str, model_path: str, batch_size: int = None) -> Model:
    if model_architecture and weights_path:
        # Load the model architecture from file
        with open(model_architecture, 'r') as json_file:
            model = model_from_json(json_file.read())

        # Load a saved checkpoint
        try:
            # Tell the model that the checkpoint will be used to inference only
            model.load_weights(weights_path).expect_partial()
            print("Loaded weights from {}".format(weights_path))
        except ImportError or ValueError as e:
            raise ValueError("Impossible to load weights from file {} due to: {}".format(weights_path, e))

    else:
        k_mse = tf.Variable(batch_size, trainable=False, name='k_mse', dtype=tf.int32)
        model = load_model(model_path, custom_objects={'custom_mse': utils.hard_mining_mse(k_mse), 'steering_loss': utils.steering_loss, 'pred_std': utils.pred_std})

    return model


def _main(flags: argparse) -> None:
    img_shape = flags.img_height, flags.img_width, flags.img_depth
    batch_size = flags.batch_size

    test_image_loader = DataGenerator.CustomSequence(flags.test_dir, flags.frame_mode, False, img_shape, batch_size, False, False, flags.dvs_repeat)

    # Create a Keras model
    model = build_model(flags.model_architecture, flags.model_weights, flags.model, batch_size)

    n_step = int(np.ceil(test_image_loader.samples / batch_size))
    steps = np.minimum(n_step, 3000)
    print("Selected {} of {} steps".format(steps, n_step))

    # Get predictions and ground
    y_gt = np.zeros((steps, batch_size), dtype=backend.floatx())
    y_mp = np.zeros((steps, batch_size), dtype=backend.floatx())

    # Create dest img folder
    img_dir = os.path.join("evaluation", "images_{}".format(flags.frame_mode))
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    csv_file = open(os.path.join("evaluation", "steering_{}.csv".format(flags.frame_mode)), 'w')
    csv_file.write("img_idx,real_angle,prediction_angle,RMSE\n")

    # Reload normalizer
    with open(os.path.join(Path(os.path.realpath(flags.test_dir)).parent, 'scaler.json'), 'r') as f:
        scaler_dict = json.load(f)

        data_min = scaler_dict['mins']
        data_max = scaler_dict['maxs']

        # Range of the transformed data
        min_bound = -1.0
        max_bound = 1.0

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
        y_gt_step = np.reshape(gt_steer, (batch_size,))
        y_mp_step = np.reshape(model.predict_on_batch(x), (batch_size,))

        # Undo transformation for ground-truth (only for steering)
        y_gt[step] = utils.normalize_nparray(y_gt_step, data_min, data_max, min_bound, max_bound)

        # Undo transformation for predictIons (only for steering)
        y_mp[step] = utils.normalize_nparray(y_mp_step, data_min, data_max, min_bound, max_bound)

        # Save images with steering overlay
        for i in range(batch_size):
            img_filename = os.path.join(img_dir, "steering_{:04d}.png".format(step * batch_size + i))
            tmp_img = x[1][i] if flags.frame_mode == "cmb" else x[i]
            save_steering_degrees(img_filename, utils.normalize_nparray(tmp_img, 0, 255), y_mp[step][i], y_gt[step][i], flags.frame_mode)
            csv_file.write("{},{},{},{}\n".format(img_filename, y_gt[step][i], y_mp[step][i], np.sqrt(np.square(y_gt[step][i] - y_mp[step][i]))))

    csv_file.close()

    # Reshape matrix
    gt_steer = y_gt.flatten()
    pred_steer = y_mp.flatten()

    # Compute random and constant baselines for steerings
    # random_steerings = random_regression_baseline(steer_gt).ravel()
    # constant_steerings = constant_baseline(steer_gt).ravel()

    plt.plot(pred_steer)
    plt.plot(gt_steer)
    # error_steer = np.sqrt(np.square(pred_steer - gt_steer))
    # plt.plot(error_steer)

    plt.title('Steering prediction {}'.format(flags.frame_mode.upper()))
    plt.ylabel('Steering angle')
    plt.xlabel('Frame')
    plt.gca().set_ylim([-50, 50])
    plt.legend(['Prediction', 'Ground-truth'], loc='upper left')

    plt.savefig("steering_prediction_{}.pdf".format(flags.frame_mode))
    # plt.show()

    error_steer = np.sqrt(np.mean(np.square(pred_steer - gt_steer), axis=0))
    print("RMSE: {:.2f}".format(error_steer))
    ex_variance = explained_variance_1d(pred_steer, gt_steer)
    print("EVA: {:.2f}".format(ex_variance))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--test_dir", help="Folder containing testing experiments", type=str, default=None)
    parser.add_argument("-w", "--model_weights", help="Load the model weights from the native Tensorflow .ckpt format", type=str, default=None)
    parser.add_argument("-a", "--model_architecture", help="Load the model architecture from a JSON file", type=str, default=None)
    parser.add_argument("-m", "--model", help="Load the model from a .model", type=str, default=None)
    parser.add_argument("-s", "--random_seed", help="Set an initial random seed or leave it empty", type=int, default=18)
    parser.add_argument("-f", "--frame_mode", help="Load mode for images, either dvs, aps or aps_diff", type=str, default=None)
    parser.add_argument("-b", "--batch_size", help="Batch size in training and evaluation", type=int, default=64)
    parser.add_argument("-r", '--dvs_repeat', help="True repeats DVS diffs three times, False uses positive, negative, and diffs", type=utils.str2bool, default=True)
    parser.add_argument("-iw", "--img_width", help="Target image width", type=int, default=200)
    parser.add_argument("-ih", "--img_height", help="Target image height", type=int, default=200)
    parser.add_argument("-id", "--img_depth", help="Target image depth", type=int, default=3)
    args = parser.parse_args()

    if args.test_dir is None:
        print("Missing --test_dir parameter")
        exit(-1)

    # if args.model_weights is None:
    #     print("Missing --model_weights parameter")
    #     exit(-1)

    # if args.model_architecture is None:
    #     print("Missing --model_architecture parameter")
    #     exit(-1)

    if args.frame_mode not in ["dvs", "aps", "aps_diff", "cmb"]:
        print("A valid --frame_mode must be selected")
        exit(-1)

    # Set a predefined seed
    if args.random_seed:
        np.random.seed(args.random_seed)
        tf.random.set_seed(args.random_seed)

    _main(args)
