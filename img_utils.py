import cv2
import numpy as np


def load_img(path, frame_mode, percentiles=None, target_size=None, crop_size=None, dvs_repeat_channel=True):
    """
    Load an image.

    # Arguments
        path: Path to image file.
        percentiles: some percentiles, in particular the median, an inferior percentile,
            and a superior percentile for both positive and negative events in order
            to remove outliers and normalize DVS images.
            Array containing [pos_median, pos_inf, pos_sup, neg_median, neg_inf, neg_sup].
        target_size: Either `None` (default to original size)
            or tuple of ints `(img_height, img_width)`.
        crop_size: Either `None` (default to original size)
            or tuple of ints `(img_height, img_width)`.

    # Returns
        Image as numpy array.
    """

    # Read the raw input image
    img = cv2.imread(path)

    if frame_mode == 'dvs':
        # Extract percentiles of interest to normalize between 0 and 1
        pos_sup = percentiles[2]  # Superior percentile for positive events
        neg_sup = percentiles[5]  # Superior percentile for negative events

        if crop_size:
            img = image_crop(img, crop_size[0], crop_size[1])

        # Extract positive-event image
        pos_events = img[:, :, 0]
        norm_pos_img = pos_events / pos_sup
        norm_pos_img = np.expand_dims(norm_pos_img, axis=-1)

        # Extract negative-event image
        neg_events = img[:, :, -1]
        norm_neg_img = neg_events / neg_sup
        norm_neg_img = np.expand_dims(norm_neg_img, axis=-1)

        # Diff events
        input_img_diff = (norm_pos_img - norm_neg_img)

        if dvs_repeat_channel:
            input_img = np.repeat(input_img_diff, 3, axis=2)
        else:
            input_img = np.concatenate((norm_pos_img, input_img_diff, norm_neg_img), axis=-1)

    elif frame_mode == 'aps':
        if len(img.shape) != 3:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        if crop_size:
            img = image_crop(img, crop_size[0], crop_size[1])

        if target_size:
            ts = (target_size[0], target_size[1])
            if (img.shape[0], img.shape[1]) != ts:
                img = cv2.resize(img, ts)

        # input_img = img.reshape((img.shape[0], img.shape[1], 1))
        input_img = img

    else:
        max_diff = np.log(255 + 1e-3) - np.log(0 + 1e-3)
        # min_diff = np.log(0 + 1e-3) -  np.log(255 + 1e-3)

        if crop_size:
            img = image_crop(img, crop_size[0], crop_size[1])

        if target_size:
            ts = (target_size[0], target_size[1])
            if (img.shape[0], img.shape[1]) != ts:
                img = cv2.resize(img, ts)

        input_img = (np.log(cv2.cvtColor(img[:, :, -1], cv2.COLOR_GRAY2RGB) + 1e-3) - np.log(cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2RGB) + 1e-3)) / max_diff
        # input_img = (np.log(img[:,:,-1] + 1e-3) - np.log(img[:,:,0] + 1e-3))/max_diff
        # input_img = input_img.reshape((input_img.shape[0], input_img.shape[1], 1))

    return np.asarray(input_img, dtype=np.float32)


def image_crop(img, crop_height=200, crop_width=346):
    """
    Crop the input image centered in width and starting from the top
    in height to remove the hood and dashboard of the car.

    # Arguments:
        crop_width: Width of the crop.
        crop_heigth: Height of the crop.

    # Returns:
        Cropped image.
    """
    half_the_width = int(img.shape[1] / 2)
    img = img[0: crop_height, half_the_width - int(crop_width / 2): half_the_width + int(crop_width / 2)]
    return img


def save_steering_degrees(img_filename: str, img: np.array, pred_steer: float, real_steer: float, mode: str) -> None:
    c, r = (100, 100), 65  # center, radius
    t_offset = 20

    # Draw circle
    cv2.circle(img, c, r, (255, 255, 255), 1, lineType=cv2.LINE_AA)
    cv2.line(img, (c[0] - r + 5, c[1]), (c[0] - r, c[1]), (255, 255, 255), 1, lineType=cv2.LINE_AA)
    cv2.line(img, (c[0] + r - 5, c[1]), (c[0] + r, c[1]), (255, 255, 255), 1, lineType=cv2.LINE_AA)
    cv2.line(img, (c[0], c[1] - r + 5), (c[0], c[1] - r), (255, 255, 255), 1, lineType=cv2.LINE_AA)
    cv2.line(img, (c[0], c[1] + r - 5), (c[0], c[1] + r), (255, 255, 255), 1, lineType=cv2.LINE_AA)

    # Draw real steering
    real_rad = + real_steer / 180. * np.pi + np.pi / 2
    t = (c[0] + int(np.cos(real_rad) * r), c[1] - int(np.sin(real_rad) * r))
    cv2.line(img, c, t, (255, 255, 255), 2, lineType=cv2.LINE_AA)
    cv2.putText(img, 'GT: {:.1f}'.format(real_steer), (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, lineType=cv2.LINE_AA)

    # Draw predicted steering
    pred_rad = + pred_steer / 180. * np.pi + np.pi / 2
    t = (c[0] + int(np.cos(pred_rad) * r), c[1] - int(np.sin(pred_rad) * r))
    cv2.line(img, c, t, (0, 255, 0), 2, lineType=cv2.LINE_AA)
    cv2.putText(img, '{}: {:.1f}'.format(mode.upper(), pred_steer), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, lineType=cv2.LINE_AA)

    cv2.imwrite(img_filename, img)


def save_image(img_filename: str, img: np.array) -> None:
    cv2.imwrite(img_filename, img)
