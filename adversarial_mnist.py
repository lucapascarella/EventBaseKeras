import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, List
from tensorflow import keras
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, Reshape, Add
from keras.metrics import categorical_accuracy
from keras.regularizers import l1_l2, l2, l1
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Activation
from keras.utils.generic_utils import get_custom_objects
from tensorflow.python.keras import backend as kernel


# custom activation function for keeping adversarial pixel values between 0.0 and 1.0
def clip(x):
    return kernel.clip(x, 0.0, 1.0)


# custom loss function for non-targeted misclassification
def negative_categorical_crossentropy(y_true, y_pred):
    return 0.0 - kernel.categorical_crossentropy(y_true, y_pred)


# add custom objects to dictionary
get_custom_objects().update({'clip': Activation(clip)})
get_custom_objects().update({'negative_categorical_crossentropy': negative_categorical_crossentropy})


# function for generating an adversarial example given a base image, adversarial class target, classifier, and regularization type
def generate_adversary(save_model_dir: str, img: np.array, target: int, regularization: keras.regularizers, loss_function: str) -> np.array:
    # input for base image
    image = Input(shape=(28, 28, 1), name='image')
    # unit input for adversarial noise
    one = Input(shape=(1,), name='unity')

    # layer for learning adversarial noise to apply to image
    noise = Dense(28 * 28, activation=None, use_bias=False, kernel_initializer='random_normal', kernel_regularizer=regularization, name='adversarial_noise')(one)

    # reshape noise in shape of image
    noise = Reshape((28, 28, 1), name='reshape')(noise)

    # add noise to image
    net = Add(name='add')([noise, image])
    # clip values to be within 0.0 and 1.0
    net = Activation('clip', name='clip_values')(net)

    # feed adversarial image to trained MNIST classifier
    mnist_model = keras.models.load_model(save_model_dir)
    outputs = mnist_model(net)

    adversarial_model = Model(inputs=[image, one], outputs=outputs)
    # freeze trained MNIST classifier layers
    adversarial_model.layers[-1].trainable = False

    adversarial_model.compile(optimizer='nadam', loss=loss_function, metrics=[categorical_accuracy])

    # target adversarial classification
    target_vector = np.zeros(10)
    target_vector[target] = 1.

    # callback for saving weights with the smallest loss
    checkpoint = ModelCheckpoint('./adversarial_weights.h5', monitor='loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', save_freq=1)
    # train adversarial image
    adversarial_model.fit(x={'image': img, 'unity': np.ones(shape=(1, 1))}, y=target_vector.reshape(1, -1), epochs=10000, verbose=0, callbacks=[checkpoint])
    # restore best weights
    adversarial_model.load_weights('./adversarial_weights.h5')

    # quantize adversarial noise
    quantized_weights = np.round(adversarial_model.get_weights()[0].reshape((28, 28)) * 255.) / 255.

    # add trained weights to original image and clip values to produce adversarial image
    adversarial_img = np.clip(img.reshape((28, 28)) + quantized_weights, 0., 1.)

    return adversarial_img, quantized_weights


def show_sub_figs(title: str, images: List[Tuple[np.array, np.array]]):
    fig = plt.figure()
    plt.title(title)
    plt.axis('off')
    for i, img in enumerate(images):
        fig.add_subplot(2, len(images), i + 1)
        plt.imshow(img[0])
        plt.axis('off')
        fig.add_subplot(2, len(images), i + 4)
        plt.imshow(img[1])
        plt.axis('off')
    plt.show()


def load_mnist() -> Tuple[np.array, np.array, np.array, np.array]:
    # load MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # preprocess data
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255.
    x_test /= 255.

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    return x_train, y_train, x_test, y_test


def train_model(save_model_dir: str, x_train: np.array, y_train: np.array, x_test: np.array, y_test: np.array):
    # compile CNN network for MNIST classification
    inputs = Input(shape=(28, 28, 1))
    net = Conv2D(32, kernel_size=(3, 3), activation='relu')(inputs)
    net = Conv2D(64, kernel_size=(3, 3), activation='relu')(net)
    net = MaxPooling2D(pool_size=(2, 2))(net)
    net = Dropout(0.25)(net)
    net = Flatten()(net)
    net = Dense(128, activation='relu')(net)
    net = Dropout(0.5)(net)
    outputs = Dense(10, activation='softmax')(net)

    mnist_model = Model(inputs=inputs, outputs=outputs, name='classification_model')
    mnist_model.compile(optimizer='nadam', loss='categorical_crossentropy', metrics=[categorical_accuracy])

    # train MNIST classifier
    early_stop = EarlyStopping(monitor='val_categorical_accuracy', min_delta=0, patience=10, verbose=0, mode='auto', baseline=None, restore_best_weights=True)

    mnist_model.fit(x_train, y_train, batch_size=128, epochs=100, verbose=1, validation_data=(x_test, y_test), callbacks=[early_stop])

    mnist_model.save(save_model_dir)

    print(mnist_model.evaluate(x_train, y_train))
    print(mnist_model.evaluate(x_test, y_test))


def _main(flags: argparse):
    x_train, y_train, x_test, y_test = load_mnist()

    # First, train and save the model
    if flags.training or not os.path.isdir(flags.save_model):
        train_model(flags.save_model, x_train, y_train, x_test, y_test)

    # Second, re-load the model back
    mnist_model = keras.models.load_model(flags.save_model)

    # select image to create an adversarial example from
    img = x_train[0:1]
    plt.imshow(img.reshape((28, 28)), vmin=0., vmax=1.)
    plt.show()
    # varify accurate classification
    prediction = mnist_model.predict(img)[0]
    print('Prediction: {}'.format(np.argmax(prediction)))

    # applying random noise does not fool the classifier
    quantized_noise = np.round(np.random.normal(loc=0.0, scale=0.3, size=img.shape) * 255.) / 255.
    noisy_img = np.clip(img + quantized_noise, 0., 1.)
    plt.imshow(noisy_img.reshape((28, 28)), vmin=0., vmax=1.)
    plt.show()
    noisy_prediction = mnist_model.predict(noisy_img)[0]
    target = np.argmax(noisy_prediction)
    print('Prediction: {}'.format(target))

    # Non-targeted misclassification image
    regularizations = [l1(0.01), l2(0.01), l1_l2(l1=0.01, l2=0.01)]
    non_targeted = 5
    generated_images = []
    for regularization in regularizations:
        generated_images.append(generate_adversary(flags.save_model, img, non_targeted, regularization, 'negative_categorical_crossentropy'))
        adversarial_prediction = mnist_model.predict(generated_images[-1][0].reshape((1, 28, 28, 1)))
        print('Expected {}, predicted: {}'.format(target, np.argmax(adversarial_prediction)))
    show_sub_figs("Non targeted", generated_images)

    # Targeted misclassification image
    targeted = 9
    generated_images = []
    for regularization in regularizations:
        generated_images.append(generate_adversary(flags.save_model, img, targeted, regularization, 'categorical_crossentropy'))
        adversarial_prediction = mnist_model.predict(generated_images[-1][0].reshape((1, 28, 28, 1)))
        print('Expected {}, predicted: {}'.format(target, np.argmax(adversarial_prediction)))
    show_sub_figs("Targeted", generated_images)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--training", help="Yes for training mode", type=bool, default=False)
    parser.add_argument("-m", "--save_model", help="Path where to save the model", type=str, default="mnist.model")
    args = parser.parse_args()

    _main(args)
