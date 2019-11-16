from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class CNNModel(tf.keras.Model):

    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, 5, activation='relu')
        self.maxpool1 = tf.keras.layers.MaxPool2D(2)
        self.conv2 = tf.keras.layers.Conv2D(64, 5, activation='relu')
        self.maxpool2 = tf.keras.layers.MaxPool2D(2)
        self.flatten = tf.keras.layers.Flatten()
        self.d1 = tf.keras.layers.Dense(1024, activation='relu')
        self.d2 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, x):
        res = self.conv1(x)
        res = self.maxpool1(res)
        res = self.conv2(res)
        res = self.maxpool2(res)
        res = self.flatten(res)
        res = self.d1(res)
        return self.d2(res)


def load_data():
    return tf.keras.datasets.mnist.load_data()


def pre_process_data(x_train: np.ndarray, x_test: np.ndarray):
    x_train_p = x_train / 255.0
    x_train_p = x_train_p[..., np.newaxis]
    x_test_p = x_test / 255.0
    x_test_p = x_test_p[..., np.newaxis]
    return x_train_p, x_test_p


def main():
    train_batch = 3
    test_batch = 32
    ((x_train_np, y_train), (x_test_np, y_test)) = load_data()
    x_train, x_test = pre_process_data(x_train_np, x_test_np)

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(train_batch)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(test_batch)
    model = CNNModel()
    loss_function = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam()
    training_loss = tf.keras.metrics.Mean(name='training_loss')
    training_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='training_accuracy')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    @tf.function
    def train_step(imgs, lbls):
        with tf.GradientTape() as tape:
            predictions = model.call(imgs)
            loss = loss_function(lbls, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        training_loss(loss)
        training_accuracy(lbls, predictions)

    @tf.function
    def test_step(imgs, lbls):
        predictions = model.call(imgs)
        t_loss = loss_function(lbls, predictions)
        test_loss(t_loss)
        test_accuracy(lbls, predictions)

    x_axis = list()
    train_losses = list()
    train_accuracies = list()
    test_losses = list()
    test_accuracies = list()
    i = 0
    for x_batch, y_batch in train_ds:
        train_step(x_batch, y_batch)
        if not i % 500 or i == ((x_train.shape[0] // train_batch) - 1):
            print('round', i)
            for test_images, test_labels in test_ds:
                test_step(test_images, test_labels)
            x_axis.append(i)
            train_losses.append(training_loss.result())
            train_accuracies.append(training_accuracy.result())
            test_losses.append(test_loss.result())
            test_accuracies.append(test_accuracy.result())
        i += 1

    plt.figure()
    plt.title('training loss values')
    plt.plot(x_axis, train_losses)

    plt.figure()
    plt.title('training accuracy values')
    plt.plot(x_axis, train_accuracies)

    plt.figure()
    plt.title('test loss values')
    plt.plot(x_axis, test_losses)

    plt.figure()
    plt.title('test accuracy values')
    plt.plot(x_axis, test_accuracies)

    plt.show()


if __name__ == '__main__':
    main()
