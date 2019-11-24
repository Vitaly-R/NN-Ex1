import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from ConvNN import plot


class DigitSumModel(tf.keras.Model):
    def __init__(self):
        super(DigitSumModel, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, 5, activation='relu')
        self.maxpool1 = tf.keras.layers.MaxPool2D(2)
        self.conv2 = tf.keras.layers.Conv2D(64, 5, activation='relu')
        self.maxpool2 = tf.keras.layers.MaxPool2D(2)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(2048, activation='relu')
        self.dense2 = tf.keras.layers.Dense(20, activation='relu')
        self.dense3 = tf.keras.layers.Dense(19, activation='softmax')

    def __call__(self, x, **kwargs):
        res = self.conv1(x)
        res = self.maxpool1(res)
        res = self.conv2(res)
        res = self.maxpool2(res)
        res = self.flatten(res)
        res = self.dense1(res)
        res = self.dense2(res)
        return self.dense3(res)


def process_data(x, y):
    inds1 = np.arange(x.shape[0])
    inds2 = np.arange(x.shape[0])
    np.random.shuffle(inds1)
    np.random.shuffle(inds2)
    xres1 = x[inds1]
    yres1 = y[inds1]
    xres2 = x[inds2]
    yres2 = y[inds2]
    xres = np.hstack((xres1, xres2))
    xres = xres[..., np.newaxis]
    xres = xres / 255.0
    yres = yres1 + yres2
    return xres, yres


def get_data(batch_size):

    ((x_train_np, y_train), (x_test_np, y_test)) = tf.keras.datasets.mnist.load_data()
    xtrain, ytrain = process_data(x_train_np, y_train)
    xtest, ytest = process_data(x_test_np, y_test)
    train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain)).shuffle(10000).batch(batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices((xtest, ytest)).batch(batch_size)
    return train_ds, test_ds


def train_model(epochs=10, batch_size=30):
    train_ds, test_ds = get_data(batch_size)

    model = DigitSumModel()

    loss_function = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam()

    training_loss = tf.keras.metrics.Mean(name='training_loss')
    training_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='training_accuracy')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    @tf.function
    def train_step(imgs, lbls):
        with tf.GradientTape() as tape:
            predictions = model(imgs)
            loss = loss_function(lbls, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        training_loss(loss)
        training_accuracy(lbls, predictions)

    @tf.function
    def test_step(imgs, lbls):
        predictions = model(imgs)
        t_loss = loss_function(lbls, predictions)
        test_loss(t_loss)
        test_accuracy(lbls, predictions)

    i = 1
    x_axis = list()
    train_losses = list()
    train_accuracies = list()
    test_losses = list()
    test_accuracies = list()
    for epoch in range(1, epochs + 1):
        for x_batch, y_batch in train_ds:
            train_step(x_batch, y_batch)
            if not i % 500 or i == 1:
                print('round', i)
                for test_images, test_labels in test_ds:
                    test_step(test_images, test_labels)
                x_axis.append(i)
                train_losses.append(training_loss.result())
                train_accuracies.append(training_accuracy.result())
                test_losses.append(test_loss.result())
                test_accuracies.append(test_accuracy.result())
            i += 1

    return x_axis, train_losses, train_accuracies, test_losses, test_accuracies


def q_5():
    x, train_losses, train_accuracies, test_losses, test_accuracies = train_model()
    plot(x, train_losses, 'Training loss', '', '')
    plot(x, train_accuracies, 'Training accuracy', '', '')
    plot(x, test_losses, 'Test loss', '', '')
    plot(x, test_accuracies, 'Test accuracy', '', '')
    plt.show()



