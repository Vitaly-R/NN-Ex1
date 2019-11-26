from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class CNNModel(tf.keras.Model):

    def __init__(self, conv1out=32, conv1window=5, conv1dropout=0.0, maxpool1dropout=0.0,
                 conv2out=64, conv2window=5, conv2dropout=0.0, maxpool2dropout=0.0,
                 dense1out=1024, dense1dropout=0.0, hidden_activation: object='relu'):
        super(CNNModel, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(conv1out, conv1window, activation=hidden_activation)
        self.dropout1 = tf.keras.layers.Dropout(conv1dropout)
        self.maxpool1 = tf.keras.layers.MaxPool2D(2)
        self.dropout2 = tf.keras.layers.Dropout(maxpool1dropout)
        self.conv2 = tf.keras.layers.Conv2D(conv2out, conv2window, activation=hidden_activation)
        self.dropout3 = tf.keras.layers.Dropout(conv2dropout)
        self.maxpool2 = tf.keras.layers.MaxPool2D(2)
        self.dropout4 = tf.keras.layers.Dropout(maxpool2dropout)
        self.flatten = tf.keras.layers.Flatten()
        self.d1 = tf.keras.layers.Dense(dense1out, activation=hidden_activation)
        self.dropout5 = tf.keras.layers.Dropout(dense1dropout)
        self.d2 = tf.keras.layers.Dense(10, activation='softmax')

    def __call__(self, x, **kwargs):
        res = self.conv1(x)
        res = self.maxpool1(res)
        res = self.conv2(res)
        res = self.maxpool2(res)
        res = self.flatten(res)
        res = self.d1(res)
        return self.d2(res)


class ShallowCNN(tf.keras.Model):

    def __init__(self):
        super(ShallowCNN, self).__init__()
        self.conv = tf.keras.layers.Conv2D(8, 3, activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(10, activation='softmax')

    def __call__(self, x, **kwargs):
        res = self.conv(x)
        res = self.flatten(res)
        return self.dense(res)


def pre_process_data(x_train: np.ndarray, x_test: np.ndarray):
    x_train_p = x_train / 255.0
    x_train_p = x_train_p[..., np.newaxis]
    x_test_p = x_test / 255.0
    x_test_p = x_test_p[..., np.newaxis]
    return x_train_p, x_test_p


def get_training_batches(x, y, batch_size, num_examples=-1):
    if num_examples == -1:
        return tf.data.Dataset.from_tensor_slices((x, y)).shuffle(10000).batch(batch_size)
    else:
        inds = np.arange(x.shape[0])
        selected = inds[np.random.randint(0, inds.shape[0], num_examples)]
        return tf.data.Dataset.from_tensor_slices((x[selected], y[selected])).shuffle(10000).batch(batch_size)


def train_model(model=CNNModel(), num_training_examples=60000):
    train_batch = 32
    test_batch = 32
    iterations = 20000

    ((x_train_np, y_train), (x_test_np, y_test)) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = pre_process_data(x_train_np, x_test_np)

    train_ds = get_training_batches(x_train, y_train, train_batch, num_training_examples)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(test_batch)

    loss_function = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam()

    training_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='training_accuracy')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    @tf.function
    def train_step(imgs, lbls):
        with tf.GradientTape() as tape:
            predictions = model(imgs)
            loss = loss_function(lbls, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        training_accuracy(lbls, predictions)

    @tf.function
    def test_step(imgs, lbls):
        predictions = model(imgs)
        test_accuracy(lbls, predictions)

    x_axis = list()
    train_accuracies = list()
    test_accuracies = list()
    i = 1

    while i <= iterations:
        for x_batch, y_batch in train_ds:
            train_step(x_batch, y_batch)
            if not i % 500 or i == 1:
                print('round', i)
                for test_images, test_labels in test_ds:
                    test_step(test_images, test_labels)
                x_axis.append(i)
                train_accuracies.append(training_accuracy.result())
                test_accuracies.append(test_accuracy.result())
            i += 1
            if i > iterations:
                break

    return x_axis, train_accuracies, test_accuracies


def plot(x, y, title='', xlabel='', ylabel=''):
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(x, y)


def q_1_2():
    model = CNNModel()
    x, train_acc, test_acc = train_model(model)
    plot(x, train_acc, 'Training Accuracy', 'round', 'accuracy')
    plot(x, test_acc, 'Test Accuracy', 'round', 'accuracy')


def q_2_2():
    model = CNNModel(hidden_activation=None)
    x, train_acc, test_acc = train_model(model)
    plot(x, train_acc, 'Training Accuracy', 'round', 'accuracy')
    plot(x, test_acc, 'Test Accuracy', 'round', 'accuracy')


def q_4_1():
    full = CNNModel()
    full_x, full_train_acc, full_test_acc = train_model(full)
    plot(full_x, full_train_acc, 'Original Network Training Accuracy', 'round', 'accuracy')
    plot(full_x, full_test_acc, 'Original Network Test Accuracy', 'round', 'accuracy')
    plot(full_x, [full_test_acc[i] / full_train_acc[i] for i in range(len(full_x))], 'Original Network Training to Test Accuracy Ratio', 'training round', 'accuracy')

    reduced = CNNModel(conv1dropout=0.2, maxpool1dropout=0.2, conv2dropout=0.2, maxpool2dropout=0.2, dense1dropout=0.2)
    reduced_x, reduced_train_acc, reduced_test_acc = train_model(reduced)
    plot(reduced_x, reduced_train_acc, 'Reduced Network Training Accuracy', 'round', 'accuracy')
    plot(reduced_x, reduced_test_acc, 'Reduced Network Test Accuracy', 'round', 'accuracy')
    plot(reduced_x, [reduced_test_acc[i] / reduced_train_acc[i] for i in range(len(reduced_x))], 'Reduced Network Test to Training Accuracy Ratio', 'training round', 'accuracy')
