import tensorflow as tf
import numpy as np
from ConvNN import plot


class ConcatenatedDigitSumModel(tf.keras.Model):
    def __init__(self):
        super(ConcatenatedDigitSumModel, self).__init__()
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


class SeparateDigitSumModel(tf.keras.Model):
    def __init__(self):
        super(SeparateDigitSumModel, self).__init__()
        self.conv1_1 = tf.keras.layers.Conv2D(32, 5, activation='relu')
        self.conv1_2 = tf.keras.layers.Conv2D(32, 5, activation='relu')
        self.maxpool1_1 = tf.keras.layers.MaxPool2D(2)
        self.maxpool1_2 = tf.keras.layers.MaxPool2D(2)
        self.conv2_1 = tf.keras.layers.Conv2D(64, 5, activation='relu')
        self.conv2_2 = tf.keras.layers.Conv2D(64, 5, activation='relu')
        self.maxpool2_1 = tf.keras.layers.MaxPool2D(2)
        self.maxpool2_2 = tf.keras.layers.MaxPool2D(2)
        self.flatten_1 = tf.keras.layers.Flatten()
        self.flatten_2 = tf.keras.layers.Flatten()
        self.dense1_1 = tf.keras.layers.Dense(1024, activation='relu')
        self.dense1_2 = tf.keras.layers.Dense(1024, activation='relu')
        self.dense2 = tf.keras.layers.Dense(1024, activation='relu')
        self.dense3 = tf.keras.layers.Dense(19, activation='softmax')

    def __call__(self, x, **kwargs):
        res1 = self.conv1_1(x[:, 0, ...])
        res2 = self.conv1_2(x[:, 1, ...])
        res1 = self.maxpool1_1(res1)
        res2 = self.maxpool1_2(res2)
        res1 = self.conv2_1(res1)
        res2 = self.conv2_2(res2)
        res1 = self.maxpool2_1(res1)
        res2 = self.maxpool2_2(res2)
        res1 = self.flatten_1(res1)
        res2 = self.flatten_2(res2)
        res1 = self.dense1_1(res1)
        res2 = self.dense1_2(res2)
        res = tf.keras.layers.concatenate([res1, res2])
        res = self.dense2(res)
        return self.dense3(res)


def process_data_separate(x, y):
    inds = np.arange(x.shape[0])
    np.random.shuffle(inds)
    xres1 = x[inds]
    yres1 = y[inds]
    xres = np.array(list(zip(list(x), list(xres1))))
    xres = xres[..., np.newaxis]
    xres = xres / 255.0
    yres = y + yres1
    return xres, yres


def process_data_concatenated(x, y):
    inds = np.arange(x.shape[0])
    np.random.shuffle(inds)
    xres1 = x[inds]
    yres1 = y[inds]
    xres = np.hstack((x, xres1))
    xres = xres[..., np.newaxis]
    xres = xres / 255.0
    yres = y + yres1
    return xres, yres


def get_data(batch_size, concatenate=True):
    ((x_train_np, y_train), (x_test_np, y_test)) = tf.keras.datasets.mnist.load_data()
    xtrain, ytrain = process_data_concatenated(x_train_np, y_train) if concatenate else process_data_separate(x_train_np, y_train)
    xtest, ytest = process_data_concatenated(x_test_np, y_test) if concatenate else process_data_separate(x_test_np, y_test)
    train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain)).shuffle(10000).batch(batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices((xtest, ytest)).batch(batch_size)
    return train_ds, test_ds


def train_model(epochs=10, batch_size=30, concatenate=True):
    train_ds, test_ds = get_data(batch_size, concatenate)

    model = ConcatenatedDigitSumModel() if concatenate else SeparateDigitSumModel()

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

    i = 1
    x_axis = list()
    train_accuracies = list()
    test_accuracies = list()
    for epoch in range(1, epochs + 1):
        for x_batch, y_batch in train_ds:
            train_step(x_batch, y_batch)
            if not i % 500 or i == 1:
                for test_images, test_labels in test_ds:
                    test_step(test_images, test_labels)
                x_axis.append(i)
                train_accuracies.append(training_accuracy.result())
                test_accuracies.append(test_accuracy.result())
                print('round', i, 'training accuracy:', train_accuracies[-1], 'test accuracy:', test_accuracies[-1])
            i += 1

    return x_axis, train_accuracies, test_accuracies


def q_5_1():
    x, train_accuracies, test_accuracies = train_model()
    plot(x, train_accuracies, 'Question 5.1 \nDigits Sum Prediction as Concatenated Input \nTraining accuracy', 'round', 'accuracy')
    plot(x, test_accuracies, 'Question 5.1 \nDigits Sum Prediction as Concatenated Input \nTest accuracy', 'round', 'accuracy')


def q_5_2():
    x, train_accuracies, test_accuracies = train_model(concatenate=False)
    plot(x, train_accuracies, 'Question 5.2 \nDigits Sum Prediction as Separate Inputs \nTraining accuracy', 'round', 'accuracy')
    plot(x, test_accuracies, 'Question 5.2 \nDigits Sum Prediction as Separate Inputs \nTest accuracy', 'round', 'accuracy')
