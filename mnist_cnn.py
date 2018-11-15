import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Dropout, BatchNormalization, GlobalAvgPool2D
from keras.optimizers import Adam
import keras.backend as K
import numpy as np

def build_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(64, (3, 3), strides=(2,2), activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), strides=(2,2), activation='relu'))
    model.add(BatchNormalization())
    model.add(GlobalAvgPool2D())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model


def build_functions(input_shape, num_classes, model):

    images = K.placeholder((None,) + input_shape) # shape = (batch_size, img_height,  img_width, channels)
    y_true = K.placeholder((None, num_classes))   # shape = (batch_size, num_classes)
    y_pred = model(images)

    loss = K.mean(K.categorical_crossentropy(y_true, y_pred))
    acc  = K.mean(K.cast(K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)), K.floatx()))

    # get updates of untrainable updates. 
    # e.g. mean and variance in BatchNormalization
    untrainable_updates = model.get_updates_for([images])

    # get updates of trainable updates.
    trainable_updates = Adam(lr=0.0001).get_updates(loss, model.trainable_weights)

    # K.learning_phase() is required if model has different behavior during train and test. 
    # e.g. BatchNormalization, Dropout
    train_func = K.function([images, y_true, K.learning_phase()], [loss, acc], untrainable_updates + trainable_updates)
    test_func  = K.function([images, y_true, K.learning_phase()], [loss, acc])

    return train_func, test_func


train_batch_size = 64
test_batch_size = 8
num_classes = 10
epochs = 10
image_shape = (28, 28, 1)


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

x_train = x_train.reshape((-1,) + image_shape)
x_test = x_test.reshape((-1,) + image_shape)

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test  = keras.utils.to_categorical(y_test, num_classes)

num_of_train_data = x_train.shape[0]
num_of_test_data = x_test.shape[0]


model = build_model(image_shape, num_classes)
train_func, test_func = build_functions(image_shape, num_classes, model)


for e in range(epochs):

    train_step_loss = []
    train_step_acc = []
    test_step_loss = []
    test_step_acc = []

    # random shuffle
    rand_idx = np.random.permutation(num_of_train_data)
    x_train = x_train[rand_idx]
    y_train = y_train[rand_idx]

    # ========== Training steps ==========
    for i in range(0, num_of_train_data, train_batch_size):
        batch_images = x_train[i:min(i + train_batch_size, num_of_train_data)]
        batch_labels = y_train[i:min(i + train_batch_size, num_of_train_data)]

        loss, acc = train_func([batch_images, batch_labels, 1])
        train_step_loss.append(loss)
        train_step_acc.append(acc)

        print ("\rEpoch:[{0}/{1}], Steps:[{2}/{3}] loss: {4:.4}, acc: {5:.4}".format(
            e+1, epochs, i+1, num_of_train_data, np.mean(train_step_loss), np.mean(train_step_acc)
        ), end='')

    # ========== Testing steps ==========
    for i in range(0, num_of_test_data, test_batch_size):

        batch_images = x_test[i:min(i + test_batch_size, num_of_test_data)]
        batch_labels = y_test[i:min(i + test_batch_size, num_of_test_data)]

        loss, acc = test_func([batch_images, batch_labels, 0])
        test_step_loss.append(loss)
        test_step_acc.append(acc)


    print ("\rEpoch:[{0}/{1}] train_loss: {2:.4}, train_acc: {3:.4}, test_loss: {4:.4}, test_acc: {5:.4}".format(
        e+1, epochs, np.mean(train_step_loss), np.mean(train_step_acc), np.mean(test_step_loss), np.mean(test_step_acc)
    ))
