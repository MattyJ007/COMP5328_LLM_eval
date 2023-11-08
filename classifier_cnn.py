from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
from loadData import getData

def build_cnn(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def classifyCnn(filename, transition_matrix, count=1): #count should be 10 according to 2.1.1

    xtr_val, str_val, xts, yts = getData(filename)

    #Preprocessing
    num_classes = len(np.unique(str_val))
    xtr_val = xtr_val.reshape(xtr_val.shape[0], xtr_val.shape[1], xtr_val.shape[2], 1)
    xtr_val = xtr_val.astype('float32')
    xtr_val /= 255

    str_val_categorical = to_categorical(str_val, num_classes)

    cnn_accuracies = []

    for _ in range(count):

        x_train, x_val, y_train, y_val = train_test_split(xtr_val, str_val_categorical, test_size=0.2, stratify=str_val)

        cnn = build_cnn(x_train.shape[1:], num_classes)
        cnn.fit(x_train, y_train, epochs=10, batch_size=64, verbose=0)
        cnn_predictions = cnn.predict(x_val)
        cnn_accuracies.append(accuracy_score(np.argmax(y_val, axis=1), np.argmax(cnn_predictions, axis=1)))

    print("CNN Average Validation Accuracy: ", sum(cnn_accuracies)/count)
