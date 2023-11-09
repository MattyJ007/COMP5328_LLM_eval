import numpy as np
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras import backend as K
from keras.utils import to_categorical
from loadData import getData

def custom_loss(transition_matrix):
    def loss(y_true, y_pred):
        y_pred_adjusted = K.dot(y_pred, K.constant(transition_matrix, dtype='float32'))
        return K.categorical_crossentropy(y_true, y_pred_adjusted)
    return loss

def build_cnn(input_shape, num_classes, transition_matrix):
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(loss=custom_loss(transition_matrix), optimizer='adam', metrics=['accuracy'])
    return model

def preprocess_data(x, y, num_classes):
    x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1).astype('float32') / 255
    y = to_categorical(y, num_classes)
    return x, y

def evaluate_model(cnn, x_test, y_test):
    predictions = cnn.predict(x_test)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test, axis=1)
    return accuracy_score(true_classes, predicted_classes)

def classifyCnn(filename, transition_matrix, count=1):
    xtr_val, ytr_val, xts, yts = getData(filename)
    num_classes = len(np.unique(ytr_val))

    xtr_val, ytr_val = preprocess_data(xtr_val, ytr_val, num_classes)
    xts, yts = preprocess_data(xts, yts, num_classes)

    accuracies = []
    for _ in range(count):
        cnn = build_cnn(xtr_val.shape[1:], num_classes, transition_matrix)
        cnn.fit(xtr_val, ytr_val, epochs=10, batch_size=64, verbose=1)
        acc = evaluate_model(cnn, xts, yts)
        accuracies.append(acc)

    print("CNN Average Validation Accuracy: ", np.mean(accuracies))
