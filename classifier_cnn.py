import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras import backend as K
from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier
from loadData import getData
import tensorflow as tf

def custom_loss(transition_matrix):
    def loss(y_true, y_pred):
        y_true = K.cast(y_true, 'float32')
        y_pred = K.cast(y_pred, 'float32')
        y_pred_adjusted = K.dot(y_pred, K.constant(transition_matrix, dtype='float32'))
        return K.categorical_crossentropy(y_true, y_pred_adjusted)
    return loss

def build_cnn(input_shape, num_classes, transition_matrix, filters=32, kernel_size=3, dense_neurons=128, dropout_rate=0.5):
    model = Sequential([
        Conv2D(filters, kernel_size=(kernel_size, kernel_size), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(dropout_rate),
        Flatten(),
        Dense(dense_neurons, activation='relu'),
        Dropout(dropout_rate),
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

def create_model(input_shape, num_classes, transition_matrix):
    def model_fn(filters, kernel_size, dense_neurons, dropout_rate):
        return build_cnn(input_shape, num_classes, transition_matrix, filters, kernel_size, dense_neurons, dropout_rate)
    return model_fn

def grid_search_cnn(xtr_val, ytr_val, input_shape, num_classes, transition_matrix):
    model_fn = create_model(input_shape, num_classes, transition_matrix)
    model = KerasClassifier(build_fn=model_fn, verbose=1)

    param_grid = {
        'filters': [32, 64],
        'kernel_size': [3],
        'dense_neurons': [128],
        'dropout_rate': [0.25, 0.5],
        'batch_size': [64],
        'epochs': [10]
    }

    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=2)
    grid_result = grid.fit(xtr_val, ytr_val)

    return grid_result.best_params_, grid_result.best_score_

def classifyCnn(filename, transition_matrix, count=1):
    xtr_val, ytr_val, xts, yts = getData(filename)
    num_classes = len(np.unique(ytr_val))

    xtr_val, ytr_val = preprocess_data(xtr_val, ytr_val, num_classes)
    xts, yts = preprocess_data(xts, yts, num_classes)

    # Perform Grid Search
    best_params, best_score = grid_search_cnn(xtr_val, ytr_val, xtr_val.shape[1:], num_classes, transition_matrix)
    print("Best Parameters:", best_params)
    print("Best Grid Search Score:", best_score)

    # Build and Evaluate the Model with the Best Parameters
    accuracies = []
    for _ in range(count):
        cnn = build_cnn(xtr_val.shape[1:], num_classes, transition_matrix, **best_params)
        cnn.fit(xtr_val, ytr_val, epochs=best_params['epochs'], batch_size=best_params['batch_size'], verbose=1)
        acc = evaluate_model(cnn, xts, yts)
        accuracies.append(acc)

    print("CNN Average Validation Accuracy: ", np.mean(accuracies))
