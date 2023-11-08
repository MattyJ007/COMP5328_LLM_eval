from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
from loadData import getData

def classifyRf(filename, transition_matrix, count=1): #count should be 10 according to 2.1.1

    xtr_val, str_val, xts, yts = getData(filename)

    #Preprocessing
    num_classes = len(np.unique(str_val))
    xtr_val = xtr_val.reshape(xtr_val.shape[0], xtr_val.shape[1], xtr_val.shape[2], 1)
    xtr_val = xtr_val.astype('float32')
    xtr_val /= 255

    str_val_categorical = to_categorical(str_val, num_classes)

    rf_accuracies = []

    for _ in range(count):

        x_train, x_val, y_train, y_val = train_test_split(xtr_val, str_val_categorical, test_size=0.2, stratify=str_val)

        y_train_rf = np.argmax(y_train, axis=1)
        y_val_rf = np.argmax(y_val, axis=1)

        x_train_rf = x_train.reshape(x_train.shape[0], -1)
        x_val_rf = x_val.reshape(x_val.shape[0], -1)

        rf = RandomForestClassifier(n_estimators=100)
        rf.fit(x_train_rf, y_train_rf)
        rf_predictions = rf.predict(x_val_rf)
        rf_accuracies.append(accuracy_score(y_val_rf, rf_predictions))

    print("RF Average Validation Accuracy: ", sum(rf_accuracies)/count)
