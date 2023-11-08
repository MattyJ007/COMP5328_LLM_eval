from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize
import numpy as np
from loadData import getData

def classifyRf(filename, transition_matrix, count=2): #count should be 10 according to 2.1.1

    xtr_val, str_val, xts, yts = getData(filename)

    # Preprocessing
    num_classes = len(np.unique(str_val))
    xtr_val = xtr_val.reshape(xtr_val.shape[0], -1).astype('float32') / 255
    str_val_categorical = to_categorical(str_val, num_classes)
    
    # Define the parameter grid for hyperparameter tuning
    param_grid = {
        'n_estimators': [200],
        'max_depth': [10]
        #'n_estimators': [50, 100, 200],
        #'max_depth': [10, 20, None]
    }

    rf = RandomForestClassifier()

    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=count, scoring='accuracy')

    grid_search.fit(xtr_val, np.argmax(str_val_categorical, axis=1))

    best_rf = grid_search.best_estimator_
    best_score = grid_search.best_score_
    
    print("RF Best Validation Accuracy: ", best_score)
    print("Best Parameters: ", grid_search.best_params_)

    best_rf.fit(xtr_val, np.argmax(str_val_categorical, axis=1))
    
    xts = xts.reshape(xts.shape[0], -1).astype('float32') / 255
    yts_pred = best_rf.predict(xts)
    test_accuracy = accuracy_score(yts, yts_pred)
    print("RF Test Accuracy: ", test_accuracy)