# Noise Classifier

## Install

- Anaconda
- Python3
- SK Learn
- Keras

### Libraries

- numpy
- matplotlib
- pandas

## Run

Transition Matrices are constants.
Default experiment no. 10

> Update CIFAR T Matrix constant after estimation then run CIFAR classification

### Help

```sh
python main.py help
```

### CIFAR Estimator

```sh
python main.py estimate
```

### Noise Classifier

#### Fashion 0.5

```sh
python main.py fashion5
```

#### Fashion 0.6

> Pick the method with the second flag: rf or cnn

```sh
python main.py fashion6 <method>
```

#### CIFAR

```sh
python main.py cifar <method>
```

#### All (Fashion5, Fashion6, CIFAR)

```sh
python main.py all <method>
```

### Visualise Data

```sh
python visualise.py
```
