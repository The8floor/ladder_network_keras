from __future__ import print_function

from keras.datasets import mnist
import keras
import random
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from ladder_net import get_ladder_network_fc


x_train = pd.read_csv("./train_encode.csv", header=None).head(600000).values
y_train = pd.read_csv("./train_label.csv", header=None).head(600000).values

x_test = pd.read_csv("./test_encode.csv", header=None).values
y_test = pd.read_csv("./test_label.csv", header=None).values

# get the dataset
inp_size = 768 # size of mnist dataset 
n_classes = 5

y_train = keras.utils.to_categorical(y_train, n_classes)
y_test  = keras.utils.to_categorical(y_test,  n_classes)

# only select 100 training samples 
idxs_annot = range(x_train.shape[0])
random.seed(0)
idxs_annot = np.random.choice(x_train.shape[0], 1000)

x_train_unlabeled = x_train
x_train_labeled   = x_train[idxs_annot]
y_train_labeled   = y_train[idxs_annot]

n_rep = x_train_unlabeled.shape[0] // x_train_labeled.shape[0]
x_train_labeled_rep = np.concatenate([x_train_labeled]*n_rep)
y_train_labeled_rep = np.concatenate([y_train_labeled]*n_rep)

# initialize the model 
model = get_ladder_network_fc(layer_sizes=[inp_size, 1000, 500, 250, 250, 250, n_classes])

# train the model for 100 epochs
for _ in range(10000):
    model.fit([x_train_labeled_rep, x_train_unlabeled], y_train_labeled_rep, epochs=1)
    y_test_pr = model.test_model.predict(x_test, batch_size=100)
    print("Test accuracy : %f" % accuracy_score(y_test.argmax(-1), y_test_pr.argmax(-1)))
