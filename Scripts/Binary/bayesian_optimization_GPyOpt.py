import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from result_metrics import calculate_metrics
from binary_model import train_binary_model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint

import keras_tuner as kt
from tensorflow import keras
from tensorflow.keras import layers

# Other imports
import GPyOpt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Read CSV
df_train = pd.read_csv("../../Datasets/flow-farm_train_smote.csv")
df_test = pd.read_csv("../../Datasets/flow-farm_test.csv")

# Transform the labels into binary values
df_train['is_attack'] = df_train['type'].apply(lambda x: 0 if x == "normal" else 1)
df_train.groupby('is_attack')['is_attack'].count()

df_test['is_attack'] = df_test['type'].apply(lambda x: 0 if x == "normal" else 1)
df_test.groupby('is_attack')['is_attack'].count()

# Delete Multi-class Column
df_train = df_train.drop('type', axis=1)
df_test = df_test.drop('type', axis=1)

# Create Encoder Training
X_columns = df_train.columns.drop('is_attack')
X_columns_test = df_test.columns.drop('is_attack')

le = LabelEncoder()
le.fit(df_train["is_attack"].values)
le.fit(df_test["is_attack"].values)

X = df_train[X_columns].values
X_test = df_test[X_columns_test].values

y = df_train["is_attack"].values
y_test = df_test["is_attack"].values

y = le.transform(y)
y_test = le.transform(y_test)

# Split into training and testing sets
X_train, X_train_test, y_train, y_train_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# Load MNIST dataset
X_train = X_train.reshape(-1, 158)
X_train_test = X_train_test.reshape(-1, 158)

# Define the DNN model and its hyperparameters
def build_model(hidden_layers, units, dropout_rate, learning_rate):
    model = Sequential()
    for i in range(hidden_layers):
        model.add(Dense(units, activation='relu'))
        model.add(Dropout(dropout_rate))
    model.add(Dense(10, activation='softmax'))
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Define the objective function to be minimized (validation loss)
def objective_function(hyperparameters):
    hidden_layers = int(hyperparameters[0])
    units = int(hyperparameters[1])
    dropout_rate = hyperparameters[2]
    learning_rate = hyperparameters[3]
    model = build_model(hidden_layers, units, dropout_rate, learning_rate)
    history = model.fit(X_train, y_train, validation_data=(X_train_test, y_train_test), epochs=5, batch_size=128, verbose=0)
    val_loss = np.min(history.history['val_loss'])
    return val_loss.item()

# Define the bounds and types of the hyperparameters to be optimized
bounds = [{'name': 'hidden_layers', 'type': 'discrete', 'domain': (1, 2, 3)},
          {'name': 'units', 'type': 'discrete', 'domain': (32, 64, 128, 256)},
          {'name': 'dropout_rate', 'type': 'continuous', 'domain': (0.0, 0.5)},
          {'name': 'learning_rate', 'type': 'continuous', 'domain': (1e-5, 1e-2)}]

# Initialize the Bayesian optimizer and run the optimization
optimizer = GPyOpt.methods.BayesianOptimization(f=objective_function, domain=bounds)
optimizer.run_optimization(max_iter=20)
print('Best hyperparameters:', optimizer.x_opt)
print('Best validation loss:', optimizer.fx_opt)

# Validate Model
# Load the best saved model
best_model = load_model('best_model_binary_smote.h5')

pred = best_model.predict(X_test)
pred = np.round(pred).astype(int)
calculate_metrics("Binary SMOTE - DNN", y_test, pred)
