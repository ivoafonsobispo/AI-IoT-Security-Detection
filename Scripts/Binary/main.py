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

#train_binary_model(X_train, y_train, X_train_test, y_train_test)

hp = kt.HyperParameters()

model = Sequential()

model.add(Dense(units = hp.Int('dense-bot', min_value=50, max_value=350, step=50), input_shape=(207509,), activation='relu'))

for i in range(hp.Int('num_dense_layers', 1, 2)):
  model.add(Dense(units=hp.Int('dense_' + str(i), min_value=50, max_value=100, step=25), activation='relu'))
  model.add(Dropout(hp.Choice('dropout_'+ str(i), values=[0.0, 0.1, 0.2, 0.3])))

model.add(Dense(10,activation="softmax"))

hp_optimizer=hp.Choice('Optimizer', values=['Adam', 'SGD'])

if hp_optimizer == 'Adam':
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-1, 1e-2, 1e-3])
elif hp_optimizer == 'SGD':
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-1, 1e-2, 1e-3])
    nesterov=True
    momentum=0.9

model.compile(optimizer = hp_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

tuner_mlp = kt.tuners.BayesianOptimization(
    model,
    seed=random_seed,
    objective='val_loss',
    max_trials=30,
    directory='.',
    project_name='tuning-mlp')
tuner_mlp.search(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_train_test, y_train_test), callbacks=callback)


best_mlp_hyperparameters = tuner_mlp.get_best_hyperparameters(1)[0]
print("Best Hyper-parameters")
best_mlp_hyperparameters.values

# Validate Model
# Load the best saved model
best_model = load_model('best_model_binary_smote.h5')

pred = best_model.predict(X_test)
pred = np.round(pred).astype(int)
calculate_metrics("Binary SMOTE - DNN", y_test, pred)