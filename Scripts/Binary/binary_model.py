from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint

def train_binary_model(x_train, y_train, x_test, y_test):
    # Define Model
    model = Sequential()
    model.add(Dense(256, input_dim=x_train.shape[1], activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid')) 

    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 

    # Define early stopping
    monitor = tf.keras.callbacks.ReduceLROnPlateau(monitor="loss",factor=0.5,mode="min",patience=10,verbose=1,min_lr=1e-7)
    checkpoint = ModelCheckpoint('best_model_binary_smote.h5', monitor='val_loss', save_best_only=True)

    # Train the model
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100, batch_size=512, callbacks=[monitor, checkpoint])
    print('[DONE] Training the model')
