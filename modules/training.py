import network

import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.layers import layers
from tensorflow.keras.models import sequential
from tensorflow.keras.optimizers import adam
from sklearn.preprocessing import MinMaxScaler

# training_data = []
# training_data = np.array(training_data)

# Loading JSON into dataframe
dataframe = pd.read_json(r'')
dataframe.head()

# Creating validation and training dataframe
val_dataframe = dataframe.sample(frac=0.2, random_state=1337)
train_dataframe = dataframe.drop(val_dataframe.index)

# Batching the datasets

train_ds = train_ds.batch(32)
val_ds = val_ds.batch(32)

scaler = MinMaxScaler(feature_range=(0, 1))
training_data = scaler.fit_transform(training_data.reshape(-1, 1))

model = Sequential([
    Dense(units=16, input_shape=(1,), activation='relu'),
    Dense(units=32, activation='relu'),
    Dense(units=2, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x=scaled_train_samples, y=train_labels, batch_size=10, epochs=30, shuffle=True, verbose=2)

