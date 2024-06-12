# Import necessary libraries
"""
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from data_loader import load_data2, vectorized_result
import random

image_size = 28*28  # 8x8
random.seed(2137)
training_data, validation_data, test_data = load_data2(image_size)

x_train = [x[0] for x in training_data]
y_train = [x[1] for x in training_data]
x_valid = [x[0] for x in validation_data]
y_valid = [vectorized_result(x[1]) for x in validation_data]
x_test = [x[0] for x in test_data]
y_test = [vectorized_result(x[1]) for x in test_data]

x_train = np.array(x_train).reshape(54000, 784)
y_train = np.array(y_train).reshape(54000, 10)
x_valid = np.array(x_valid).reshape(6000, 784)
y_valid = np.array(y_valid).reshape(6000, 10)
x_test = np.array(x_test).reshape(10000, 784)
y_test = np.array(y_test).reshape(10000, 10)

# Create a simple sequential model
model = Sequential()
model.add(Dense(25, input_dim=image_size, activation='relu'))
model.add(Dense(10, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model using the fit method
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_valid, y_valid))

# Make predictions using the predict method
predictions = model.predict(x_test)

# Print some example predictions
for i in range(5):
    print(f"Actual: {y_test[i]}, Predicted: {predictions[i]}")
"""