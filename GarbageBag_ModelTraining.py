# Import necessary modules from TensorFlow and Keras
from tensorflow.keras import layers, models
import tensorflow_addons as tfa
import numpy as np

# Define a convolutional neural network (CNN) model for binary classification
cnn = models.Sequential([
    # First convolutional layer with 8 filters, 3x3 kernel, ReLU activation, and same padding
    layers.Conv2D(filters=8, kernel_size=(3,3), activation='relu', strides=(1, 1), padding='same', input_shape=(32,32,3)),
    layers.MaxPooling2D(2,2),  # Max-pooling layer with a 2x2 filter to reduce spatial dimensions

    layers.Dropout(0.12),  # Dropout layer for regularization (12% dropout rate)
    layers.BatchNormalization(),  # Batch normalization to stabilize and accelerate training

    # Second convolutional layer with 10 filters, 3x3 kernel, ReLU activation, and stride of 2
    layers.Conv2D(filters=10, kernel_size=(3,3), activation='relu', strides=(2, 2), padding='same'),
    layers.MaxPooling2D(2,2),  # Max-pooling layer to further reduce dimensions

    layers.BatchNormalization(),  # Batch normalization layer

    layers.Flatten(),  # Flatten layer to reshape the data for fully connected layers

    layers.Dense(70, activation='relu'),  # Fully connected layer with 70 units and ReLU activation
    layers.Dense(50, activation='relu'),  # Fully connected layer with 50 units and ReLU activation
    layers.Dropout(0.14),  # Dropout layer with a dropout rate of 14%

    layers.Dense(1, activation='sigmoid')  # Output layer for binary classification with sigmoid activation
])

# Learning rate settings for cyclical learning rate
INIT_LR = 4.43e-05  # Initial learning rate
MAX_LR = 1.85e-04  # Maximum learning rate

# Define batch size and calculate steps per epoch
BATCH_SIZE = 8
steps_per_epoch = len(x_train) // BATCH_SIZE  # Calculate number of steps per epoch based on batch size

# Define cyclical learning rate using TensorFlow Addons
clr = tfa.optimizers.CyclicalLearningRate(
    initial_learning_rate=INIT_LR,
    maximal_learning_rate=MAX_LR,
    scale_fn=lambda x: 1 / (2. ** (x - 1)),  # Scaling function for cyclic learning rate
    step_size=2 * steps_per_epoch
)

# Define loss function and optimizer
loss = keras.losses.BinaryCrossentropy(from_logits=False)  # Binary cross-entropy loss for binary classification
opt = keras.optimizers.Adam(learning_rate=clr)  # Adam optimizer with cyclical learning rate

# Compile the CNN model with optimizer, loss, and binary accuracy as a metric
cnn.compile(optimizer=opt, loss=loss, metrics=['binary_accuracy'])

# Train the model on training data with validation
history = cnn.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=BATCH_SIZE, epochs=50)
