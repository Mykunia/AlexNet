import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, datasets
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Load and preprocess dataset (CIFAR-10)
def load_data():
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
    
    # Normalize pixel values to range [0,1]
    x_train, x_test = x_train / 255.0, x_test / 255.0
    
    # Convert labels to categorical (one-hot encoding)
    y_train, y_test = tf.keras.utils.to_categorical(y_train, 10), tf.keras.utils.to_categorical(y_test, 10)
    
    return (x_train, y_train), (x_test, y_test)

# Adjusted AlexNet Model for 32x32 images
def build_alexnet(input_shape=(32, 32, 3), num_classes=10):
    model = models.Sequential([
        # First Convolutional Layer (5x5 instead of 11x11)
        layers.Conv2D(96, (5, 5), strides=1, activation='relu', input_shape=input_shape, padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((3, 3), strides=2, padding="same"),  # Prevent excessive shrinking
        
        # Second Convolutional Layer
        layers.Conv2D(256, (5, 5), activation='relu', padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((3, 3), strides=2, padding="same"),
        
        # Third Convolutional Layer
        layers.Conv2D(384, (3, 3), activation='relu', padding="same"),
        
        # Fourth Convolutional Layer
        layers.Conv2D(384, (3, 3), activation='relu', padding="same"),
        
        # Fifth Convolutional Layer
        layers.Conv2D(256, (3, 3), activation='relu', padding="same"),
        layers.MaxPooling2D((3, 3), strides=2, padding="same"),  # Ensure valid feature map size
        
        # Flatten Layer
        layers.Flatten(),
        
        # Fully Connected Layers
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

# Compile and train the model
def train_alexnet(model, x_train, y_train, x_test, y_test, epochs=10, batch_size=128):
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),  # Adam optimizer
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history = model.fit(
        x_train, y_train, 
        epochs=epochs, 
        batch_size=batch_size,
        validation_data=(x_test, y_test),
        verbose=1
    )
    
    return history

# Plot training history
def plot_history(history):
    plt.figure(figsize=(12, 5))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

# Confusion Matrix
def plot_confusion_matrix(model, x_test, y_test):
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    cm = confusion_matrix(y_true, y_pred_classes)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()
    
    # Print classification report
    print("Classification Report:\n", classification_report(y_true, y_pred_classes))

# Run everything
(x_train, y_train), (x_test, y_test) = load_data()
alexnet_model = build_alexnet()
history = train_alexnet(alexnet_model, x_train, y_train, x_test, y_test, epochs=20, batch_size=128)

# Plot learning history
plot_history(history)

# Evaluate model and plot confusion matrix
plot_confusion_matrix(alexnet_model, x_test, y_test)
 