import time
import psutil
import tensorflow as tf
from tensorflow.keras import layers, models

def get_mnist_data():
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
    test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255
    return (train_images, train_labels), (test_images, test_labels)

def build_mnist_model():
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
  
class CPUMonitor(tf.keras.callbacks.Callback):
    """
    Create the CPU monitor callback (reports to Ray Train)
    """
    def __init__(self, ray=False):
        super().__init__()
        self.ray = ray

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
        self.cpu_percent = psutil.cpu_percent()
        if self.ray:
            train.report({"epoch": epoch, "cpu_begin": self.cpu_percent})  
        print(f"\nEpoch {epoch+1} starting CPU usage: {self.cpu_percent}%")

    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.epoch_start_time
        cpu_percent = psutil.cpu_percent()
        if self.ray:
            train.report({"epoch": epoch, "cpu_end": cpu_percent, "epoch_time": epoch_time})
        print(f"Epoch {epoch+1} ending CPU usage: {cpu_percent}%, Epoch time: {epoch_time:.2f}s")
