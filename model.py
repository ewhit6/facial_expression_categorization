import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import csv
from pathlib import Path
import numpy as np
from numpy import asarray
from PIL import Image
import os

train_labels = []
train_images = []

test_labels = []
test_images = []

path = Path(__file__).parent / "fer2013.csv"
# with path.open() as f:
# test = list(csv.reader(f))

with path.open() as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            if row[2] == 'Training':
                train_labels.append([int(row[0])])
                pixel_list = [int(n) for n in row[1].split()]
                # Do something to the pixels...
                image = Image.new('L', [48,48], 255)
                image.putdata(pixel_list)
                pixels = asarray(image)
                pixels = pixels.astype('float32')
                pixels /= 255.0
                train_images.append(pixels)
            else:
                test_labels.append([int(row[0])])
                pixel_list = [int(n) for n in row[1].split()]
                # Do something to the pixels...
                image = Image.new('L', [48,48], 255)
                image.putdata(pixel_list)
                pixels = asarray(image)
                pixels = pixels.astype('float32')
                pixels /= 255.0
                test_images.append(pixels)
            line_count += 1
    print(f'Processed {line_count} lines.')

# Normalize pixel values to be between 0 and 1
# train_images, test_images = train_images / 255.0, test_images / 255.0

train_labels = np.array(train_labels)
train_images = np.array(train_images).reshape(-1, 48, 48, 1)

test_labels = np.array(test_labels)
test_images = np.array(test_images).reshape(-1, 48, 48, 1)
# Verify the data
class_names = ['0', '1', '2', '3', '4', '5', '6']

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    # The CIFAR labels happen to be arrays,
    # which is why you need the extra index
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()

# Create the convolutional base
model = models.Sequential()
model.add(layers.Conv2D(48, (1, 1), activation='relu', input_shape=(48, 48, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(96, (1, 1), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(96, (1, 1), activation='relu'))

model.summary()

# Add dense layers on top
model.add(layers.Flatten())
model.add(layers.Dense(96, activation='relu'))
model.add(layers.Dense(10))

model.summary()

# compile and train the model
model.compile(optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10,
                validation_data=(test_images, test_labels))

# Evaluate the model
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print(test_acc)