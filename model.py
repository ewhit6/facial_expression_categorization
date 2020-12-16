import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.layers import BatchNormalization, Dropout
import matplotlib.pyplot as plt
import csv
from pathlib import Path
import numpy as np
from numpy import asarray
from PIL import Image
import os


def parse_csv_to_nparrays():
    train_labels = []
    train_images = []

    test_labels = []
    test_images = []

    path = Path(__file__).parent / "fer2013new.csv"

    print("Processing Dataset...")

    with path.open() as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
                line_count += 1
            else:
                label_votes = row[2:12]
                majority_vote = max(label_votes)
                max_label = label_votes.index(majority_vote)
                if row[0] == 'Training':
                    train_labels.append([int(max_label)])
                else:
                    row_labels = row[2:12]
                    test_labels.append([int(max_label)])
                line_count += 1

    path = Path(__file__).parent / "fer2013.csv"

    with path.open() as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                if row[2] == 'Training':
                    pixel_list = [int(n) for n in row[1].split()]
                    image = Image.new('L', [48,48], 255)
                    image.putdata(pixel_list)
                    pixels = asarray(image)
                    pixels = pixels.astype('float32')
                    # Normalize pixel values to be between 0 and 1
                    pixels /= 255.0
                    train_images.append(pixels)
                else:
                    # test_labels.append([int(row[0])])
                    pixel_list = [int(n) for n in row[1].split()]
                    image = Image.new('L', [48,48], 255)
                    image.putdata(pixel_list)
                    pixels = asarray(image)
                    pixels = pixels.astype('float32')
                    # Normalize pixel values to be between 0 and 1
                    pixels /= 255.0
                    test_images.append(pixels)
                line_count += 1
        print(f'Processed {line_count} lines.')

    train_labels = np.array(train_labels)
    train_images = np.array(train_images).reshape(-1, 48, 48, 1)

    test_labels = np.array(test_labels)
    test_images = np.array(test_images).reshape(-1, 48, 48, 1)

    return [train_labels, train_images, test_labels, test_images]

def create_model():
    model = models.Sequential()
 # kernel_constraint=max_norm(3)

    # Create the convolutional base
    model.add(layers.Conv2D(64, (3, 3), padding='same', input_shape=(48, 48, 1)))
    model.add(BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(layers.Conv2D(128, (5, 5), padding='same'))
    model.add(BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(layers.Conv2D(512, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(layers.Conv2D(512, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(layers.Flatten())

    # Add dense layers on top
    model.add(layers.Dense(256))
    model.add(BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(Dropout(0.25))

    model.add(layers.Dense(512))
    model.add(BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(Dropout(0.25))

    model.add(layers.Dense(10, activation='softmax'))    

    # compile model
    model.compile(optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])
    return model