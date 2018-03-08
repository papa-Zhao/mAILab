#!/usr/bin/env python3

# HW0003 MNIST and LeNet

import numpy as np

# Basic

# 2. print the first image of the train-images-idx3-ubyte, 28 * 28 bytes for
# each images

def read_image(file, start=0, count=1):
# ignore 16 bytes metadata and some 28 * 28 images
    file.seek(16 + start * 28 * 28)
    dt = np.dtype((np.uint8, (28,28)))
    return np.fromfile(file, dtype=dt, count=count)

def print_image(image):
    for i in image:
        for j in i:
            print("{:02X}".format(j), end=' ')
        print()
    print()

train_images_file = open("datasets/train-images-idx3-ubyte", mode='rb')

first_image = read_image(train_images_file, 0, 1) # read the first image from file
first_image = first_image[0]
print("First image")
print_image(first_image)

# 3. print the average of the first ten images with unconditional rounding
ten_images = read_image(train_images_file, 0, 10)
average_image = np.sum(ten_images, axis=0) // 10
print("Average image")
print_image(average_image)

# 4. print the average of the first ten labels rounding two decimal places

def read_label(file, start=0, count=1):
    file.seek(8 + start)
    dt = np.dtype(np.uint8)
    return np.fromfile(file, dtype=dt, count=count)

train_labels_file = open("datasets/train-labels-idx1-ubyte", mode='rb')

ten_labels = read_label(train_labels_file, 0, 10)
average = np.sum(ten_labels) / 10
print("Label Average")
print("{:.2f}".format(average))

# 5. print the first image with padding to 32 * 32
padding_image = np.pad(first_image, [2, 2], mode='edge')
print("Padding image")
print_image(padding_image)

# Advanced

# 6. From 2, save the first image as bmp file
# Need scipy and Pillow

from scipy.misc import imsave

imsave('image.bmp', first_image)