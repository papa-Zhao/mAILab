import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist


def plot_images(image_list):
    L = len(image_list)

    for i in range(L):
        plt.imshow(image_list[i], cmap=plt.cm.gray_r)
        plt.show()

(img_train, lbl_train), (img_test, lbl_test) = mnist.load_data()


def pooling(imgs, num_imgs, type_):
    pooled_images = []

    if type_ is "max":
        for i in range(num_imgs):
            ori_image = imgs[i]  # ori_image is the original image in imgs
            m, n = ori_image.shape
            pooled_image = np.zeros([m // 2, n // 2])

            for row in range(0, m, 2):
                for col in range(0, n, 2):
                    pooled_image[row // 2, col // 2] = np.amax(ori_image[row:row + 2, col:col + 2])
            pooled_images.append(pooled_image)
    elif type_ is "average":
        for i in range(num_imgs):
            ori_image = imgs[i]  # ori_image is the original image in imgs
            m, n = ori_image.shape
            pooled_image = np.zeros([m // 2, n // 2])

            for row in range(0, m, 2):
                for col in range(0, n, 2):
                    pooled_image[row // 2, col // 2] = np.mean(ori_image[row:row + 2, col:col + 2])
            pooled_images.append(pooled_image)
    else:
        raise ValueError("Argument type_ should be either max or average!")

    return pooled_images

plot_images(pooling(imgs=img_train, num_imgs=5, type_="max"))

plot_images(pooling(imgs=img_train, num_imgs=5, type_="average"))