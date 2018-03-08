import numpy as np
import matplotlib.pyplot as plt

images_num = 5
train_images_file = open("train-images-idx3-ubyte", mode='rb')
train_images_file.seek(16)
five_images = np.fromfile(train_images_file,np.dtype((np.uint8, (28,28))),count=images_num)
#plot_images(five_images)
five_images = np.pad(five_images,((0,0),(1,1),(1,1)),mode='constant')

Gx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
Gy = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])

def plot_images(image_list):
    L = len(image_list)

    for i in range(L):
        plt.subplot(1, L, i + 1)
        plt.imshow(image_list[i], cmap=plt.cm.gray)
        plt.axis('off')
    plt.show()


def Conv_filter(five_images,filter,image_num,image_size,filter_size):
    convoluted_image = np.zeros([image_num, image_size, image_size])
    for i in range(convoluted_image.shape[0]):
        for j in range(convoluted_image.shape[1]):
            for k in range(convoluted_image.shape[2]):
                convoluted_image[i, j, k] = np.sum(np.multiply(five_images[i, j:j + filter_size, k:k + filter_size], filter),
                                                      axis=None)  # element-wise product np.multiply()
    return convoluted_image

convoluted_image_Gx = Conv_filter(five_images,Gx,5,28,3)
#plot_images(convoluted_image_Gx)
convoluted_image_Gy = Conv_filter(five_images,Gy,5,28,3)
#plot_images(convoluted_image_Gy)


