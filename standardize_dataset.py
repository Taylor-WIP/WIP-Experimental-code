import tensorflow as tf
import numpy as np

from tensorflow.keras.utils import save_img
import os


class StandardizeImage:
    def __init__(self, image_path, save_path):
        self.image_path = image_path
        self.save_path = save_path

    def load_image(self):
        return tf.io.decode_png(tf.io.read_file(self.image_path))

    def load_reshape(self):
        image = self.load_image()
        h, w, c = image.shape
        im = tf.reshape(image, [h * w * c]).numpy()
        return im

    def scale(self, im):
        im2 = tf.cast(im, tf.float32)
        xnorm = im2 / 255
        return xnorm

    def standardize_mean(self, im, new_mean=1):
        mean = np.mean(im)
        im2 = im - mean + new_mean
        return im2

    def standardize_std(self, im):
        std = np.std(im)
        im2 = im / std
        return im2

    def return_shape(self, im):
        length = im.shape
        h = int(np.sqrt(length))
        ## won't work for imageNet
        image = tf.reshape(im, [h, h, 1])
        return image

    def save_image(self, image):
        # save_img(self.save_path, image, scale=False )
        np.save(self.save_path, image)


# current_path = "../images/cifar10/res_examples/grad/0_MiniResNet.png"
# current_save_path= "../images/cifar10/res_examples/stand_0s_MiniResNet.png"
# std_save_path = "../images/cifar10/res_examples/std_0s_MiniResNet.png"
# z_save_path = "../images/cifar10/res_examples/z_0s_MiniResNet.png"
#
#
# stand = StandardizeMean(current_path, save_path=current_save_path)
# standStd = StandardizeMean(current_path, save_path=std_save_path)
# standZ = StandardizeMean(current_path, save_path=z_save_path)
#

# current_im = stand.load_reshape()
#
# normalized_im = stand.normalize(current_im)
#
# # print(normalized_im)
# # print(stand.standardize_mean(normalized_im))
#
# standardized_im = stand.standardize_mean(normalized_im)
# #
# # print(standardized_im)
# # print(normalized_im)
#
# new_image = stand.return_shape(standardized_im)
#
#
#
# std_im = stand.standardize_std(normalized_im)
# z_im = stand.standardize_std(standardized_im)
#
# std_image = stand.return_shape(std_im)
# z_image = stand.return_shape(z_im)
#
# # stand.save_image(new_image)
# standStd.save_image(std_image)
# standZ.save_image(z_image)

# print(normalized_im)
# print(standardized_im)
# print(z_im)
# print(std_im)

# print(normalized_im - standardized_im)
# print(normalized_im - std_im)
# print(normalized_im - z_im)
# print(std_im - z_im)
# print(std_im - standardized_im)


saliency_methods = ["gradient"]

# , "smoothGrad", "integratedGrad"
networks = ["MiniResNetB", "MiniResNet"]

sets = ["Training", "Test_set"]

for saliency_method in saliency_methods:
    for set in sets:
        for network in networks:

            image_dir = "../images/cifar10/{}/ResNets/{}/{}".format(
                saliency_method, set, network
            )
            save_dir = "../images/cifar10/{}/ResNets/z_standardized/{}/{}".format(
                saliency_method, set, network
            )
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            files = os.listdir(image_dir)

            for file in files:
                new_file = file.replace(".png", ".npy")
                current_path = image_dir + "/" + file
                current_save_path = save_dir + "/z_" + new_file
                #
                # randomize = RandomizeDataset(current_path, save_path=current_save_path)
                # current_im = randomize.load_reshape()
                # shuffled_im = randomize.shuffle_image(current_im)
                # new_image = randomize.return_shape(shuffled_im)
                # randomize.save_image(new_image)

                stand = StandardizeImage(current_path, save_path=current_save_path)

                current_im = stand.load_reshape()
                scaled_im = stand.scale(current_im)
                standardized_im = stand.standardize_mean(scaled_im)

                z_im = stand.standardize_std(standardized_im)

                new_image = stand.return_shape(z_im)

                stand.save_image(new_image)

                # std_im = stand.standardize_std(normalized_im)
                #
                #
                # std_image = stand.return_shape(std_im)
                # z_image = stand.return_shape(z_im)
                #
                # # stand.save_image(new_image)
                # standStd.save_image(std_image)
                # standZ.save_image(z_image)


# def normalize(x):
#     x = np.abs(x)
#     c = 1e-7
#     xnorm = (x - np.min(x)) / (np.max(x) - np.min(x) + c)
#     return xnorm
#
#
# def standadize_normalize(x):
#     x = normalize(x)
#     print(x)
#
#     mean = np.mean(x)
#     print(mean)
#
#     x -= mean - 1
#
#     print(x)
#     mean = np.mean(x)
#     print(mean)
#     # mean2 = np.mean(normx)
#     # print(mean2)
#     return x
#
#
# s = [1, 5, 6, 11, 3, 5, 22]
# ar = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
# # standadize_normalize(s)
# # print(np.mean(s))
# # normalize(s)
# # print(np.mean(s))
#
#
# def randomize_pixels(x):
#     print(x)
#     x = tf.reshape(x, [9]).numpy()
#     print(x)
#     # x = np.asarray(x)
#     # print(x.shape)
#     np.random.default_rng().shuffle(x)
#     print(x)
#     return x


# arr = np.arange(10)
# np.random.shuffle(arr)
# print(arr)
#

# randomize_pixels(ar)
