import tensorflow as tf
import numpy as np

from tensorflow.keras.utils import save_img
import os


class RandomizeDataset:
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

    def shuffle_image(self, im):
        np.random.default_rng().shuffle(im)
        return im

    def return_shape(self, im):
        length = im.shape
        h = int(np.sqrt(length))
        ## won't work for imageNet
        image = tf.reshape(im, [h, h, 1])
        return image

    def save_image(self, image):
        save_img(self.save_path, image)


saliency_methods = ["gradient", "smoothGrad", "integratedGrad"]


for saliency_method in saliency_methods:

    image_dir = "../images/cifar10/{}/ResNets/Test_set/MiniResNetB".format(
        saliency_method
    )
    save_dir = "../images/cifar10/{}/ResNets/rand_Test_set/MiniResNetB".format(
        saliency_method
    )
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    files = os.listdir(image_dir)

    for file in files:
        current_path = image_dir + "/" + file
        current_save_path = save_dir + "/r_" + file

        randomize = RandomizeDataset(current_path, save_path=current_save_path)
        current_im = randomize.load_reshape()
        shuffled_im = randomize.shuffle_image(current_im)
        new_image = randomize.return_shape(shuffled_im)
        randomize.save_image(new_image)
