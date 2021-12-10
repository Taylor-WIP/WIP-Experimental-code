import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os


class Analysis:
    def __init__(self):
        self.data = None
        self.mean = None

    def load_image(self, image_path):
        return tf.io.decode_png(tf.io.read_file(image_path))

    def load_reshape(self, image_path):
        image = self.load_image(image_path)
        h, w, c = image.shape
        im = tf.expand_dims(
            tf.reshape(self.load_image(image_path), [h * w * c]), axis=0
        )
        if self.data != None:
            self.data = tf.concat([self.data, im], axis=0)
        else:
            self.data = im

    def plot(self):
        batch, l = self.data.shape
        im = tf.reshape(self.data, [batch * l]).numpy()
        plt.hist(im, bins=256)
        plt.savefig("plots/Stand_Test_MiniResNet_hist.png")
        plt.clf()
        if self.mean == None:
            print("errori")
            exit()
        mean_im = self.mean.numpy()
        plt.bar(np.arange(0, len(mean_im)), mean_im, width=1)
        plt.savefig("plots/Stand_Test_MiniResNet_bar.png")

    def mean_std(self):
        self.mean = tf.math.reduce_mean(tf.cast(self.data, tf.float32), axis=0)


analysis = Analysis()

directory = "../images/cifar10/gradient/ResNets/z_standardized/Test_set/MiniResNet"
files = os.listdir(directory)
for file in files:
    current_path = directory + "/" + file
    analysis.load_reshape(current_path)
analysis.mean_std()
analysis.plot()


# print(files)

# analysis.load_reshape("../images/cifar10/examples/grad/5_MiniResNetB.png")
# analysis.load_reshape("../images/cifar10/examples/grad/4_MiniResNetB.png")

print((analysis.mean.numpy()[0:50]))

# print(analysis.mean)
# print(analysis.data)


#
# test_image = Analysis().plot_histogram("../images/cifar10/examples/grad/5_MiniResNetB.png")
#
#
# print(test_image)
