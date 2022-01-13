import tensorflow as tf
import numpy as np
import saliency.core as saliency

from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import save_img
import os

import matplotlib.pyplot as plt
import tqdm
import re


from datasets import Cifar10


class MultiplyMaps:
    def __init__(
        self, dataset, maps_directory, save_directory, networks, batch_size=1, get_id=True
    ):
        self.dataset = dataset(batch_size, get_id=get_id)
        self.maps_directory = maps_directory
        self.save_directory = save_directory
        self.get_id = get_id
        self.networks=networks


    def mulitiply_by_input(self):

        (train_ds, test_ds) = self.dataset.get_and_process()

        methods = ["gradient", "smoothGrad", "integratedGrad"]



        count = 0
        for batch in tqdm.tqdm(test_ds):
            image_batch, labels_batch, ids_batch = batch
            ids_batch = tf.squeeze(ids_batch, axis=0).numpy().decode("ascii")
            image_batch = tf.squeeze(image_batch, axis=0)
            labels_batch = tf.squeeze(labels_batch, axis=0)

            for network in self.networks:
                for method in methods:

                    map_dir = "../images/cifar10/{}/{}/Test_set/{}/{}_{}.png".format(method, self.maps_directory, network, ids_batch, labels_batch)
                    map =  tf.cast(tf.io.decode_png(tf.io.read_file(map_dir)), tf.float32)/255


                    multiplied_image= tf.multiply(image_batch, map)


                    # save_img("{}/Test_set/{}/{}_{}.png".format(self.save_directory, network, ids_batch, labels_batch),
                    # multiplied_image,
                    # )

                    save_location="../images/cifar10/{}/{}/Test_set/{}".format(method, self.save_directory, network)

                    save_img("{}/{}_{}.png".format(save_location, ids_batch, labels_batch),
                    multiplied_image,
                    )
                    #
                    # save_img("{}/{}.png".format(self.save_directory, ids_batch),
                    # image_batch,
                    # )


            # if count == 1:
            #     break
            #
            # count+=1




        count1 = 0
        for batch in tqdm.tqdm(train_ds):
            image_batch, labels_batch, ids_batch = batch
            ids_batch = tf.squeeze(ids_batch, axis=0).numpy().decode("ascii")
            image_batch = tf.squeeze(image_batch, axis=0)
            labels_batch = tf.squeeze(labels_batch, axis=0)

            for network in self.networks:
                for method in methods:

                    map_dir = "../images/cifar10/{}/{}/Training/{}/{}_{}.png".format(method, self.maps_directory, network, ids_batch, labels_batch)
                    map =  tf.cast(tf.io.decode_png(tf.io.read_file(map_dir)), tf.float32)/255


                    multiplied_image= tf.multiply(image_batch, map)
                    print(multiplied_image)

                    # save_img("{}/Training/{}/{}_{}.png".format(self.save_directory, network, ids_batch, labels_batch),
                    # multiplied_image,
                    # )

                    save_location="../images/cifar10/{}/{}/Training/{}".format(method, self.save_directory, network)

                    save_img("{}/{}_{}.png".format(save_location, ids_batch, labels_batch),
                    multiplied_image,
                    )


                    # save_img("{}/{}.png".format(self.save_directory, ids_batch),
                    # image_batch,
                    # )


            # if count1 == 1:
            #     break
            #
            # count1+=1


MultiplyMaps(dataset=Cifar10,
maps_directory = "ResNets/xInput/InitialMaps",
save_directory = "ResNets/xInput/MultipliedMaps", networks= ["MiniResNet", "MiniResNetB"]).mulitiply_by_input()
