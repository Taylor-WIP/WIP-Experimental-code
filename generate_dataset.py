import tensorflow as tf
import numpy as np
import saliency.core as saliency

from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import save_img
import os

import matplotlib.pyplot as plt
import tqdm


class GenerateSalDataset:
    def __init__(
        self, model, model_path, dataset, experiment_directory="other", batch_size=1, get_id=False
    ):
        self.model = model(dataset(batch_size))
        self.model_path = model_path
        self.dataset = dataset(batch_size, get_id=get_id)
        self.experiment_directory = experiment_directory
        self.get_id=get_id

    def load_model(self):
        model = self.model.build()
        model.load_weights(self.model_path)
        model.summary()
        return model

    def generate_dataset(self):

        (train_ds, test_ds) = self.dataset.get_and_process()

        model = self.load_model()
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        conv_layer = model.get_layer(index=-5)
        aug_model = tf.keras.models.Model(
            [model.inputs], [conv_layer.output, model.output]
        )
        class_idx_str = "class_idx_str"

        def call_model_function(images, call_model_args=None, expected_keys=None):
            target_class_idx = call_model_args[class_idx_str]
            images = tf.convert_to_tensor(images)
            with tf.GradientTape() as tape:
                if expected_keys == [saliency.base.INPUT_OUTPUT_GRADIENTS]:
                    tape.watch(images)
                    _, output_layer = aug_model(images)
                    output_layer = output_layer[:, target_class_idx]
                    gradients = np.array(tape.gradient(output_layer, images))
                    return {saliency.base.INPUT_OUTPUT_GRADIENTS: gradients}
                else:
                    conv_layer, output_layer = aug_model(images)
                    gradients = np.array(tape.gradient(output_layer, conv_layer))
                    return {
                        saliency.base.CONVOLUTION_LAYER_VALUES: conv_layer,
                        saliency.base.CONVOLUTION_OUTPUT_GRADIENTS: gradients,
                    }

        gradient_saliency = saliency.GradientSaliency()
        integrated_gradients = saliency.IntegratedGradients()
        grad_cam = saliency.GradCam()

        grad_directory_test = "../images/{}/gradient/{}/Test_set/{}".format(
            self.dataset.dataset_name, self.experiment_directory, self.model.name
        )
        if not os.path.exists(grad_directory_test):
            os.makedirs(grad_directory_test)

        grad_directory_train = "../images/{}/gradient/{}/Training/{}".format(
            self.dataset.dataset_name, self.experiment_directory, self.model.name
        )
        if not os.path.exists(grad_directory_train):
            os.makedirs(grad_directory_train)

        smooth_directory_test = "../images/{}/smoothGrad/{}/Test_set/{}".format(
            self.dataset.dataset_name, self.experiment_directory, self.model.name
        )
        if not os.path.exists(smooth_directory_test):
            os.makedirs(smooth_directory_test)

        smooth_directory_train = "../images/{}/smoothGrad/{}/Training/{}".format(
            self.dataset.dataset_name, self.experiment_directory, self.model.name
        )
        if not os.path.exists(smooth_directory_train):
            os.makedirs(smooth_directory_train)

        intGrad_directory_test = "../images/{}/integratedGrad/{}/Test_set/{}".format(
            self.dataset.dataset_name, self.experiment_directory, self.model.name
        )
        if not os.path.exists(intGrad_directory_test):
            os.makedirs(intGrad_directory_test)

        intGrad_directory_train = "../images/{}/integratedGrad/{}/Training/{}".format(
            self.dataset.dataset_name, self.experiment_directory, self.model.name
        )
        if not os.path.exists(intGrad_directory_train):
            os.makedirs(intGrad_directory_train)

        ####################   GRADCAM ######################
        # gradCam_directory_test = "../images/{}/gradCam/Test_set/{}".format(
        #     self.dataset.dataset_name, self.experiment_directory, self.model.name
        # )
        # if not os.path.exists(gradCam_directory_test):
        #     os.makedirs(gradCam_directory_test)
        #
        # gradCam_directory_train = "../images/{}/gradCam/Training/{}".format(
        #     self.dataset.dataset_name, self.experiment_directory, self.model.name
        # )
        # if not os.path.exists(gradCam_directory_train):
        #     os.makedirs(gradCam_directory_train)

        #########   GRADCAM ######################
        # gradCam_directory_test = "../images/test/gradCam/{}/Test_set/".formate(
        #     self.dataset_name
        # )
        # if not os.path.exists(gradCam_directory_test):
        #     os.makedirs(gradCam_directory_test)
        #
        # gradCam_directory_train = "../images/test/gradCam/{}/Training/".formate(
        #     self.dataset_name
        # )
        # if not os.path.exists(gradCam_directory_train):
        #     os.makedirs(gradCam_directory_train)

        count = 0
        for batch in tqdm.tqdm(test_ds):
            if self.get_id:
                batch = image_batch, labels_batch, ids_batch
                ids_batch = tf.squeeze(ids_batch, axis=0).numpy().decode("ascii")

            else:
                image_batch, labels_batch = batch


            image_batch = tf.squeeze(image_batch, axis=0)
            labels_batch = tf.squeeze(labels_batch, axis=0)

            # print(ids_batch)
            # exit()
            call_model_args = {class_idx_str: labels_batch}
            baseline = np.zeros(image_batch.shape)

            vanilla_mask_3d = gradient_saliency.GetMask(
                image_batch, call_model_function, call_model_args
            )
            smoothgrad_mask_3d = gradient_saliency.GetSmoothedMask(
                image_batch,
                call_model_function,
                call_model_args,
                stdev_spread=0.15,
                nsamples=50,
            )
            vanilla_integrated_gradients_mask_3d = integrated_gradients.GetMask(
                image_batch,
                call_model_function,
                call_model_args,
                x_steps=25,
                x_baseline=baseline,
                batch_size=20,
            )
            ######### GRADCAM #################
            # grad_cam_mask_3d = grad_cam.GetMask(
            #     image_batch, call_model_function, call_model_args
            # )

            gradMap = tf.expand_dims(
                saliency.VisualizeImageGrayscale(vanilla_mask_3d), axis=-1
            )
            smoothMap = tf.expand_dims(
                saliency.VisualizeImageGrayscale(smoothgrad_mask_3d), axis=-1
            )

            intGradMap = tf.expand_dims(
                saliency.VisualizeImageGrayscale(vanilla_integrated_gradients_mask_3d),
                axis=-1,
            )

            ######### GRADCAM #################
            # gradCamMap = tf.expand_dims(
            #     saliency.VisualizeImageGrayscale(grad_cam_mask_3d), axis=-1
            # )

            if self.get_id == False:
                ids_batch = count

            save_img(
                "{}/{}_{}.png".format(grad_directory_test, ids_batch, labels_batch),
                gradMap,
            )
            save_img(
                "{}/{}_{}.png".format(smooth_directory_test, ids_batch, labels_batch),
                smoothMap,
            )

            save_img(
                "{}/{}_{}.png".format(intGrad_directory_test, ids_batch, labels_batch),
                intGradMap,
            )

            ######### GRADCAM #################
            # save_img(
            #     "../images/test/gradCam/{}_image.png".format(ids_batch),
            #     image_batch,
            # )
            #
            # save_img(
            #     "{}/{}_{}.png".format(gradCam_directory_test, ids_batch, labels_batch),
            #     gradCamMap,
            # )

            count += 1
            # if count == 5:
            #     break

        count1 = 0
        for batch in tqdm.tqdm(train_ds):
            if self.get_id:
                batch = image_batch, labels_batch, ids_batch
                ids_batch = tf.squeeze(ids_batch, axis=0).numpy().decode("ascii")

            else:
                image_batch, labels_batch = batch

            image_batch = tf.squeeze(image_batch, axis=0)
            labels_batch = tf.squeeze(labels_batch, axis=0)


            call_model_args = {class_idx_str: labels_batch}
            baseline = np.zeros(image_batch.shape)

            vanilla_mask_3d = gradient_saliency.GetMask(
                image_batch, call_model_function, call_model_args
            )
            smoothgrad_mask_3d = gradient_saliency.GetSmoothedMask(
                image_batch, call_model_function, call_model_args
            )
            vanilla_integrated_gradients_mask_3d = integrated_gradients.GetMask(
                image_batch,
                call_model_function,
                call_model_args,
                x_steps=25,
                x_baseline=baseline,
                batch_size=20,
            )
            # map = normalize(vanilla_mask_3d)
            # smoothMap = normalize(smoothgrad_mask_3d)

            ######### GRADCAM #################
            # grad_cam_mask_3d = grad_cam.GetMask(
            #     image_batch, call_model_function, call_model_args
            # )

            gradMap = tf.expand_dims(
                saliency.VisualizeImageGrayscale(vanilla_mask_3d), axis=-1
            )
            smoothMap = tf.expand_dims(
                saliency.VisualizeImageGrayscale(smoothgrad_mask_3d), axis=-1
            )

            intGradMap = tf.expand_dims(
                saliency.VisualizeImageGrayscale(vanilla_integrated_gradients_mask_3d),
                axis=-1,
            )

            ######### GRADCAM #################
            # gradCamMap = tf.expand_dims(
            #     saliency.VisualizeImageGrayscale(grad_cam_mask_3d), axis=-1
            # )

            if self.get_id == False:
                ids_batch = count1

            save_img(
                "{}/{}_{}.png".format(grad_directory_train, ids_batch, labels_batch),
                gradMap,
            )

            save_img(
                "{}/{}_{}.png".format(smooth_directory_train, ids_batch, labels_batch),
                smoothMap,
            )

            save_img(
                "{}/{}_{}.png".format(intGrad_directory_train, ids_batch, labels_batch),
                intGradMap,
            )
            ######### GRADCAM #################
            # save_img(
            #     "{}/{}_{}.png".format(gradCam_directory_train, ids_batch, labels_batch),
            #     gradCamMap,
            # )

            count1 += 1
            # if count1 == 5:
            #     break

    ### FOR SAVING EXAMPLE MAPS ###
    def generate_examples(self):

        (test_ds) = self.dataset.get_and_process()

        model = self.load_model()
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        conv_layer = model.get_layer("conv5_block3_2_conv")
        aug_model = tf.keras.models.Model(
            [model.inputs], [conv_layer.output, model.output]
        )
        class_idx_str = "class_idx_str"

        def call_model_function(images, call_model_args=None, expected_keys=None):
            target_class_idx = call_model_args[class_idx_str]
            images = tf.convert_to_tensor(images)
            with tf.GradientTape() as tape:
                if expected_keys == [saliency.base.INPUT_OUTPUT_GRADIENTS]:
                    tape.watch(images)
                    _, output_layer = aug_model(images)
                    output_layer = output_layer[:, target_class_idx]
                    gradients = np.array(tape.gradient(output_layer, images))
                    return {saliency.base.INPUT_OUTPUT_GRADIENTS: gradients}
                else:
                    conv_layer, output_layer = aug_model(images)
                    gradients = np.array(tape.gradient(output_layer, conv_layer))
                    return {
                        saliency.base.CONVOLUTION_LAYER_VALUES: conv_layer,
                        saliency.base.CONVOLUTION_OUTPUT_GRADIENTS: gradients,
                    }

        gradient_saliency = saliency.GradientSaliency()
        integrated_gradients = saliency.IntegratedGradients()
        grad_cam = saliency.GradCam()

        count = 0
        for image_batch, labels_batch in tqdm.tqdm(test_ds):
            image_batch = tf.squeeze(image_batch, axis=0)
            labels_batch = tf.squeeze(labels_batch, axis=0)

            call_model_args = {class_idx_str: labels_batch}
            baseline = np.zeros(image_batch.shape)

            # vanilla_mask_3d = gradient_saliency.GetMask(
            #     image_batch, call_model_function, call_model_args
            # )
            # smoothgrad_mask_3d = gradient_saliency.GetSmoothedMask(
            #     image_batch,
            #     call_model_function,
            #     call_model_args,
            #     stdev_spread=0.15,
            #     nsamples=50,
            # )
            # vanilla_integrated_gradients_mask_3d = integrated_gradients.GetMask(
            #     image_batch,
            #     call_model_function,
            #     call_model_args,
            #     x_steps=25,
            #     x_baseline=baseline,
            #     batch_size=20,
            # )

            grad_cam_mask_3d = grad_cam.GetMask(
                image_batch, call_model_function, call_model_args
            )

            # gradMap = tf.expand_dims(
            #     saliency.VisualizeImageGrayscale(vanilla_mask_3d), axis=-1
            # )
            # smoothMap = tf.expand_dims(
            #     saliency.VisualizeImageGrayscale(smoothgrad_mask_3d), axis=-1
            # )
            #
            # intGradMap = tf.expand_dims(
            #     saliency.VisualizeImageGrayscale(vanilla_integrated_gradients_mask_3d),
            #     axis=-1,
            # )
            gradCamMap = tf.expand_dims(
                saliency.VisualizeImageGrayscale(grad_cam_mask_3d), axis=-1
            )

            save_img(
                "../images/test/gradCam/{}/{}_image.png".format(
                    self.dataset.dataset_name, count
                ),
                image_batch,
            )

            # save_img(
            #     "../images/{}/gradient/{}/{}_{}.png".format(
            #         self.dataset.dataset_name, self.model.name, count, labels_batch
            #     ),
            #     gradMap,
            # )
            #
            # save_img(
            #     "../images/{}/smoothGrad/{}/{}_{}.png".format(
            #         self.dataset.dataset_name, self.model.name, count, labels_batch
            #     ),
            #     smoothMap,
            # )
            #
            # save_img(
            #     "../images/{}/integratedGrad/{}/{}_{}.png".format(
            #         self.dataset.dataset_name, self.model.name, count, labels_batch
            #     ),
            #     intGradMap,
            # )
            save_img(
                "../images/test/gradCam/{}/{}_{}.png".format(
                    self.dataset.dataset_name, count, labels_batch
                ),
                gradCamMap,
            )

            # save_img(
            #     "../images/{}/examples/{}_{}_10_100smooth{}.png".format(
            #         self.dataset.dataset_name, count, self.model.name, labels_batch
            #     ),
            #     smoothMap,
            # )
            count += 1
            if count == 10:
                break
