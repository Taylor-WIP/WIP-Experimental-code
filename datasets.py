import tensorflow as tf
import tensorflow_datasets as tfds
import functools
from abc import ABC, abstractmethod


class AbstractDataset(ABC):
    def __init__(self, batch_size, get_id=False):
        self.batch_size = batch_size
        self.get_id = get_id

    @property
    @abstractmethod
    def input_shape(self):
        pass

    @property
    @abstractmethod
    def num_classes(self):
        pass

    @abstractmethod
    def get_and_process(self):
        pass


class TfDataset(AbstractDataset):
    @abstractmethod
    def augmentation(self):
        pass

    @property
    @abstractmethod
    def dataset_name(self):
        pass

    def get_and_process(self):

        train_ds = tfds.load(
            self.dataset_name, split="train", shuffle_files=False, as_supervised=True
        )
        train_ds = train_ds.cache()
        train_ds = train_ds.map(
            functools.partial(self.augmentation, training=True),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        train_ds = train_ds.shuffle(self.batch_size * 50)
        train_ds = train_ds.batch(self.batch_size)
        train_ds = train_ds.prefetch(4)
        val_ds = tfds.load(
            self.dataset_name, split="test", shuffle_files=False, as_supervised=True
        )
        val_ds = val_ds.map(
            self.augmentation, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

        val_ds = val_ds.batch(self.batch_size)

        return train_ds, val_ds


####  PRIMARY DATASETS ####


class Mnist(TfDataset):
    @property
    def input_shape(self):
        return (28, 28, 1)

    @property
    def num_classes(self):
        return 10

    @property
    def dataset_name(self):
        return "mnist"

    def augmentation(self, image, label, training=False):
        return tf.cast(image, tf.float32) / 255.0, label


class FashionMnist(TfDataset):
    @property
    def input_shape(self):
        return (28, 28, 1)

    @property
    def num_classes(self):
        return 10

    @property
    def dataset_name(self):
        return "FashionMnist"

    def augmentation(self, image, label, training=False):
        return tf.cast(image, tf.float32) / 255.0, label


class Cifar10(TfDataset):
    @property
    def input_shape(self):
        return (32, 32, 3)

    @property
    def num_classes(self):
        return 10

    def augmentation(self, features_dict, training=False):
        image = features_dict["image"]
        label = features_dict["label"]
        id = features_dict["id"]
        if training:
            image = tf.image.random_flip_left_right(image)
            # image=tf.image.random_flip_up_down(image)

        if self.get_id:
            return tf.cast(image, tf.float32) / 255.0, label, id

        return tf.cast(image, tf.float32) / 255.0, label

    @property
    def dataset_name(self):
        return "cifar10"

    def get_and_process(self):

        train_ds = tfds.load(
            self.dataset_name,
            split="train",
            shuffle_files=False,
            as_supervised=False,
        )
        train_ds = train_ds.cache()
        train_ds = train_ds.map(
            functools.partial(self.augmentation, training=True),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        train_ds = train_ds.shuffle(self.batch_size * 50)
        train_ds = train_ds.batch(self.batch_size)
        train_ds = train_ds.prefetch(4)
        val_ds = tfds.load(
            self.dataset_name,
            split="test",
            shuffle_files=False,
            as_supervised=False,
        )
        val_ds = val_ds.map(
            self.augmentation, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

        val_ds = val_ds.batch(self.batch_size)

        return train_ds, val_ds


class ImageNetV2(TfDataset):
    @property
    def input_shape(self):
        return (224, 224, 3)

    @property
    def num_classes(self):
        return 1000

    def augmentation(self, image, label, training=False):
        image = tf.image.resize_with_crop_or_pad(
            image, self.input_shape[0], self.input_shape[1]
        )
        return image, label

    @property
    def dataset_name(self):
        return "imagenet_v2"

    def get_and_process(self):
        val_ds = tfds.load(
            self.dataset_name, split="test", shuffle_files=False, as_supervised=True
        )
        val_ds = val_ds.map(
            self.augmentation, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

        val_ds = val_ds.batch(self.batch_size)

        return val_ds


####  SECONDARY DATASETS ####
#############################


################## NEW - MAPS*INPUT IMAGES #############################################

######### CIFAR10*MAPS ################
########## SPLIT BY CLASS MAPS FOR MiniResNet(A) ##########
class ByClassMultipliedMapsMiniResNetGrad(AbstractDataset):
        @property
        def input_shape(self):
            return (32, 32, 3)

        @property
        def dataset_name(self):
            return "ByClassMultipliedMaps_MiniResNet_Grad"

        @property
        def num_classes(self):
            return 10

        @property
        def data_dir(self):
            return "../images/cifar10/gradient/ResNets/xInput/ByClassMultipliedMaps/MiniResNet/Training"

        def augmentation(self, image, label, training=False):
            if training:
                image = tf.image.random_flip_left_right(image)
                # image=tf.image.random_flip_up_down(image)

            return image, label

        def get_and_process(self):
            train_ds = tf.keras.preprocessing.image_dataset_from_directory(
                self.data_dir,
                color_mode="rgb",
                validation_split=0.2,
                subset="training",
                seed=123,
                image_size=(self.input_shape[0], self.input_shape[1]),
                batch_size=self.batch_size,
            )

            val_ds = tf.keras.preprocessing.image_dataset_from_directory(
                self.data_dir,
                color_mode="rgb",
                validation_split=0.2,
                subset="validation",
                seed=123,
                image_size=(self.input_shape[0], self.input_shape[1]),
                batch_size=self.batch_size,
            )

            return train_ds, val_ds

class ByClassMultipliedMapsMiniResNetSG(ByClassMultipliedMapsMiniResNetGrad):
        @property
        def dataset_name(self):
            return "ByClassMultipliedMaps_MiniResNet_SG"

        @property
        def data_dir(self):
            return "../images/cifar10/smoothGrad/ResNets/xInput/ByClassMultipliedMaps/MiniResNet/Training"


class ByClassMultipliedMapsMiniResNetIG(ByClassMultipliedMapsMiniResNetGrad):
        @property
        def dataset_name(self):
            return "ByClassMultipliedMaps_MiniResNet_IG"

        @property
        def data_dir(self):
            return "../images/cifar10/integratedGrad/ResNets/xInput/ByClassMultipliedMaps/MiniResNet/Training"



######### CIFAR10*MAPS ################
########## BINARY CLASSIFICATION BETWEEN MiniResNet(A) & MiNiResNetB MAPS*INPUT ##########
class MultipliedMapsABGrad(AbstractDataset):
        @property
        def input_shape(self):
            return (32, 32, 3)

        @property
        def dataset_name(self):
            return "MultipliedMaps_AB_Grad"

        @property
        def num_classes(self):
            return 2

        @property
        def data_dir(self):
            return "../images/cifar10/gradient/ResNets/xInput/MultipliedMaps/Training"

        def augmentation(self, image, label, training=False):
            if training:
                image = tf.image.random_flip_left_right(image)
                # image=tf.image.random_flip_up_down(image)

            return image, label

        def get_and_process(self):
            train_ds = tf.keras.preprocessing.image_dataset_from_directory(
                self.data_dir,
                color_mode="rgb",
                validation_split=0.2,
                subset="training",
                seed=123,
                image_size=(self.input_shape[0], self.input_shape[1]),
                batch_size=self.batch_size,
            )

            val_ds = tf.keras.preprocessing.image_dataset_from_directory(
                self.data_dir,
                color_mode="rgb",
                validation_split=0.2,
                subset="validation",
                seed=123,
                image_size=(self.input_shape[0], self.input_shape[1]),
                batch_size=self.batch_size,
            )

            return train_ds, val_ds

class MultipliedMapsABSG(MultipliedMapsABGrad):
        @property
        def dataset_name(self):
            return "MultipliedMaps_AB_SG"

        @property
        def data_dir(self):
            return "../images/cifar10/smoothGrad/ResNets/xInput/MultipliedMaps/Training"


class MultipliedMapsABSG(MultipliedMapsABGrad):
        @property
        def dataset_name(self):
            return "MultipliedMaps_AB_IG"

        @property
        def data_dir(self):
            return "../images/cifar10/integratedGrad/ResNets/xInput/MultipliedMaps/Training"




####### MAPS FROM CIFAR10  #########
####### BY ORIGINAL INPUT CLASS - DATA FOR 10 RESNETS ######
class SplitClass10ResNetsCifar10Grad(AbstractDataset):
    @property
    def input_shape(self):
        return (32, 32, 1)

    @property
    def dataset_name(self):
        return "SplitClass10ResNets_Cifar10_Grad"

    @property
    def num_classes(self):
        return 10

    @property
    def data_dir(self):
        return "../images/cifar10/gradient/ResNets/splitClassManyResNets/TenResNets/Training"

    def augmentation(self, image, label, training=False):
        if training:
            image = tf.image.random_flip_left_right(image)
            # image=tf.image.random_flip_up_down(image)

        return image, label

    def get_and_process(self):
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.data_dir,
            color_mode="grayscale",
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=self.batch_size,
        )

        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.data_dir,
            color_mode="grayscale",
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=self.batch_size,
        )

        return train_ds, val_ds


class SplitClass10ResNetsCifar10SG(SplitClass10ResNetsCifar10Grad):
    @property
    def dataset_name(self):
        return "SplitClass10ResNets_Cifar10_SG"

    @property
    def data_dir(self):
        return "../images/cifar10/smoothGrad/ResNets/splitClassManyResNets/TenResNets/Training"


class SplitClass10ResNetsCifar10IG(SplitClass10ResNetsCifar10Grad):
    @property
    def dataset_name(self):
        return "SplitClass10ResNets_Cifar10_IG"

    @property
    def data_dir(self):
        return "../images/cifar10/integratedGrad/ResNets/splitClassManyResNets/TenResNets/Training"


#### ELEVETH RESNET SPLIT BY CLASS TRAINING AND TEST SETS #####
### TRAINING SET BUT IN TESTING FORMAT!! ###
class SplitClass11thCifar10Grad(AbstractDataset):
    @property
    def input_shape(self):
        return (32, 32, 1)

    @property
    def num_classes(self):
        return 10

    @property
    def dataset_name(self):
        return "SplitClass11th_Cifar10_Grad"

    @property
    def data_dir(self):
        return "../images/cifar10/gradient/ResNets/splitClassManyResNets/EleventhResNet/Training"

    def get_and_process(self):
        test_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.data_dir,
            color_mode="grayscale",
            seed=123,
            image_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=self.batch_size,
        )
        return test_ds


class SplitClass11thCifar10SG(SplitClass11thCifar10Grad):
    @property
    def dataset_name(self):
        return "SplitClass11th_Cifar10_SG"

    @property
    def data_dir(self):
        return "../images/cifar10/smoothGrad/ResNets/splitClassManyResNets/EleventhResNet/Training"


class SplitClass11thCifar10IG(SplitClass11thCifar10Grad):
    @property
    def dataset_name(self):
        return "SplitClass11th_Cifar10_IG"

    @property
    def data_dir(self):
        return "../images/cifar10/integratedGrad/ResNets/splitClassManyResNets/EleventhResNet/Training"


######################################################################################
#### TEST SET ####
class SplitClass11thCifar10TESTGrad(SplitClass11thCifar10Grad):
    @property
    def dataset_name(self):
        return "SplitClass11th_Cifar10TEST_Grad"

    @property
    def data_dir(self):
        return "../images/cifar10/gradient/ResNets/splitClassManyResNets/EleventhResNet/Test_set"


class SplitClass11thCifar10TESTSG(SplitClass11thCifar10Grad):
    @property
    def dataset_name(self):
        return "SplitClass11th_Cifar10TEST_SG"

    @property
    def data_dir(self):
        return "../images/cifar10/smoothGrad/ResNets/splitClassManyResNets/EleventhResNet/Test_set"


class SplitClass11thCifar10TESTIG(SplitClass11thCifar10Grad):
    @property
    def dataset_name(self):
        return "SplitClass11th_Cifar10TEST_IG"

    @property
    def data_dir(self):
        return "../images/cifar10/integratedGrad/ResNets/splitClassManyResNets/EleventhResNet/Test_set"


################################################################################
####### BY ORIGINAL INPUT CLASS - DATA FOR 2 RESNETS COMBINED ######
class SplitClass2ResNetsCifar10Grad(AbstractDataset):
    @property
    def input_shape(self):
        return (32, 32, 1)

    @property
    def dataset_name(self):
        return "SplitClass2ResNets_Cifar10_Grad"

    @property
    def num_classes(self):
        return 10

    @property
    def data_dir(self):
        return "../images/cifar10/gradient/ResNets/splitClassManyResNets/AB/Training"

    def augmentation(self, image, label, training=False):
        if training:
            image = tf.image.random_flip_left_right(image)
            # image=tf.image.random_flip_up_down(image)

        return image, label

    def get_and_process(self):
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.data_dir,
            color_mode="grayscale",
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=self.batch_size,
        )

        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.data_dir,
            color_mode="grayscale",
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=self.batch_size,
        )

        return train_ds, val_ds


class SplitClass2ResNetsCifar10SG(SplitClass10ResNetsCifar10Grad):
    @property
    def dataset_name(self):
        return "SplitClass2ResNets_Cifar10_SG"

    @property
    def data_dir(self):
        return "../images/cifar10/smoothGrad/ResNets/splitClassManyResNets/AB/Training"


class SplitClass2ResNetsCifar10IG(SplitClass10ResNetsCifar10Grad):
    @property
    def dataset_name(self):
        return "SplitClass2ResNets_Cifar10_IG"

    @property
    def data_dir(self):
        return (
            "../images/cifar10/integratedGrad/ResNets/splitClassManyResNets/AB/Training"
        )


#### SPLIT BY CLASS MINICNN(A) FOR TESTING RESNET TRAINED NETWORKS ON ####
### TRAINING SET BUT IN TESTING FORMAT!! ###
class SplitClassCNNACifar10Grad(AbstractDataset):
    @property
    def input_shape(self):
        return (32, 32, 1)

    @property
    def num_classes(self):
        return 10

    @property
    def dataset_name(self):
        return "SplitClassCnnA_Cifar10_Grad"

    @property
    def data_dir(self):
        return (
            "../images/cifar10/gradient/MiniCNNs/splitClassManyCNNs/AB/MiniCNN/Training"
        )

    def get_and_process(self):
        test_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.data_dir,
            color_mode="grayscale",
            seed=123,
            image_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=self.batch_size,
        )
        return test_ds


class SplitClassCNNACifar10SG(SplitClassCNNACifar10Grad):
    @property
    def dataset_name(self):
        return "SplitClassCnnA_Cifar10_SG"

    @property
    def data_dir(self):
        return "../images/cifar10/smoothGrad/MiniCNNs/splitClassManyCNNs/AB/MiniCNN/Training"


class SplitClassCNNACifar10IG(SplitClassCNNACifar10Grad):
    @property
    def dataset_name(self):
        return "SplitClassCnnA_Cifar10_IG"

    @property
    def data_dir(self):
        return "../images/cifar10/integratedGrad/MiniCNNs/splitClassManyCNNs/AB/MiniCNN/Training"


#####################################################################
####### BY ORIGINAL INPUT CLASS ######
####### CIFAR10 RESNET (A) #########


class ResAByClassCifar10Grad(AbstractDataset):
    @property
    def input_shape(self):
        return (32, 32, 1)

    @property
    def dataset_name(self):
        return "ResA_ByClassCifar10_Grad"

    @property
    def num_classes(self):
        return 10

    @property
    def data_dir(self):
        return "../images/cifar10/gradient/ResNets/splitByClass/A"

    def augmentation(self, image, label, training=False):
        if training:
            image = tf.image.random_flip_left_right(image)
            # image=tf.image.random_flip_up_down(image)

        return image, label

    def get_and_process(self):
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.data_dir,
            color_mode="grayscale",
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=self.batch_size,
        )

        # train_ds = train_ds.unbatch()
        # train_ds = train_ds.map(
        #     functools.partial(self.augmentation, training=True),
        #     num_parallel_calls=tf.data.experimental.AUTOTUNE,
        # )
        # train_ds = train_ds.batch(self.batch_size)

        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.data_dir,
            color_mode="grayscale",
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=self.batch_size,
        )

        return train_ds, val_ds


class ResAByClassCifar10SG(AbstractDataset):
    @property
    def input_shape(self):
        return (32, 32, 1)

    @property
    def dataset_name(self):
        return "ResA_ByClassCifar10_SG"

    @property
    def num_classes(self):
        return 10

    @property
    def data_dir(self):
        return "../images/cifar10/smoothGrad/ResNets/splitByClass/A"

    def augmentation(self, image, label, training=False):
        if training:
            image = tf.image.random_flip_left_right(image)
            # image=tf.image.random_flip_up_down(image)

        return image, label

    def get_and_process(self):
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.data_dir,
            color_mode="grayscale",
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=self.batch_size,
        )

        # train_ds = train_ds.unbatch()
        # train_ds = train_ds.map(
        #     functools.partial(self.augmentation, training=True),
        #     num_parallel_calls=tf.data.experimental.AUTOTUNE,
        # )
        # train_ds = train_ds.batch(self.batch_size)

        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.data_dir,
            color_mode="grayscale",
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=self.batch_size,
        )

        return train_ds, val_ds


class ResAByClassCifar10IG(AbstractDataset):
    @property
    def input_shape(self):
        return (32, 32, 1)

    @property
    def dataset_name(self):
        return "ResA_ByClassCifar10_IG"

    @property
    def num_classes(self):
        return 10

    @property
    def data_dir(self):
        return "../images/cifar10/integratedGrad/ResNets/splitByClass/A"

    def augmentation(self, image, label, training=False):
        if training:
            image = tf.image.random_flip_left_right(image)
            # image=tf.image.random_flip_up_down(image)

        return image, label

    def get_and_process(self):
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.data_dir,
            color_mode="grayscale",
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=self.batch_size,
        )

        # train_ds = train_ds.unbatch()
        # train_ds = train_ds.map(
        #     functools.partial(self.augmentation, training=True),
        #     num_parallel_calls=tf.data.experimental.AUTOTUNE,
        # )
        # train_ds = train_ds.batch(self.batch_size)

        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.data_dir,
            color_mode="grayscale",
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=self.batch_size,
        )

        return train_ds, val_ds


####### CIFAR10 (TrainingSet for testing) MiniResNetB #########
class ResBByClassCifar10Grad(AbstractDataset):
    @property
    def input_shape(self):
        return (32, 32, 1)

    @property
    def dataset_name(self):
        return "ResB_ByClassCifar10_Grad"

    @property
    def num_classes(self):
        return 10

    @property
    def data_dir(self):
        return "../images/cifar10/gradient/ResNets/splitByClass/B"

    def augmentation(self, image, label, training=False):
        if training:
            image = tf.image.random_flip_left_right(image)
            # image=tf.image.random_flip_up_down(image)

        return image, label

    def get_and_process(self):
        return tf.keras.preprocessing.image_dataset_from_directory(
            self.data_dir,
            color_mode="grayscale",
            seed=123,
            image_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=self.batch_size,
        )


class ResBByClassCifar10SG(AbstractDataset):
    @property
    def input_shape(self):
        return (32, 32, 1)

    @property
    def dataset_name(self):
        return "ResB_ByClassCifar10_SG"

    @property
    def num_classes(self):
        return 10

    @property
    def data_dir(self):
        return "../images/cifar10/smoothGrad/ResNets/splitByClass/B"

    def augmentation(self, image, label, training=False):
        if training:
            image = tf.image.random_flip_left_right(image)
            # image=tf.image.random_flip_up_down(image)

        return image, label

    def get_and_process(self):
        return tf.keras.preprocessing.image_dataset_from_directory(
            self.data_dir,
            color_mode="grayscale",
            seed=123,
            image_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=self.batch_size,
        )


class ResBByClassCifar10IG(AbstractDataset):
    @property
    def input_shape(self):
        return (32, 32, 1)

    @property
    def dataset_name(self):
        return "ResB_ByClassCifar10_IG"

    @property
    def num_classes(self):
        return 10

    @property
    def data_dir(self):
        return "../images/cifar10/integratedGrad/ResNets/splitByClass/B"

    def augmentation(self, image, label, training=False):
        if training:
            image = tf.image.random_flip_left_right(image)
            # image=tf.image.random_flip_up_down(image)

        return image, label

    def get_and_process(self):
        return tf.keras.preprocessing.image_dataset_from_directory(
            self.data_dir,
            color_mode="grayscale",
            seed=123,
            image_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=self.batch_size,
        )


#####################################TESTDATASET Saliency maps###############################################################
####### CIFAR10 TESTSET MiniResNet(A) #########
#############################################
class ResTESTAByClassCifar10Grad(AbstractDataset):
    @property
    def input_shape(self):
        return (32, 32, 1)

    @property
    def dataset_name(self):
        return "ResTestA_ByClassCifar10_Grad"

    @property
    def num_classes(self):
        return 10

    @property
    def data_dir(self):
        return "../images/cifar10/gradient/ResNets/splitByClassTEST/A"

    def augmentation(self, image, label, training=False):
        if training:
            image = tf.image.random_flip_left_right(image)
            # image=tf.image.random_flip_up_down(image)

        return image, label

    def get_and_process(self):
        return tf.keras.preprocessing.image_dataset_from_directory(
            self.data_dir,
            color_mode="grayscale",
            seed=123,
            image_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=self.batch_size,
        )


class ResTESTAByClassCifar10SG(AbstractDataset):
    @property
    def input_shape(self):
        return (32, 32, 1)

    @property
    def dataset_name(self):
        return "ResTestA_ByClassCifar10_SG"

    @property
    def num_classes(self):
        return 10

    @property
    def data_dir(self):
        return "../images/cifar10/smoothGrad/ResNets/splitByClassTEST/A"

    def augmentation(self, image, label, training=False):
        if training:
            image = tf.image.random_flip_left_right(image)
            # image=tf.image.random_flip_up_down(image)

        return image, label

    def get_and_process(self):
        return tf.keras.preprocessing.image_dataset_from_directory(
            self.data_dir,
            color_mode="grayscale",
            seed=123,
            image_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=self.batch_size,
        )


class ResTESTAByClassCifar10IG(AbstractDataset):
    @property
    def input_shape(self):
        return (32, 32, 1)

    @property
    def dataset_name(self):
        return "ResTestA_ByClassCifar10_IG"

    @property
    def num_classes(self):
        return 10

    @property
    def data_dir(self):
        return "../images/cifar10/integratedGrad/ResNets/splitByClassTEST/A"

    def augmentation(self, image, label, training=False):
        if training:
            image = tf.image.random_flip_left_right(image)
            # image=tf.image.random_flip_up_down(image)

        return image, label

    def get_and_process(self):
        return tf.keras.preprocessing.image_dataset_from_directory(
            self.data_dir,
            color_mode="grayscale",
            seed=123,
            image_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=self.batch_size,
        )


####### CIFAR10 TESTSET MiniResNet(A) #########
#############################################
class ResTESTBByClassCifar10Grad(AbstractDataset):
    @property
    def input_shape(self):
        return (32, 32, 1)

    @property
    def dataset_name(self):
        return "ResTestB_ByClassCifar10_Grad"

    @property
    def num_classes(self):
        return 10

    @property
    def data_dir(self):
        return "../images/cifar10/gradient/ResNets/splitByClassTEST/B"

    def augmentation(self, image, label, training=False):
        if training:
            image = tf.image.random_flip_left_right(image)
            # image=tf.image.random_flip_up_down(image)

        return image, label

    def get_and_process(self):
        return tf.keras.preprocessing.image_dataset_from_directory(
            self.data_dir,
            color_mode="grayscale",
            seed=123,
            image_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=self.batch_size,
        )


class ResTESTBByClassCifar10SG(AbstractDataset):
    @property
    def input_shape(self):
        return (32, 32, 1)

    @property
    def dataset_name(self):
        return "ResTestB_ByClassCifar10_SG"

    @property
    def num_classes(self):
        return 10

    @property
    def data_dir(self):
        return "../images/cifar10/smoothGrad/ResNets/splitByClassTEST/B"

    def augmentation(self, image, label, training=False):
        if training:
            image = tf.image.random_flip_left_right(image)
            # image=tf.image.random_flip_up_down(image)

        return image, label

    def get_and_process(self):
        return tf.keras.preprocessing.image_dataset_from_directory(
            self.data_dir,
            color_mode="grayscale",
            seed=123,
            image_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=self.batch_size,
        )


class ResTESTBByClassCifar10IG(AbstractDataset):
    @property
    def input_shape(self):
        return (32, 32, 1)

    @property
    def dataset_name(self):
        return "ResTestB_ByClassCifar10_IG"

    @property
    def num_classes(self):
        return 10

    @property
    def data_dir(self):
        return "../images/cifar10/integratedGrad/ResNets/splitByClassTEST/B"

    def augmentation(self, image, label, training=False):
        if training:
            image = tf.image.random_flip_left_right(image)
            # image=tf.image.random_flip_up_down(image)

        return image, label

    def get_and_process(self):
        return tf.keras.preprocessing.image_dataset_from_directory(
            self.data_dir,
            color_mode="grayscale",
            seed=123,
            image_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=self.batch_size,
        )


#######################################################################
#######  SECONDARY IMAGENETV2 ########


class ResImageNetV2Grad(AbstractDataset):
    @property
    def input_shape(self):
        return (224, 224, 1)

    @property
    def dataset_name(self):
        return "Res_ImageNetV2_Grad"

    @property
    def num_classes(self):
        return 3

    @property
    def data_dir(self):
        return "../images/imagenet_v2/gradient"

    def get_and_process(self):
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.data_dir,
            color_mode="grayscale",
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=self.batch_size,
        )

        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.data_dir,
            color_mode="grayscale",
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=self.batch_size,
        )

        return train_ds, val_ds


class ResImageNetV2SmoothGrad(AbstractDataset):
    @property
    def input_shape(self):
        return (224, 224, 1)

    @property
    def dataset_name(self):
        return "Res_ImageNetV2_SG"

    @property
    def num_classes(self):
        return 3

    @property
    def data_dir(self):
        return "../images/imagenet_v2/smoothGrad"

    def get_and_process(self):
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.data_dir,
            color_mode="grayscale",
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=self.batch_size,
        )

        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.data_dir,
            color_mode="grayscale",
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=self.batch_size,
        )

        return train_ds, val_ds


class ResImageNetV2IntegratedGrad(AbstractDataset):
    @property
    def input_shape(self):
        return (224, 224, 1)

    @property
    def dataset_name(self):
        return "Res_ImageNetV2_IG"

    @property
    def num_classes(self):
        return 3

    @property
    def data_dir(self):
        return "../images/imagenet_v2/integratedGrad"

    def get_and_process(self):
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.data_dir,
            color_mode="grayscale",
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=self.batch_size,
        )

        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.data_dir,
            color_mode="grayscale",
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=self.batch_size,
        )

        return train_ds, val_ds


################# NUMPY DATASETS #################
#####(standardized datasets)#############
class StandRes2Cifar10Grad(AbstractDataset):
    @property
    def input_shape(self):
        return (32, 32, 1)

    @property
    def dataset_name(self):
        return "Stand_Res2_Cifar10_Grad"

    @property
    def num_classes(self):
        return 2

    @property
    def data_dir(self):
        return "../images/cifar10/gradient/ResNets/Training"

    def standardize(self, image, label):
        mean = tf.math.reduce_mean(image)
        std = tf.math.reduce_std(image)
        image = (image - mean) / std
        return image, label

    def get_and_process(self):
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.data_dir,
            color_mode="grayscale",
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=1,
        )

        train_ds = train_ds.unbatch()
        train_ds = train_ds.map(
            self.standardize, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        train_ds = train_ds.batch(self.batch_size)

        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.data_dir,
            color_mode="grayscale",
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=1,
        )
        val_ds = val_ds.unbatch()
        val_ds = val_ds.map(
            self.standardize, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        val_ds = val_ds.batch(self.batch_size)

        return train_ds, val_ds


##### CIFAR10 ######

####  SECONDARY DATASETS ####

# Dataset with 5 models for gradient saleincy method
class Maps5Cifar10Grad(AbstractDataset):
    @property
    def input_shape(self):
        return (32, 32, 1)

    @property
    def dataset_name(self):
        return "Maps5_Cifar10_Grad"

    @property
    def num_classes(self):
        return 5

    @property
    def data_dir(self):
        return "../images/cifar10/gradient/Training"

    def get_and_process(self):
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.data_dir,
            color_mode="grayscale",
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=self.batch_size,
        )

        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.data_dir,
            color_mode="grayscale",
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=self.batch_size,
        )

        return train_ds, val_ds


# Dataset with 5 models for smoothGrad saleincy method
class Maps5Cifar10SmoothGrad(Maps5Cifar10Grad):
    @property
    def dataset_name(self):
        return "Maps5_Cifar10_SmoothGrad"

    @property
    def data_dir(self):
        return "../images/cifar10/smoothGrad/Training"


# Dataset with 5 models for intergrated gradients saleincy method
class Maps5Cifar10Integrated(Maps5Cifar10Grad):
    @property
    def dataset_name(self):
        return "Maps5_Cifar10_Integrated"

    @property
    def data_dir(self):
        return "../images/cifar10/integratedGrad/Training"


##############################################
####### CNN LAYER COMPARE EXPERIMENTS #######
#############################################


#### GRADIENT ####
class LayersCNNs2c3(Maps5Cifar10Grad):
    @property
    def dataset_name(self):
        return "Layers_Cnns2c3_Grad"

    @property
    def num_classes(self):
        return 2

    @property
    def data_dir(self):
        return "../images/cifar10/gradient/Training/cnns_2_3"


class LayersCNNs11c12(Maps5Cifar10Grad):
    @property
    def dataset_name(self):
        return "Layers_Cnns11c12_Grad"

    @property
    def num_classes(self):
        return 2

    @property
    def data_dir(self):
        return "../images/cifar10/gradient/Training/cnns_11_12"


#### SMOOTHGRAD ####
class LayersCNNs2c3SG(Maps5Cifar10Grad):
    @property
    def dataset_name(self):
        return "Layers_Cnns2c3_SG"

    @property
    def num_classes(self):
        return 2

    @property
    def data_dir(self):
        return "../images/cifar10/smoothGrad/Training/cnns_2_3"


class LayersCNNs11c12SG(Maps5Cifar10Grad):
    @property
    def dataset_name(self):
        return "Layers_Cnns11c12_SG"

    @property
    def num_classes(self):
        return 2

    @property
    def data_dir(self):
        return "../images/cifar10/smoothGrad/Training/cnns_11_12"


#### INTEGRATEDGRAD ####
class LayersCNNs2c3IG(Maps5Cifar10Grad):
    @property
    def dataset_name(self):
        return "Layers_Cnns2c3_IG"

    @property
    def num_classes(self):
        return 2

    @property
    def data_dir(self):
        return "../images/cifar10/integratedGrad/Training/cnns_2_3"


class LayersCNNs11c12IG(Maps5Cifar10Grad):
    @property
    def dataset_name(self):
        return "Layers_Cnns11c12_IG"

    @property
    def num_classes(self):
        return 2

    @property
    def data_dir(self):
        return "../images/cifar10/integratedGrad/Training/cnns_11_12"


############################### NEW LAYERS EXPERIMETNS ########################

### GRAD ####
class LayersCNNs3c4Grad(LayersCNNs2c3):
    @property
    def dataset_name(self):
        return "Layers_Cnns3c4_Grad"

    @property
    def data_dir(self):
        return "../images/cifar10/gradient/Training/cnns_3_4"


class LayersCNNs4c5Grad(LayersCNNs2c3):
    @property
    def dataset_name(self):
        return "Layers_Cnns4c5_Grad"

    @property
    def data_dir(self):
        return "../images/cifar10/gradient/Training/cnns_4_5"


class LayersCNNs5c6Grad(LayersCNNs2c3):
    @property
    def dataset_name(self):
        return "Layers_Cnns5c6_Grad"

    @property
    def data_dir(self):
        return "../images/cifar10/gradient/Training/cnns_5_6"


class LayersCNNs2c6Grad(LayersCNNs2c3):
    @property
    def dataset_name(self):
        return "Layers_Cnns2c6_Grad"

    @property
    def data_dir(self):
        return "../images/cifar10/gradient/Training/cnns_2_6"


class LayersCNNs3c6Grad(LayersCNNs2c3):
    @property
    def dataset_name(self):
        return "Layers_Cnns3c6_Grad"

    @property
    def data_dir(self):
        return "../images/cifar10/gradient/Training/cnns_3_6"


#### SMOOTHGRAD ####
class LayersCNNs3c4SG(LayersCNNs2c3):
    @property
    def dataset_name(self):
        return "Layers_Cnns3c4_SG"

    @property
    def data_dir(self):
        return "../images/cifar10/smoothGrad/Training/cnns_3_4"


class LayersCNNs4c5SG(LayersCNNs2c3):
    @property
    def dataset_name(self):
        return "Layers_Cnns4c5_SG"

    @property
    def data_dir(self):
        return "../images/cifar10/smoothGrad/Training/cnns_4_5"


class LayersCNNs5c6SG(LayersCNNs2c3):
    @property
    def dataset_name(self):
        return "Layers_Cnns5c6_SG"

    @property
    def data_dir(self):
        return "../images/cifar10/smoothGrad/Training/cnns_5_6"


class LayersCNNs2c6SG(LayersCNNs2c3):
    @property
    def dataset_name(self):
        return "Layers_Cnns2c6_SG"

    @property
    def data_dir(self):
        return "../images/cifar10/smoothGrad/Training/cnns_2_6"


class LayersCNNs3c6SG(LayersCNNs2c3):
    @property
    def dataset_name(self):
        return "Layers_Cnns3c6_SG"

    @property
    def data_dir(self):
        return "../images/cifar10/smoothGrad/Training/cnns_3_6"


#### INTEGRATEDGRAD ####
class LayersCNNs3c4IG(LayersCNNs2c3):
    @property
    def dataset_name(self):
        return "Layers_Cnns3c4_IG"

    @property
    def data_dir(self):
        return "../images/cifar10/integratedGrad/Training/cnns_3_4"


class LayersCNNs4c5IG(LayersCNNs2c3):
    @property
    def dataset_name(self):
        return "Layers_Cnns4c5_IG"

    @property
    def data_dir(self):
        return "../images/cifar10/integratedGrad/Training/cnns_4_5"


class LayersCNNs5c6IG(LayersCNNs2c3):
    @property
    def dataset_name(self):
        return "Layers_Cnns5c6_IG"

    @property
    def data_dir(self):
        return "../images/cifar10/integratedGrad/Training/cnns_5_6"


class LayersCNNs2c6IG(LayersCNNs2c3):
    @property
    def dataset_name(self):
        return "Layers_Cnns2c6_IG"

    @property
    def data_dir(self):
        return "../images/cifar10/integratedGrad/Training/cnns_2_6"


class LayersCNNs3c6IG(LayersCNNs2c3):
    @property
    def dataset_name(self):
        return "Layers_Cnns3c6_IG"

    @property
    def data_dir(self):
        return "../images/cifar10/integratedGrad/Training/cnns_3_6"


##############################
######################################################################
######RESNET EXPERIMENT ######
class Res2Cifar10Grad(Maps5Cifar10Grad):
    @property
    def dataset_name(self):
        return "Res2_Cifar10_Grad"

    @property
    def num_classes(self):
        return 2

    @property
    def data_dir(self):
        return "../images/cifar10/gradient/ResNets/Training"


class Res2Cifar10SmoothGrad(Maps5Cifar10Grad):
    @property
    def dataset_name(self):
        return "Res2_Cifar10_SmoothGrad"

    @property
    def num_classes(self):
        return 2

    @property
    def data_dir(self):
        return "../images/cifar10/smoothGrad/ResNets/Training"


class Res2Cifar10Integrated(Maps5Cifar10Grad):
    @property
    def dataset_name(self):
        return "Res2_Cifar10_Integrated"

    @property
    def num_classes(self):
        return 2

    @property
    def data_dir(self):
        return "../images/cifar10/integratedGrad/ResNets/Training"


#################################################
###### 10 RANDOM RESNETS EXPERIMENT ############
###################################################################
class ManyResNetsGrad(AbstractDataset):
    @property
    def input_shape(self):
        return (32, 32, 1)

    @property
    def dataset_name(self):
        return "ManyResNets_Grad"

    @property
    def num_classes(self):
        return 10

    ## Note, using the Test_set for training
    @property
    def data_dir(self):
        return "../images/cifar10/gradient/ResNets/ManyResNets/Test_set"

    def get_and_process(self):
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.data_dir,
            color_mode="grayscale",
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=self.batch_size,
        )

        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.data_dir,
            color_mode="grayscale",
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=self.batch_size,
        )

        return train_ds, val_ds


class ManyResNetsSG(ManyResNetsGrad):
    @property
    def dataset_name(self):
        return "ManyResNets_SG"

    ## Note, using the Test_set for training
    @property
    def data_dir(self):
        return "../images/cifar10/smoothGrad/ResNets/ManyResNets/Test_set"


class ManyResNetsIG(ManyResNetsGrad):
    @property
    def dataset_name(self):
        return "ManyResNets_IG"

    ## Note, using the Test_set for training
    @property
    def data_dir(self):
        return "../images/cifar10/integratedGrad/ResNets/ManyResNets/Test_set"


#########################################################################################
#############Comparing 2 MiniCNNs ##############
class MiniCNN2Cifar10Grad(Res2Cifar10Grad):
    @property
    def dataset_name(self):
        return "MiniCNN2_Cifar10_Grad"

    @property
    def data_dir(self):
        return "../images/cifar10/gradient/MiniCNNs/Training"


class MiniCNN2Cifar10SmoothGrad(Res2Cifar10Grad):
    @property
    def dataset_name(self):
        return "MiniCNN2_Cifar10_SmoothGrad"

    @property
    def data_dir(self):
        return "../images/cifar10/smoothGrad/MiniCNNs/Training"


class MiniCNN2Cifar10Integrated(Res2Cifar10Grad):
    @property
    def dataset_name(self):
        return "MiniCNN2_Cifar10_Integrated"

    @property
    def data_dir(self):
        return "../images/cifar10/integratedGrad/MiniCNNs/Training"


#############Comparing 2 small2CNNs ##############


class Small2CNN2Cifar10Grad(Res2Cifar10Grad):
    @property
    def dataset_name(self):
        return "Small2CNN2_Cifar10_Grad"

    @property
    def data_dir(self):
        return "../images/cifar10/gradient/Small2CNNs/Training"


class Small2CNN2Cifar10SmoothGrad(Res2Cifar10Grad):
    @property
    def dataset_name(self):
        return "Small2CNN2_Cifar10_SmoothGrad"

    @property
    def data_dir(self):
        return "../images/cifar10/smoothGrad/Small2CNNs/Training"


class Small2CNN2Cifar10Integrated(Res2Cifar10Grad):
    @property
    def dataset_name(self):
        return "Small2CNN2_Cifar10_Integrated"

    @property
    def data_dir(self):
        return "../images/cifar10/integratedGrad/Small2CNNs/Training"


#############Comparing 3 "A Networks"  ##############
####################################################


class ANetworks3Cifar10Grad(Maps5Cifar10Grad):
    @property
    def dataset_name(self):
        return "ANetworks3_Cifar10_Grad"

    @property
    def num_classes(self):
        return 3

    @property
    def data_dir(self):
        return "../images/cifar10/gradient/AB/ANetworks"


### B networks (same as A, but different initialisation)
class TESTBNetworks3Cifar10Grad(AbstractDataset):
    @property
    def input_shape(self):
        return (32, 32, 1)

    @property
    def dataset_name(self):
        return "TESTBNetworks3_Cifar10_Grad"

    @property
    def num_classes(self):
        return 3

    @property
    def data_dir(self):
        return "../images/cifar10/gradient/AB/BNetworks"

    def get_and_process(self):
        return tf.keras.preprocessing.image_dataset_from_directory(
            self.data_dir,
            color_mode="grayscale",
            seed=123,
            image_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=self.batch_size,
        )


class TESTANetworks3Cifar10Grad(AbstractDataset):
    @property
    def input_shape(self):
        return (32, 32, 1)

    @property
    def dataset_name(self):
        return "TESTANetworks3_Cifar10_Grad"

    @property
    def num_classes(self):
        return 3

    @property
    def data_dir(self):
        return "../images/cifar10/gradient/AB/ANetworks"

    def get_and_process(self):
        return tf.keras.preprocessing.image_dataset_from_directory(
            self.data_dir,
            color_mode="grayscale",
            seed=123,
            image_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=self.batch_size,
        )


######Randomized ResNet maps #######
##############################


class RandRes2Cifar10Grad(Maps5Cifar10Grad):
    @property
    def dataset_name(self):
        return "Rand_Res2_Cifar10_Grad"

    @property
    def num_classes(self):
        return 2

    @property
    def data_dir(self):
        return "../images/cifar10/gradient/ResNets/rand_Training"


class RandRes2Cifar10SmoothGrad(Maps5Cifar10Grad):
    @property
    def dataset_name(self):
        return "Rand_Res2_Cifar10_SmoothGrad"

    @property
    def num_classes(self):
        return 2

    @property
    def data_dir(self):
        return "../images/cifar10/smoothGrad/ResNets/rand_Training"


class RandRes2Cifar10Integrated(Maps5Cifar10Grad):
    @property
    def dataset_name(self):
        return "Rand_Res2_Cifar10_Integrated"

    @property
    def num_classes(self):
        return 2

    @property
    def data_dir(self):
        return "../images/cifar10/integratedGrad/ResNets/rand_Training"


######Standardized ResNet maps #######
##############################

# class StandRes2Cifar10Grad(Maps5Cifar10Grad):
#     @property
#     def dataset_name(self):
#         return "Stand_Res2_Cifar10_Grad"
#
#     @property
#     def num_classes(self):
#         return 2
#
#     @property
#     def data_dir(self):
#         return "../images/cifar10/gradient/ResNets/z_standardized/Training"
#
#


##### TEST SETS - FOR EVALUATING MODElS######


### Cifar10 ####
# TEST Dataset with 5 models for gradient saleincy method
class TESTMaps5Cifar10Grad(AbstractDataset):
    @property
    def input_shape(self):
        return (32, 32, 1)

    @property
    def dataset_name(self):
        return "TESTMaps5_Cifar10_Grad"

    @property
    def num_classes(self):
        return 5

    @property
    def data_dir(self):
        return "../images/cifar10/gradient/Test_set"

    def get_and_process(self):
        return tf.keras.preprocessing.image_dataset_from_directory(
            self.data_dir,
            color_mode="grayscale",
            seed=123,
            image_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=self.batch_size,
        )


# TEST Dataset with 5 models for smoothGrad saleincy method
class TESTMaps5Cifar10SmoothGrad(TESTMaps5Cifar10Grad):
    @property
    def dataset_name(self):
        return "TESTMaps5_Cifar10_SmoothGrad"

    @property
    def data_dir(self):
        return "../images/cifar10/smoothGrad/Test_set"


# TEST Dataset with 5 models for intergrated gradients saleincy method
class TESTMaps5Cifar10Integrated(TESTMaps5Cifar10Grad):
    @property
    def dataset_name(self):
        return "TESTMaps5_Cifar10_Integrated"

    @property
    def data_dir(self):
        return "../images/cifar10/integratedGrad/Test_set"


####################################
### Test set for 2 ResNets ###
class TESTRes2Cifar10Grad(TESTMaps5Cifar10Grad):
    @property
    def dataset_name(self):
        return "TestRes2_Cifar10_Grad"

    @property
    def num_classes(self):
        return 2

    @property
    def data_dir(self):
        return "../images/cifar10/gradient/ResNets/Test_set"


class TESTRes2Cifar10SmoothGrad(TESTMaps5Cifar10Grad):
    @property
    def dataset_name(self):
        return "TestRes2_Cifar10_SmoothGrad"

    @property
    def num_classes(self):
        return 2

    @property
    def data_dir(self):
        return "../images/cifar10/smoothGrad/ResNets/Test_set"


class TESTRes2Cifar10Integrated(TESTMaps5Cifar10Grad):
    @property
    def dataset_name(self):
        return "TestRes2_Cifar10_Integrated"

    @property
    def num_classes(self):
        return 2

    @property
    def data_dir(self):
        return "../images/cifar10/integratedGrad/ResNets/Test_set"


######Randomized ResNet maps #######
##############################


class TESTRandRes2Cifar10Grad(TESTMaps5Cifar10Grad):
    @property
    def dataset_name(self):
        return "testRand_Res2_Cifar10_Grad"

    @property
    def num_classes(self):
        return 2

    @property
    def data_dir(self):
        return "../images/cifar10/gradient/ResNets/rand_Test_set"


class TESTRandRes2Cifar10SmoothGrad(TESTMaps5Cifar10Grad):
    @property
    def dataset_name(self):
        return "testRand_Res2_Cifar10_SmoothGrad"

    @property
    def num_classes(self):
        return 2

    @property
    def data_dir(self):
        return "../images/cifar10/smoothGrad/ResNets/rand_Test_set"


class TESTRandRes2Cifar10Integrated(TESTMaps5Cifar10Grad):
    @property
    def dataset_name(self):
        return "testRand_Res2_Cifar10_Integrated"

    @property
    def num_classes(self):
        return 2

    @property
    def data_dir(self):
        return "../images/cifar10/integratedGrad/ResNets/rand_Test_set"


#### FashionMnist ####

# Dataset with 5 models for gradient saleincy method
class Maps5FashionMnistGrad(AbstractDataset):
    @property
    def input_shape(self):
        return (28, 28, 1)

    @property
    def dataset_name(self):
        return "Maps5_FashionMnist_Grad"

    @property
    def num_classes(self):
        return 5

    @property
    def data_dir(self):
        return "../images/FashionMnist/gradient/Training"

    def get_and_process(self):
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.data_dir,
            color_mode="grayscale",
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=self.batch_size,
        )

        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.data_dir,
            color_mode="grayscale",
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=self.batch_size,
        )

        return train_ds, val_ds


# Dataset with 5 models for smoothGrad saleincy method
class Maps5FashionMnistSmoothGrad(AbstractDataset):
    @property
    def input_shape(self):
        return (28, 28, 1)

    @property
    def dataset_name(self):
        return "Maps5_FashionMnist_SmoothGrad"

    @property
    def num_classes(self):
        return 5

    @property
    def data_dir(self):
        return "../images/FashionMnist/smoothGrad/Training"

    def get_and_process(self):
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.data_dir,
            color_mode="grayscale",
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=self.batch_size,
        )

        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.data_dir,
            color_mode="grayscale",
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=self.batch_size,
        )

        return train_ds, val_ds


# Dataset with 5 models for intergrated gradients saleincy method
class Maps5FashionMnistIntegrated(AbstractDataset):
    @property
    def input_shape(self):
        return (28, 28, 1)

    @property
    def dataset_name(self):
        return "Maps5_FashionMnist_Integrated"

    @property
    def num_classes(self):
        return 5

    @property
    def data_dir(self):
        return "../images/FashionMnist/integratedGrad/Training"

    def get_and_process(self):
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.data_dir,
            color_mode="grayscale",
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=self.batch_size,
        )

        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.data_dir,
            color_mode="grayscale",
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=self.batch_size,
        )

        return train_ds, val_ds


### FashionMNist ####
# TEST Dataset with 5 models for gradient saleincy method
class TESTMaps5FashionMnistGrad(AbstractDataset):
    @property
    def input_shape(self):
        return (28, 28, 1)

    @property
    def dataset_name(self):
        return "TESTMaps5_FashionMnist_Grad"

    @property
    def num_classes(self):
        return 5

    @property
    def data_dir(self):
        return "../images/FashionMnist/gradient/Test_set"

    def get_and_process(self):
        return tf.keras.preprocessing.image_dataset_from_directory(
            self.data_dir,
            color_mode="grayscale",
            seed=123,
            image_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=self.batch_size,
        )


# TEST Dataset with 5 models for smoothGrad saleincy method
class TESTMaps5FashionMnistSmoothGrad(AbstractDataset):
    @property
    def input_shape(self):
        return (28, 28, 1)

    @property
    def dataset_name(self):
        return "TESTMaps5_FashionMnist_SmoothGrad"

    @property
    def num_classes(self):
        return 5

    @property
    def data_dir(self):
        return "../images/FashionMnist/smoothGrad/Test_set"

    def get_and_process(self):
        return tf.keras.preprocessing.image_dataset_from_directory(
            self.data_dir,
            color_mode="grayscale",
            seed=123,
            image_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=self.batch_size,
        )


# TEST Dataset with 5 models for intergrated gradients saleincy method
class TESTMaps5FashionMnistIntegrated(AbstractDataset):
    @property
    def input_shape(self):
        return (28, 28, 1)

    @property
    def dataset_name(self):
        return "TESTMaps5_FashionMnist_Integrated"

    @property
    def num_classes(self):
        return 5

    @property
    def data_dir(self):
        return "../images/FashionMnist/integratedGrad/Test_set"

    def get_and_process(self):
        return tf.keras.preprocessing.image_dataset_from_directory(
            self.data_dir,
            color_mode="grayscale",
            seed=123,
            image_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=self.batch_size,
        )


### OLD ###
# class MnistSaliency(AbstractDataset):
#     @property
#     def input_shape(self):
#         return (28, 28, 1)
#
#     @property
#     def num_classes(self):
#         return 2
#
#     @property
#     def data_dir(self):
#         return "../images/MNIST_Maps"
#
#     def get_and_process(self):
#         train_ds = tf.keras.preprocessing.image_dataset_from_directory(
#             self.data_dir,
#             color_mode="grayscale",
#             validation_split=0.2,
#             subset="training",
#             seed=123,
#             image_size=(self.input_shape[0], self.input_shape[1]),
#             batch_size=self.batch_size,
#         )
#
#         val_ds = tf.keras.preprocessing.image_dataset_from_directory(
#             self.data_dir,
#             color_mode="grayscale",
#             validation_split=0.2,
#             subset="validation",
#             seed=123,
#             image_size=(self.input_shape[0], self.input_shape[1]),
#             batch_size=self.batch_size,
#         )
#
#         return train_ds, val_ds
#
#
# class MapsFashionMnist(AbstractDataset):
#     @property
#     def input_shape(self):
#         return (28, 28, 1)
#
#     @property
#     def dataset_name(self):
#         return "3Maps_FashionMnist"
#
#     @property
#     def num_classes(self):
#         return 3
#
#     @property
#     def data_dir(self):
#         return "../images/FashionMnist/Training"
#
#     def get_and_process(self):
#         train_ds = tf.keras.preprocessing.image_dataset_from_directory(
#             self.data_dir,
#             color_mode="grayscale",
#             validation_split=0.2,
#             subset="training",
#             seed=123,
#             image_size=(self.input_shape[0], self.input_shape[1]),
#             batch_size=self.batch_size,
#         )
#
#         val_ds = tf.keras.preprocessing.image_dataset_from_directory(
#             self.data_dir,
#             color_mode="grayscale",
#             validation_split=0.2,
#             subset="validation",
#             seed=123,
#             image_size=(self.input_shape[0], self.input_shape[1]),
#             batch_size=self.batch_size,
#         )
#
#         return train_ds, val_ds
#
#
# class cnn2MapsFashionMnist(AbstractDataset):
#     @property
#     def input_shape(self):
#         return (28, 28, 1)
#
#     @property
#     def dataset_name(self):
#         return "cnn2Maps_FashionMnist"
#
#     @property
#     def num_classes(self):
#         return 2
#
#     @property
#     def data_dir(self):
#         return "../images/fashion_mnist/Training"
#
#     def get_and_process(self):
#         train_ds = tf.keras.preprocessing.image_dataset_from_directory(
#             self.data_dir,
#             color_mode="grayscale",
#             validation_split=0.2,
#             subset="training",
#             seed=123,
#             image_size=(self.input_shape[0], self.input_shape[1]),
#             batch_size=self.batch_size,
#         )
#
#         val_ds = tf.keras.preprocessing.image_dataset_from_directory(
#             self.data_dir,
#             color_mode="grayscale",
#             validation_split=0.2,
#             subset="validation",
#             seed=123,
#             image_size=(self.input_shape[0], self.input_shape[1]),
#             batch_size=self.batch_size,
#         )
#
#         return train_ds, val_ds
#
#
# class TestMapsFashionMnist(MapsFashionMnist):
#     @property
#     def dataset_name(self):
#         return "Test_Maps_FashionMnist"
#
#     @property
#     def data_dir(self):
#         return "../images/FashionMnist/Test_set"
#
#     def get_and_process(self):
#         return tf.keras.preprocessing.image_dataset_from_directory(
#             self.data_dir,
#             color_mode="grayscale",
#             seed=123,
#             image_size=(self.input_shape[0], self.input_shape[1]),
#             batch_size=self.batch_size,
#         )
#
#
# class test_vs_train_FashionMnist(AbstractDataset):
#     @property
#     def input_shape(self):
#         return (28, 28, 1)
#
#     @property
#     def dataset_name(self):
#         return "testVtrain_fmnist"
#
#     @property
#     def num_classes(self):
#         return 2
#
#     @property
#     def data_dir(self):
#         return "../images/FashionMnist/vs_test-train"
#
#     def get_and_process(self):
#         train_ds = tf.keras.preprocessing.image_dataset_from_directory(
#             self.data_dir,
#             color_mode="grayscale",
#             validation_split=0.2,
#             subset="training",
#             seed=123,
#             image_size=(self.input_shape[0], self.input_shape[1]),
#             batch_size=self.batch_size,
#         )
#
#         val_ds = tf.keras.preprocessing.image_dataset_from_directory(
#             self.data_dir,
#             color_mode="grayscale",
#             validation_split=0.2,
#             subset="validation",
#             seed=123,
#             image_size=(self.input_shape[0], self.input_shape[1]),
#             batch_size=self.batch_size,
#         )
#
#         return train_ds, val_ds
