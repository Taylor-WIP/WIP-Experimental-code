import tensorflow as tf
from abc import ABC, abstractmethod


class AbstractModel(ABC):
    def __init__(self, dataset):
        self.input_shape = dataset.input_shape
        self.num_classes = dataset.num_classes

    @property
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def build(self):
        pass


####### PRETRAINED MODELS #######
##################################


class ResNet50(AbstractModel):
    name = "ResNet50"

    def build(self):
        return tf.keras.applications.resnet50.ResNet50(
            weights="imagenet", include_top=True
        )


class ResNet101(AbstractModel):
    name = "ResNet101"

    def build(self):
        return tf.keras.applications.resnet.ResNet101(
            weights="imagenet", include_top=True
        )


class ResNet152(AbstractModel):
    name = "ResNet152"

    def build(self):
        return tf.keras.applications.resnet.ResNet152(
            weights="imagenet", include_top=True
        )


######################################################################################################
####### ResNet LAYERS NETWORKS ############


class ResNet3L(AbstractModel):
    name = "Res_3L"
    width = 64
    depth = 1

    def res_block(self, x, num_channels, strides=1):
        input = x
        x = tf.keras.layers.Conv2D(
            num_channels,
            (3, 3),
            strides,
            padding="same",
        )(input)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2D(num_channels, (3, 3), padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        if strides == 1:
            x = tf.keras.layers.Add()([x, input])
        return x

    def build(self):
        input = tf.keras.Input(self.input_shape)
        x = tf.keras.layers.Conv2D(self.width, (3, 3), strides=2, padding="same")(input)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        for i in range(0, self.depth):
            x = self.res_block(x, self.width)

        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(self.num_classes)(x)
        x = tf.keras.layers.Activation("softmax")(x)

        return tf.keras.Model(inputs=input, outputs=x)


###### CNN LAYERS NETWORKS ######
##################################


class CNN2L(AbstractModel):
    name = "CNN_2L"
    width = 64
    depth = 1

    def block(self, x, num_channels, strides=1):
        input = x
        x = tf.keras.layers.Conv2D(num_channels, (3, 3), strides, padding="same")(input)
        # x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        return x

    def build(self):
        input = tf.keras.Input(self.input_shape)
        x = tf.keras.layers.Conv2D(self.width, (3, 3), strides=2, padding="same")(input)
        # x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        for i in range(0, self.depth):
            x = self.block(x, self.width)

        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(self.num_classes)(x)
        x = tf.keras.layers.Activation("softmax")(x)

        return tf.keras.Model(inputs=input, outputs=x)


class CNN3L(CNN2L):
    depth = 1

    @property
    def name(self):
        return "CNN_{}L".format(self.depth + 2)

    @property
    def base_model_path(self):
        return "../models/CNN_{}L_cifar10.h5".format(self.depth + 1)

    def build(self):
        base_model = super().build()

        base_model.load_weights(self.base_model_path)
        for layer in base_model.layers[:-3]:
            layer.trainable = False

        part_model_output = base_model.layers[-4].output

        x = self.block(part_model_output, self.width)

        x = base_model.layers[-3](x)
        x = base_model.layers[-2](x)
        x = base_model.layers[-1](x)

        inputs = base_model.inputs
        return tf.keras.Model(inputs=inputs, outputs=x)


class CNN4L(CNN3L):
    depth = 2


class CNN5L(CNN3L):
    depth = 3


class CNN6L(CNN3L):
    depth = 4


class CNN11L(CNN2L):
    ##preveiously called 18L
    name = "CNN_11L"
    depth = 10


class CNN12L(CNN11L):
    # depth of previous model excluding first layer
    depth = 10

    @property
    def name(self):
        return "CNN_{}L".format(self.depth + 2)

    @property
    def base_model_path(self):
        return "../models/CNN_{}L_cifar10.h5".format(self.depth + 1)

    def build(self):
        # build model with depth of previous (-1 layers) model
        base_model = super().build()
        # Load in weights from the base trained model
        base_model.load_weights(self.base_model_path)
        for layer in base_model.layers[:-3]:
            layer.trainable = False

        part_model_output = base_model.layers[-4].output

        x = self.block(part_model_output, self.width)

        x = base_model.layers[-3](x)
        x = base_model.layers[-2](x)
        x = base_model.layers[-1](x)

        inputs = base_model.inputs
        return tf.keras.Model(inputs=inputs, outputs=x)


class CNN13L(CNN11L):
    depth = 11

    @property
    def name(self):
        return "CNN_{}L".format(self.depth + 2)

    @property
    def base_model_path(self):
        return "../models/CNN_{}L_cifar10.h5".format(self.depth + 1)

    def build(self):
        base_model = super().build()

        base_model.load_weights(self.base_model_path)
        for layer in base_model.layers[:-3]:
            layer.trainable = False

        part_model_output = base_model.layers[-4].output

        x = self.block(part_model_output, self.width)

        x = base_model.layers[-3](x)
        x = base_model.layers[-2](x)
        x = base_model.layers[-1](x)

        inputs = base_model.inputs
        return tf.keras.Model(inputs=inputs, outputs=x)


# super.build
#
# model.load_weights
#
# for layer in model:
#     layer.trainable=False
#
# x


########


class Small2CNN(AbstractModel):
    name = "Small2CNN"

    def build(self):
        model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Conv2D(
                    32, (3, 3), activation="relu", input_shape=self.input_shape
                ),
                tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(512, activation="relu"),
                tf.keras.layers.Dropout(0.4),
                tf.keras.layers.Dense(self.num_classes),
                tf.keras.layers.Activation("softmax"),
            ]
        )
        return model


class Small2CNNB(Small2CNN):
    name = "Small2CNNB"


class Small3CNN(AbstractModel):
    name = "Small3CNN"

    def build(self):
        model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Conv2D(
                    32, (3, 3), activation="relu", input_shape=self.input_shape
                ),
                tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(512, activation="relu"),
                tf.keras.layers.Dropout(0.4),
                tf.keras.layers.Dense(self.num_classes),
                tf.keras.layers.Activation("softmax"),
            ]
        )
        return model


class MLP(AbstractModel):
    name = "MLP"

    def build(self):
        model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Flatten(input_shape=self.input_shape),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(self.num_classes),
                tf.keras.layers.Activation("softmax"),
            ]
        )
        return model


class MiniResNet(AbstractModel):
    name = "MiniResNet"
    width = 32
    depth = 1
    kernel_initializer = "glorot_uniform"

    def res_block(self, x, num_channels, strides=1):
        input = x
        x = tf.keras.layers.Conv2D(
            num_channels,
            (3, 3),
            strides,
            padding="same",
            kernel_initializer=self.kernel_initializer,
        )(input)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2D(num_channels, (3, 3), padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        if strides == 1:
            x = tf.keras.layers.Add()([x, input])
        return x

    def build(self):
        input = tf.keras.Input(self.input_shape)
        x = tf.keras.layers.Conv2D(self.width, (3, 3), strides=2, padding="same")(input)
        x = tf.keras.layers.BatchNormalization()(x)
        for i in range(0, self.depth):
            x = self.res_block(x, self.width)

        x = self.res_block(x, self.width * 2, strides=2)
        for i in range(0, self.depth):
            x = self.res_block(x, self.width * 2)

        x = self.res_block(x, self.width * 4, strides=2)
        for i in range(0, self.depth):
            x = self.res_block(x, self.width * 4)

        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.4)(x)

        x = tf.keras.layers.Dense(self.num_classes)(x)
        x = tf.keras.layers.Activation("softmax")(x)

        return tf.keras.Model(inputs=input, outputs=x)


class MiniResNetB(MiniResNet):
    name = "MiniResNetB"
    kernel_initializer = tf.keras.initializers.GlorotUniform(seed=None)


class MiniResNetC(MiniResNet):
    name = "MiniResNetC"


class MiniResNetD(MiniResNet):
    name = "MiniResNetD"


class MiniResNetE(MiniResNet):
    name = "MiniResNetE"


class MiniResNetF(MiniResNet):
    name = "MiniResNetF"


class MiniResNetG(MiniResNet):
    name = "MiniResNetG"


class MiniResNetH(MiniResNet):
    name = "MiniResNetH"


class MiniResNetI(MiniResNet):
    name = "MiniResNetI"


class MiniResNetJ(MiniResNet):
    name = "MiniResNetJ"


class MiniResNetK(MiniResNet):
    name = "MiniResNetK"


class MiniResNet1(MiniResNet):
    name = "MiniResNet1"
    kernel_initializer = tf.keras.initializers.GlorotNormal(seed=None)


class MiniCNN(AbstractModel):
    name = "MiniCNN"
    width = 32
    depth = 1
    dropout = 0.4

    def block(self, x, num_channels, strides=1):
        input = x
        x = tf.keras.layers.Conv2D(num_channels, (3, 3), strides, padding="same")(input)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2D(num_channels, (3, 3), padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        return x

    def build(self):
        input = tf.keras.Input(self.input_shape)
        x = tf.keras.layers.Conv2D(self.width, (3, 3), strides=2, padding="same")(input)
        x = tf.keras.layers.BatchNormalization()(x)
        for i in range(0, self.depth):
            x = self.block(x, self.width)

        x = self.block(x, self.width * 2, strides=2)
        for i in range(0, self.depth):
            x = self.block(x, self.width * 2)

        x = self.block(x, self.width * 4, strides=2)
        for i in range(0, self.depth):
            x = self.block(x, self.width * 4)

        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(self.dropout)(x)

        x = tf.keras.layers.Dense(self.num_classes)(x)
        x = tf.keras.layers.Activation("softmax")(x)

        return tf.keras.Model(inputs=input, outputs=x)


class MiniCNNB(MiniCNN):
    name = "MiniCNNB"


class MiniCNNC(MiniCNN):
    name = "MiniCNNC"


class MiniCNND(MiniCNN):
    name = "MiniCNND"


class MiniCNNE(MiniCNN):
    name = "MiniCNNE"


class MiniCNNF(MiniCNN):
    name = "MiniCNNF"


class MiniCNNG(MiniCNN):
    name = "MiniCNNG"


class MiniCNNH(MiniCNN):
    name = "MiniCNNH"


class MiniCNNI(MiniCNN):
    name = "MiniCNNI"


class MiniCNNJ(MiniCNN):
    name = "MiniCNNJ"


class MiniCNNK(MiniCNN):
    name = "MiniCNNK"


class MiniCNNX(MiniCNN):
    name = "MiniCNNX"

#### SECONDARY MODELS ######


class EnhancedMiniCNN(MiniCNN):
    name = "EnhancedMiniCNN"
    depth = 2
    width = 64
    dropout = 0.6

    def block(self, x, num_channels, strides=1):
        x = super().block(x, num_channels, strides)
        x = tf.keras.layers.ReLU()(x)
        return x


class Secondary2CNN(AbstractModel):
    name = "Secondary2CNN"

    def build(self):
        model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Conv2D(
                    32, (3, 3), activation="relu", input_shape=self.input_shape
                ),
                tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(512, activation="relu"),
                tf.keras.layers.Dense(self.num_classes),
                tf.keras.layers.Activation("softmax"),
            ]
        )
        return model
