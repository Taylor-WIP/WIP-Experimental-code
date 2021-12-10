import tensorflow as tf
import numpy as np


from tensorflow.keras.models import load_model


class EvaluateModel:
    def __init__(self, model, model_path, dataset, batch_size=1):
        self.model = model(dataset(batch_size))
        self.model_path = model_path
        self.dataset = dataset(batch_size)

    def load_model(self):
        model = self.model.build()
        model.load_weights(self.model_path)
        model.summary()
        return model

    def evaluate(self):

        (test_ds) = self.dataset.get_and_process()

        model = self.load_model()
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        model.evaluate(test_ds)
        #

        # classifications =[[],[],[]]
        #
        # count=0
        # for image_batch, labels_batch in test_ds:
        #
        #     model_output = model(image_batch)
        #     label_output = tf.argmax(model_output, 1).numpy()[0]
        #     label = labels_batch.numpy()[0]
        #
        #     if label_output == label:
        #         classifications[label].append(1)
        #     else:
        #         classifications[label].append(0)
        #
        #     # count+=1
        #     # if count == 20:
        #     #     break
        #
        # print(np.mean(classifications[0]))
        # print(np.mean(classifications[1]))
        # print(np.mean(classifications[2]))


class EvaluateStandardized:
    def __init__(self, dataset, batch_size=1):
        self.dataset = dataset(batch_size)

    def evaluate(self):

        (train_ds, val_ds) = self.dataset.get_and_process()
        count = 0
        for image, labels_batch in train_ds:
            mean = tf.math.reduce_mean(image)
            std = tf.math.reduce_std(image)
            print(mean)
            print(std)
            print(image)
            break
            # count+=1
            # if count == 2:
            # break
