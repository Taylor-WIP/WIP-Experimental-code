import tensorflow as tf


class Train:
    def __init__(
        self,
        model,
        dataset,
        epochs: int = 1,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
    ):
        self.model = model(dataset(batch_size))
        self.epochs = epochs
        self.dataset = dataset(batch_size)
        self.learning_rate = learning_rate

    @property
    def save_name(self):
        return self.model.name + "_" + self.dataset.dataset_name

    def train(self):
        train_ds, val_ds = self.dataset.get_and_process()

        model = self.model.build()
        model.summary()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(self.learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        model.fit(
            train_ds,
            epochs=self.epochs,
            validation_data=val_ds,
        )
        model.save("../models/" + self.save_name + ".h5")

        return model
