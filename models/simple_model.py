import tensorflow as tf

from core.models import BaseModel
import utils

class ExampleClassifier(BaseModel):
    name = 'simple_model'
    quick_metrics = ['time', 'classification_loss']
    slow_metrics = []

    @classmethod
    def specific_default_hparams(cls):
        hparams = utils.hparams.HParams(
            learning_rate=1e-4,
        )
        return hparams

    def build_model(self):
        # layers
        self.my_dense_layer = tf.keras.layers.Dense(128, activation='relu')
        self.my_second_dense_layer = tf.keras.layers.Dense(self.dataset.n_classes)

        # optimizer (notice how the hparam is used here)
        self.optimizer = tf.keras.optimizers.Adam(self.hps['learning_rate'])

        # prepare losses
        self.losses_manager.add_categorical_crossentropy('classification', weight=1., from_logits=True)

        # build a trainer (we need to build because this should be a tf.function)
        self.model_trainer = self.build_model_trainer()

    def build_model_trainer(self):
        @tf.function(experimental_relax_shapes=True)
        def train(labels, images):
            # gradient tape gets the gradients
            with tf.GradientTape(persistent=True) as tape:
                # forward the image through our layers
                flattened_images = tf.keras.layers.Flatten()(images)
                logits = self.my_second_dense_layer(self.my_dense_layer(flattened_images))

                # compute the loss (note how we are using the loss we've set up before)
                one_hot_labels = tf.one_hot(labels, self.dataset.n_classes)
                loss = self.losses_manager.compute_loss('classification', one_hot_labels, logits)

            # optimize the weights by using the gradients
            self.optimizer.apply_gradients(zip(tape.gradient(loss, self.trainable_variables), self.trainable_variables))

            # return a dictionary with our quick metrics,
            # we'll keep it simple and return the classification loss
            return {'classification_loss': loss}
        return train

    def train_on_batch(self, batch):
        images, labels = next(batch)
        return self.model_trainer(labels, images)

    def train_iterator(self):
        return iter(self.dataset.get_iterator('train', self.hps['batch_size'], repeat=True))