import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten, ReLU, Dropout, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from core.models import BaseModel
import utils
import numpy as np
# from metrics.classification import SceneGraphAppearenceEmbeddingObjAccuracy

def construct_model(input):
    # L1
    x = Conv2D(filters=64, kernel_size=(15, 15), strides=3)(input)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=2)(x)
    # L2
    x = Conv2D(filters=128, kernel_size=(5, 5), strides=1)(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=2)(x)
    # L3
    x = Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same')(x)
    x = ReLU()(x)
    # L4
    x = Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same')(x)
    x = ReLU()(x)
    # L5
    x = Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same')(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=2)(x)
    x = Flatten()(x)
    # L6
    x = Dense(512)(x)
    x = ReLU()(x)
    x = Dropout(0.5)(x)
    # L7
    x = Dense(512)(x)
    x = ReLU()(x)
    x = Dropout(0.5)(x)
    # L8
    x = Dense(91)(x)

    # returns a 345 x 1 x 1 tensor
    return x

# def top_3_accuracy(y_true, y_pred):
#     return top_k_categorical_accuracy(y_true, y_pred, k=3)

class Sketch_a_Net(BaseModel):
    name='sketch-a-net'
    quick_metrics = ['time', 'classification_loss', 'categorical_crossentropy', 'categorical_accuracy']
    slow_metrics = []

    # arrumar
    @classmethod
    def specific_default_hparams(cls):
        hparams = utils.hparams.HParams(
            optimizer="Adam",
            learning_rate=0.002,
            loss='categorical_crossentropy',
        )
        return hparams

    def build_model(self):

        # # layers
        # self.my_dense_layer = tf.keras.layers.Dense(128, activation='relu')
        # self.my_second_dense_layer = tf.keras.layers.Dense(128)
        # Input implements a Keras tensor
        input1 = Input(shape=(96, 96, 1))
        input2 = Input(shape=(96, 96, 1))
        input3 = Input(shape=(96, 96, 1))
        input4 = Input(shape=(96, 96, 1))
        input5 = Input(shape=(96, 96, 1))

        # outputs
        y1 = construct_model(input1)
        y2 = construct_model(input2)
        y3 = construct_model(input3)
        y4 = construct_model(input4)
        y5 = construct_model(input5)
        
        # merging models
        out = concatenate([y1, y2, y3, y4, y5])
        # print(out)

        # output layer
        predicitions = Dense(91, activation="softmax")(out)
        # out = Concatenate.average([y1, y2, y3, y4, y5])

        # Final model
        # self.model = Model(inputs=[input1, input2, input3, input4, input5], outputs=predicitions)
        self.model = Model(inputs=[input1, input2, input3, input4, input5], outputs=predicitions)

         # build a trainer (we need to build because this should be a tf.function)
        self.optimizer = tf.keras.optimizers.Adam(self.hps['learning_rate'])
        self.losses_manager.add_categorical_crossentropy('classification', weight=1., from_logits=True)

        self.model_trainer = self.build_model_trainer()
        print(self.model.summary())
    
    def build_model_trainer(self):    
        @tf.function(experimental_relax_shapes=True)
        def train(labels, images):
            # gradient tape gets the gradients
            with tf.GradientTape(persistent=True) as tape:
                # forward the image through our layers
                # flattened_images = tf.keras.layers.Flatten()(images)
                one_hot_labels = tf.one_hot(labels, self.dataset.n_classes)
                logits = self.model([images, images, images, images, images])
                # print("LOGITS: ", logits)

                # compute the loss (note how we are using the loss we've set up before)
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
    
    def compute_embedding_classification_predictions_on_validation_set(self):
        obj_ys, all_y = [], []
        batch_iterator = self.dataset.get_iterator('valid', self.hps['batch_size'], repeat=False, shuffle=None)
        for batch in batch_iterator:
            images, labels = batch
            # obj_encoding = self.forward(images, labels)
            # obj_ys.append(self.embedding_classifier(obj_encoding))
            obj_ys.append(self.model([images, images, images, images, images]))
            all_y.append(labels)

        obj_ys = np.concatenate(obj_ys, axis=0)
        all_y = np.concatenate(all_y, axis=0)
        return obj_ys, None, all_y, None

