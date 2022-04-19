"""
Credits to the Tensorflow Tutorial "Transformer model for language understanding"
"""

"""
Validation loss: 1.18
Validation reconstruction accuracy: 0.9619
Validation classification accuracy: 0.7273
"""

import os
import argparse
import sys
import time

sys.path.insert(0, '/store/lshimabucoro/projects/bumblebee/prep_data')
sys.path.insert(0, '/store/lshimabucoro/projects/bumblebee/utils')
sys.path.insert(0, '/store/lshimabucoro/projects/bumblebee/models')
import quickdraw_raw_to_tfrecords as dataloader
import stroke3_tokenizer as tokenizer
import feature_sort_pooling as FSPool

import numpy as np
import tensorflow as tf


"""
Steps to build the Transformer:

Auxiliary functions: #!DONE
    1- Create function tokenize inputs
    2- Create Embedding Layer
    3- Create Positional Encoding Functions
    4- Create attention functions 

Classes: #!DONE
    1- Create Multi-Head Attention Class
    2- Create Encoder Layer class
    3- Create Decoder Layer class
    4- Create Encoder Block class (n encoder layers)
    5- Create Decoder Block class (n decoder layers)
    6- Put everything together and build the Transformer class

Training: #!DONE
    1- Create loss and accuaracy functions
    2- Define training schedule

Usage:  #!DONE
    1- Criar função para uso do Transformer como ferramenta de predição
    2- Criar funções de plotagem
"""
EPOCHS = 5
BATCH_SIZE = 64
TRAIN_BUFFER_SIZE = 7000*345*10
TEST_BUFFER_SIZE = 345*2500

def setup_gpu(gpu_ids):
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            sel_gpus = [gpus[g] for g in gpu_ids]
            tf.config.set_visible_devices(sel_gpus, 'GPU')
            for g in sel_gpus:
                tf.config.experimental.set_memory_growth(g, True)
        except RuntimeError as e:
            # visible devices must be set before GPUs have been initialized
            print(e)

# !LOAD AND TOKENIZE DATA 
"""
The data here is loaded from the TFRecord files, shuffled and them tokenized
using the Grid Tokenizer based on the Stroke3 format. After it's all been mapped
it is divided into batches n=BUFFER_SIZE/BATCH_SIZE batches of size BATCH_SIZE.
"""
def tokenize_single_tf(sketch, label, resolution=100, max_len=512):
    grid_tok = tokenizer.GridTokenizer(resolution, max_len)
    _, enconded_sketch = grid_tok.encode_tf(tf.cast(sketch, dtype=tf.int64), cls=False)

    return enconded_sketch, label

def tokenize_single(sketch, label, resolution=100, max_len=512):
    grid_tok = tokenizer.GridTokenizer(resolution, max_len)
    _, enconded_sketch = grid_tok.encode(sketch)

    return tf.convert_to_tensor(enconded_sketch), label

# limits the size of the sketch to max_seq_len by adding padding or removing elements
def pad_raw_single(sketch, label, max_seq_len=300):
    exceding_points = tf.shape(sketch)[0] - max_seq_len
    if exceding_points < 0:
        padding = tf.zeros(shape=(tf.abs(exceding_points), 3))
        sketch = tf.concat([sketch, padding], axis=0)
    if exceding_points > 0:
        sketch = sketch[0:max_seq_len,:]

    return sketch, label

def join_dataset_files(dataset_dir):
    buffer = TRAIN_BUFFER_SIZE
    datasets = {}

    for split in ['train', 'valid', 'test']:
        tfrecords_pattern = os.path.join(dataset_dir, "{}*.records".format(split))
        files = tf.io.matching_files(tfrecords_pattern)

        dataset = dataloader.parse_dataset(files)
        if split != 'train':
            buffer =  TEST_BUFFER_SIZE

        # dataset = dataset.shuffle(buffer)
        # dataset = dataset.map(tokenize_single_tf)
        # batches = dataset.batch(BATCH_SIZE)
        dataset = dataset.map(pad_raw_single)
        batches = dataset.batch(BATCH_SIZE)
        # batches = dataset.padded_batch(BATCH_SIZE)

        datasets[split] = batches

    return datasets

# !POSITIONAL ENCODING
"""
Positional encodings are used so as to give the model some information
regarding the position of the words in the sentence, since the embedding only
gives context about the word itself, and not its relationship with neighbouring words.

The formula is the one used in the original paper, which consists of sin/cos transformations.
"""
def get_angles(pos, i, point_size):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(point_size))
    return pos * angle_rates

# position of the word in the sentence and point_size = size of the word
def positional_encoding(position, point_size):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(point_size)[np.newaxis, :],
                            point_size)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)

# !MASKS MATRICES
"""
Masking is so so we "hide" irrelevant entries as input.

The Padding Mask does this by setting a 1 where there is a padding value
in the input, which signals to the model that it should ignore that entry.

The Look-ahead Mask is used so the models does not "take a peak" at the future
tokens it aims to predict. This essentially means that it only takes into account
the n-1 previous tokens in order to predict the nth one. This is done by creating
a upper triangular matrix where the 1s indicate the hidden characters.
"""
# function to create masking matrices so the model ignores the padding values
def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

# create look ahead mask, which hides future tokens in a sequence
def create_look_ahead_mask(size):
    # lower triangular part (-1, 0) is set to zero
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)

# !ATTENTION CALCULATION
"""
The attention calculates how each component relates to each other by 
dividing them into the Queries, Keys and Values matrix and calculating
their similarities.

The result is a series of vectors and the attention weights, which are
adjusted in each iteration.
"""

# Calculates masked and unmasked (standard) attention
def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
    output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights

# ? ============================ AUXILIARY LAYERS ============================

# !MULTIHEAD ATTENTION 
"""
The Multi-head Attention block is used so we have not only one, but multiple
attention layers all in one block, which improves the overall performance of the 
attention block, since each attention head can focus on detecting distinct patterns
from each other, which increases the representation subspace.
"""

class MultiHeadAttention(tf.keras.layers.Layer):

  def __init__(self, point_size, num_heads):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.point_size = point_size

    assert point_size % self.num_heads == 0

    self.depth = point_size // self.num_heads

    self.wq = tf.keras.layers.Dense(point_size)
    self.wk = tf.keras.layers.Dense(point_size)
    self.wv = tf.keras.layers.Dense(point_size)

    self.dense = tf.keras.layers.Dense(point_size)

  def split_heads(self, x, batch_size):
    """
    Split the last dimension into (num_heads, depth).
    Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
    """
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])

  def call(self, v, k, q, mask):
    batch_size = tf.shape(q)[0]

    # Here we multiply out inputs by the Q, K and V weight matrices so as to obtain Q, K, and V themselves  
    q = self.wq(q)  # (batch_size, seq_len, point_size)
    k = self.wk(k)  # (batch_size, seq_len, point_size)
    v = self.wv(v)  # (batch_size, seq_len, point_size)

    q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
    k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
    v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

    # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
    # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
    scaled_attention, attention_weights = scaled_dot_product_attention(
        q, k, v, mask)

    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

    concat_attention = tf.reshape(scaled_attention,
                                  (batch_size, -1, self.point_size))  # (batch_size, seq_len_q, point_size)

    output = self.dense(concat_attention)  # (batch_size, seq_len_q, point_size)

    return output, attention_weights

# !FEED FOWARD BLOCK
"""
Here we have the feed-foward block responsible for processing the selected information 
given by the attention block.
"""

def point_wise_feed_forward_network(point_size, n_points):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(n_points, activation='relu'),  # (batch_size, seq_len, n_points)
      tf.keras.layers.Dense(point_size)  # (batch_size, seq_len, point_size)
  ])

# ? ============================ ENCODER AND DECODER LAYERS ============================
"""
The Encoder Layer consists a Multi-head Attention Block followed by a Feed Foward NN,
to which are applied Normalization Layers and Dropout in order to avoid overfitting and 
gradient issues.
"""

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, point_size, num_heads, n_points, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.multi_head_attention = MultiHeadAttention(point_size, num_heads)
        self.feed_foward_nn = point_wise_feed_forward_network(point_size, n_points)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):

        attention_output, _ = self.multi_head_attention(x, x, x, mask)  # (batch_size, input_seq_len, point_size)
        attention_output = self.dropout1(attention_output, training=training)
        out1 = self.layernorm1(x + attention_output)  # (batch_size, input_seq_len, point_size)

        feed_foward_nn_output = self.feed_foward_nn(out1)  # (batch_size, input_seq_len, point_size)
        feed_foward_nn_output = self.dropout2(feed_foward_nn_output, training=training)
        out2 = self.layernorm2(out1 + feed_foward_nn_output)  # (batch_size, input_seq_len, point_size)
        # (64, 512, 128)
        return out2


"""
The Decoder Layer consists of two Multi-head Attention Blocks, self-attention (expected output) and 
encoder-decoder attention (input-output), followed by a Feed Foward NN, to which are applied 
Normalization Layers and Dropout in order to avoid overfitting and gradient issues.
"""
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, point_size, num_heads, n_points, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(point_size, num_heads)
        self.mha2 = MultiHeadAttention(point_size, num_heads)

        self.ffn = point_wise_feed_forward_network(point_size, n_points)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    # def call(self, x, enc_output, training,
    #             look_ahead_mask, padding_mask):
    def call(self, x, enc_output, training,
                look_ahead_mask):
        # enc_output.shape == (batch_size, input_seq_len, point_size)

        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, point_size)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        # attn2, attn_weights_block2 = self.mha2(
        #     enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, point_size)
        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1, None)  # (batch_size, target_seq_len, point_size)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, point_size)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, point_size)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, point_size)

        return out3, attn_weights_block1, attn_weights_block2

# ? ============================ ENCODER AND DECODER BLOCKS ============================
"""
The Encoder Block is formed by a Embedding Layer, followed by the Positional Encoding
Layer and n Encoder Layers. It's a wrap-up of the entire Encoder.
"""
# We want the number of zero elements from the padding mask so we
# count the number of zero ones and subtract from the total number
# of elements
def create_sketch_size_vec(mask, seq_len):
    count = 512 - tf.math.reduce_sum(tf.math.reduce_sum(tf.math.count_nonzero(mask, axis=1), axis=1), axis=1)

    return count

class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, point_size, num_heads, n_points, input_vocab_size,
                maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()

        self.point_size = point_size
        self.num_layers = num_layers

        # self.embedding = tf.keras.layers.Embedding(input_vocab_size, point_size)
        self.input_dense_layer = tf.keras.layers.Dense(n_points)
        self.pos_encoding = positional_encoding(maximum_position_encoding,
                                                self.point_size)

        self.enc_layers = [EncoderLayer(point_size, num_heads, n_points, rate)
                            for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

        self.fsp_layer = FSPool.FSEncoder(point_size, point_size, n_points)

        self.classifier_dense_layer = tf.keras.layers.Dense(345)

    def call(self, x, label, training, mask):

        seq_len = tf.shape(x)[1]

        # x (batch_size, input_seq_len)
        # adding embedding and position encoding.
        # x = self.embedding(x)  # (batch_size, input_seq_len, point_size)
        tf.print("First Encoder")
        tf.print(x.shape)
        x = self.input_dense_layer(x)
        tf.print(x.shape)
        x *= tf.math.sqrt(tf.cast(self.point_size, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        # passing though the n encoder layers created
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)   # (batch_size, input_seq_len, point_size)
        
        # (batch_size, n_classes)
        sizes = create_sketch_size_vec(mask, 512)   # rank 1 tensor containing the original sketch sizes
        class_input = self.fsp_layer(x, sizes)

        classification_logits = self.classifier_dense_layer(class_input) 
        class_pred = tf.nn.softmax(classification_logits)    # (batch_size, n_classes)

        return x, class_pred

"""
The Decoder Block is formed by a Embedding Layer, followed by the Positional Encoding
Layer and n Decoder Layers. It's a wrap-up of the entire Decoder.
"""

class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, point_size, num_heads, n_points, target_vocab_size,
                    maximum_position_encoding, rate=0.1):
        super(Decoder, self).__init__()

        self.point_size = point_size
        self.num_layers = num_layers

        # self.embedding = tf.keras.layers.Embedding(target_vocab_size, point_size)
        self.input_dense_layer = tf.keras.layers.Dense(n_points)
        self.pos_encoding = positional_encoding(maximum_position_encoding, point_size)

        self.dec_layers = [DecoderLayer(point_size, num_heads, n_points, rate)
                            for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    # Here we remove the padding mask
    # def call(self, x, enc_output, training,
    #             look_ahead_mask, padding_mask):
    def call(self, x, enc_output, training,
                look_ahead_mask):

        seq_len = tf.shape(x)[1]
        attention_weights = {}

        # x = self.embedding(x)  # (batch_size, target_seq_len, point_size)
        x = self.input_dense_layer(x)
        x *= tf.math.sqrt(tf.cast(self.point_size, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                    look_ahead_mask)

            attention_weights[f'decoder_layer{i+1}_block1'] = block1
            attention_weights[f'decoder_layer{i+1}_block2'] = block2

        # x.shape == (batch_size, target_seq_len, point_size)
        return x, attention_weights

# ? ============================ TRANSFORMER ============================
"""
Now we assemble everything into the Transformer class, which is comprised of 
Encoder and Decoder Blocks followed by a final Dense Layer responsible for 
giving our model outputs.
"""

class Transformer(tf.keras.Model):
    def __init__(self, num_layers, point_size, num_heads, n_points, input_vocab_size,
                    target_vocab_size, pe_input, pe_target, rate=0.1):
        super().__init__()
        self.encoder = Encoder(num_layers, point_size, num_heads, n_points,
                                    input_vocab_size, pe_input, rate)

        self.decoder = Decoder(num_layers, point_size, num_heads, n_points,
                                target_vocab_size, pe_target, rate)

        # one value for x, one for y and 2 for up-down pen
        self.final_layer = tf.keras.layers.Dense(4)

    def call(self, inputs, training):
        # Keras models prefer if you pass all your inputs in the first argument
        inp, tar, label = inputs

        # enc_padding_mask, look_ahead_mask, dec_padding_mask = self.create_masks(inp, tar)
        enc_padding_mask, look_ahead_mask = self.create_masks(inp, tar)

        enc_output, class_pred = self.encoder(inp, label, training, enc_padding_mask)  # (batch_size, inp_seq_len, point_size)

        # dec_output.shape == (batch_size, tar_seq_len, point_size)
        # dec_output, attention_weights = self.decoder(
        #     tar, enc_output, training, look_ahead_mask, dec_padding_mask)
        dec_output, attention_weights = self.decoder(tar, enc_output, training, look_ahead_mask)

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)
        up_down_pred = tf.nn.softmax(final_output[-2:])     # last two outputs are used to predict whether the pen is up or down

        return final_output[0:2], attention_weights, class_pred, up_down_pred

    def create_masks(self, inp, tar):
        # Encoder padding mask
        enc_padding_mask = create_padding_mask(inp)
        tf.print("Encoder padding mask:")
        tf.print(enc_padding_mask)

        # Used in the 2nd attention block in the decoder.
        # This padding mask is used to mask the encoder outputs.
        # dec_padding_mask = create_padding_mask(inp)

        # Used in the 1st attention block in the decoder.
        # It is used to pad and mask future tokens in the input received by
        # the decoder.
        look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = create_padding_mask(tar)
        tf.print("Look-ahead mask:")
        tf.print(look_ahead_mask)
        tf.print("Decoder padding mask:")
        tf.print(dec_target_padding_mask)
        look_ahead_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

        return enc_padding_mask, look_ahead_mask#, dec_padding_mask

# ? ============================ TRAINING SCHEDULE ============================

# The custom schedule defines the learning rate
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, point_size, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.point_size = point_size
        self.point_size = tf.cast(self.point_size, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.point_size) * tf.math.minimum(arg1, arg2)

# ? ============================ LOSS AND METRICS ============================
# The loss function used is the Sparse Categorical Crossentropy
def loss_function(real, pred, real_class, pred_class):
    x_y_pred, up_down_pred = pred
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real[:2], x_y_pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    original_task_loss = tf.reduce_sum(loss_)/tf.reduce_sum(mask)
    class_loss = class_loss_function(real_class, pred_class, 345)   # regular classification
    pen_loss = class_loss_function(real[-1], up_down_pred, 2)   	    # pen up-down classification

    return original_task_loss + class_loss + pen_loss

def class_loss_function(label_class, pred_class, n_classes):
    cross_entropy_loss = tf.keras.losses.CategoricalCrossentropy()
    label_one_hot = tf.one_hot(label_class, depth=n_classes)    # (batch_size, point_size)
    class_loss = cross_entropy_loss(label_one_hot, pred_class)

    return class_loss


# Here we use the standard accuracy
def accuracy_function(real, pred, classification=False):
    axis = 2
    if classification:
        axis = 1
    accuracies = tf.equal(real, tf.argmax(pred, axis=axis))

    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)

    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)

    return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)

# ? ============================ TRAINING ============================

def checkpoint_manager(transformer, optimizer, checkpoint_path, restore=True):
    ckpt = tf.train.Checkpoint(transformer=transformer,
                            optimizer=optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    # if a checkpoint exists, restore the latest checkpoint.
    if restore and ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('\nLatest checkpoint restored!!\n')
    
    return ckpt_manager

class TrainingTransformer():
    def __init__(self, transformer, optimizer, train_loss, train_accuracy, class_accuracy):
        self.transformer = transformer
        self.optimizer = optimizer
        self.train_loss = train_loss
        self.train_accuracy = train_accuracy
        self.class_accuracy = class_accuracy
    
    def __call__(self):
        return self.transformer, self.optimizer, self.train_loss, self.train_accuracy, self.class_accuracy

@tf.function
def train_step(inp, tar, label, training_obj):
    """
    inp & tar : sketches in stroke3 format
    label : sketch class
    training_obj : object containing fundamental components of the training schedule
    """
    transformer, optimizer, train_loss, train_accuracy, class_accuracy = training_obj()
    tar_real = tar[:, :]    # all points
    tar_inp = tar[:, :-1]   # all points except the last one

    with tf.GradientTape() as tape:
        x_y_pred, _, class_pred, up_down_pred = transformer([inp, tar_inp, label],
                                    training = True)
        loss = loss_function(tar_real, (x_y_pred, up_down_pred), label, class_pred)

    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    # overall loss, which accounts for point prediction + classification
    train_loss(loss)
    # next point prediction (accuracy)
    train_accuracy(accuracy_function(tar_real[:2], x_y_pred) + accuracy_function(tar_real[-1], up_down_pred, classification=True))
    # classification accuracy
    class_accuracy(accuracy_function(label, class_pred, classification=True))

def training_schedule(train_batches, training_obj, ckpt_manager):
    _, _, train_loss, train_accuracy, class_accuracy = training_obj()
    for epoch in range(EPOCHS):
        start = time.time()

        train_loss.reset_states()
        train_accuracy.reset_states()
        class_accuracy.reset_states()

        for (batch, (img, label)) in enumerate(train_batches):
            train_step(img, img, label, training_obj)

            if batch % 50 == 0:
                print(f'Epoch {epoch + 1} Batch {batch} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f} Classification Accuracy {class_accuracy.result():.4f}')
                
                if batch % 5000 == 0 and batch != 0:
                    ckpt_save_path = ckpt_manager.save()
                    print(f'\nSaving checkpoint for batch {batch} at {ckpt_save_path}\n')
                    print(f'Time taken for {batch} batches: {time.time() - start:.2f} secs\n')
            

        ckpt_save_path = ckpt_manager.save()
        print(f'Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}')

        print(f'Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f} Classification Accuracy {class_accuracy.result():.4f}')

        print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')

# d_model = size of the word (depth of word)
# dff = number of words
# pe_input/pe_target are related to the positional encoding
def train_model(dataset, num_layers=6, point_size=128, n_points=512, num_heads=8, dropout_rate=0.1, vocab_size=10004):
    transformer = Transformer(
    num_layers=num_layers,
    point_size=point_size,
    num_heads=num_heads,
    n_points=n_points,
    input_vocab_size=vocab_size,
    target_vocab_size=vocab_size,
    pe_input=1000,
    pe_target=1000,
    rate=dropout_rate)

    learning_rate = CustomSchedule(point_size)
    # learning_rate = 1e-4
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')
    class_accuracy = tf.keras.metrics.Mean(name='classification_accuracy')
    checkpoint_path = "/store/lshimabucoro/projects/bumblebee/scratch/checkpoints/train/sketchformer-fsp-lr-schedule"


    ckp_mng = checkpoint_manager(transformer, optimizer, checkpoint_path, restore=False)
    training_obj = TrainingTransformer(transformer, optimizer, train_loss, train_accuracy, class_accuracy)

    training_schedule(dataset, training_obj, ckp_mng)

def validate_single_batch(input, target, label, transformer, valid_functions):
    valid_loss, valid_accuracy, valid_class_accuracy = valid_functions
    tar_inp = target[:, :-1]
    tar_real = target[:, 1:]

    predictions, _, class_pred = transformer([input, tar_inp, label], training=False)
    valid_loss.update_state(loss_function(tar_real, predictions, label, class_pred))
    valid_accuracy.update_state(accuracy_function(tar_real, predictions))
    valid_class_accuracy.update_state(accuracy_function(label, class_pred, classification=True))
    # return loss, accuracy, class_accuracy

def validate_model(valid_batches, transformer, optimizer, checkpoint_path):
    ckpt_mng = checkpoint_manager(transformer, optimizer, checkpoint_path, restore=True)
    valid_loss = tf.keras.metrics.Mean(name='validation_loss')
    valid_accuracy = tf.keras.metrics.Mean(name='validation_accuracy')
    valid_class_accuracy = tf.keras.metrics.Mean(name='classification_accuracy')


    valid_loss.reset_states()
    valid_accuracy.reset_states()
    valid_class_accuracy.reset_states()

    for (batch, (img, label)) in enumerate(valid_batches):
        validate_single_batch(img, img, label, transformer, (valid_loss, valid_accuracy, valid_class_accuracy))
        if batch % 50 == 0:
            print(f"Batch {batch} - loss {valid_loss.result():.4f} accuracy {valid_accuracy.result():.4f} classification accuracy {valid_class_accuracy.result():.4f}")

class Translator(tf.Module):
    def __init__(self, sketch_tokenizer, transformer):
        self.sketch_tokenizer = sketch_tokenizer
        self.transformer = transformer

    def __call__(self, sketch, max_length=512):
        assert isinstance(sketch, tf.Tensor)
        if len(sketch.shape) == 0:
            sketch = sketch[tf.newaxis]

        # stroke3
        scale, sketch = self.sketch_tokenizer.encode_tf(sketch)

        encoder_input = sketch[tf.newaxis, :]
        print(encoder_input)

        start_end = tf.constant(value=[10002, 10003])
        start = start_end[0][tf.newaxis]
        end = start_end[1][tf.newaxis]

        output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
        output_array = output_array.write(index=0, value=tf.cast(start, dtype=tf.int64))
        print(output_array)

        for i in tf.range(max_length):
            output = tf.transpose(output_array.stack())
            predictions, _ = self.transformer([encoder_input, output], training=False)

            # select the last token from the seq_len dimension
            predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

            predicted_id = tf.argmax(predictions, axis=-1)

            # concatentate the predicted_id to the output which is given to the decoder
            # as its input.
            output_array = output_array.write(i+1, tf.cast(predicted_id[0], dtype=tf.int64))
            if tf.cast(predicted_id, dtype=tf.int64) == tf.cast(end, dtype=tf.int64):
                break

        output = tf.transpose(output_array.stack())
        print(tf.squeeze(output))
        stroke3_sketch = self.sketch_tokenizer.decode_single(tf.squeeze(output), scale)  # shape: ()
        print(stroke3_sketch)
        tokenizer.stroke3_to_image(stroke3_sketch, scale, file_name="reconstruction.png")

        # tokens = sketch_tokenizer.en.lookup(output)[0]

        # `tf.function` prevents us from using the attention_weights that were
        # calculated on the last iteration of the loop. So recalculate them outside
        # the loop.
        _, attention_weights = self.transformer([encoder_input, output[:,:-1]], training=False)

        return stroke3_sketch, attention_weights

# def checkpoint_manager(transformer, optimizer):
#     checkpoint_path = "/store/lshimabucoro/projects/bumblebee/scratch/checkpoints/train/initial-sketch-transformer"

#     ckpt = tf.train.Checkpoint(transformer=transformer,
#                             optimizer=optimizer)

#     ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

#     # if a checkpoint exists, restore the latest checkpoint.
#     if ckpt_manager.latest_checkpoint:
#         ckpt.restore(ckpt_manager.latest_checkpoint)
#         print('\nLatest checkpoint restored!!\n')
    
#     return ckpt_manager

def main():
    parser = argparse.ArgumentParser(description="Initial sketch transformer - L. Shimabucoro")
    parser.add_argument('--dataset-dir', type=str, default='/store/lshimabucoro/projects/bumblebee/scratch/datasets/quickdraw_raw_345')
    parser.add_argument('--class-index-dict', type=str, default='/store/lshimabucoro/projects/bumblebee/scratch/datasets/quickdraw_raw_345/meta.json')
    args = parser.parse_args()

    setup_gpu([1])
                                        
    num_layers = 6
    point_size = 128
    n_points = 512
    num_heads = 8
    dropout_rate = 0.1

    transformer = Transformer(
    num_layers=num_layers,
    point_size=point_size,
    num_heads=num_heads,
    n_points=n_points,
    input_vocab_size=10004,
    target_vocab_size=10004,
    pe_input=1000,
    pe_target=1000,
    rate=dropout_rate)

    learning_rate = CustomSchedule(point_size)
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    # train_loss = tf.keras.metrics.Mean(name='train_loss')
    # train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

    # print(transformer)
    # ckp_mng = checkpoint_manager(transformer, optimizer, restore=True)
    # training_obj = TrainingTransformer(transformer, optimizer, train_loss, train_accuracy)

    # training_schedule(datasets['train'], training_obj, ckp_mng)

    datasets = join_dataset_files(args.dataset_dir)
    # i = 0
    # for batch in datasets['train']:
    #     tf.print(batch[0].shape)
    #     if i == 5:
    #         break
    #     i += 1



    # checkpoint_path = "/store/lshimabucoro/projects/bumblebee/scratch/checkpoints/train/sketchformer-fsp-lr-schedule"
    """
    num_layers = 6
    n_points = max_seq_len (300)
    point_size = 3 (x,y,pen_up_down)
    vocab_size = not relevant?
    num_heads = 8
    dropout_rate = 0.1
    """
    train_model(datasets['train'], num_layers=6, point_size=128, n_points=300, num_heads=8, dropout_rate=0.1, vocab_size=10004)
    # translator = Translator(tokenizer.GridTokenizer(100, 512), transformer)
    # validate_model(datasets['valid'], transformer, optimizer, checkpoint_path)


if __name__ == "__main__":
    main()