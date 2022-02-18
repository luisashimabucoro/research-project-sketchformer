"""
Credits to the Tensorflow Tutorial "Transformer model for language understanding"
"""

import argparse
import json
import sys
sys.path.insert(0, '/store/lshimabucoro/projects/bumblebee/prep_data')
sys.path.insert(0, '/store/lshimabucoro/projects/bumblebee/utils')
import quickdraw_raw_to_tfrecords as dataloader
import stroke3_tokenizer as tokenizer

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

Usage:
    1- Criar função para uso do Transformer como ferramenta de predição
    2- Criar funções de plotagem
"""

BATCH_SIZE = 50
BUFFER_SIZE = 345*2500

# !LOAD AND TOKENIZE DATA 
"""
The data here is loaded from the TFRecord files, shuffled and them tokenized
using the Grid Tokenizer based on the Stroke3 format. After it's all been mapped
it is divided into batches n=BUFFER_SIZE/BATCH_SIZE batches of size BATCH_SIZE.
"""

def tokenize_single(sketch, label, resolution=100, max_len=100):
    grid_tok = tokenizer.GridTokenizer(resolution, max_len)
    enconded_sketch = grid_tok.encode(sketch)

    return tf.convert_to_tensor(enconded_sketch[1]), label

def create_batches(dataset):
    return(
        dataset
        .shuffle(BUFFER_SIZE)
        .map(lambda *x : tf.py_function(func=tokenize_single, inp=[*x], Tout=[tf.int64, tf.int64]), num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
    )
    # dataset = dataset.shuffle(BUFFER_SIZE)
    # dataset = dataset.map(lambda *x : tf.py_function(func=tokenize_single, inp=[*x], Tout=[tf.int64, tf.int64]))
    # batches = dataset.batch(BATCH_SIZE)

    # return batches

# !POSITIONAL ENCODING
"""
Positional encodings are used so as to give the model some information
regarding the position of the words in the sentence, since the embedding only
gives context about the word itself, and not its relationship with neighbouring words.

The formula is the one used in the original paper, which consists of sin/cos transformations.
"""
def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

# position of the word in the sentence and d_model = size of the word
def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

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

  def __init__(self, d_model, num_heads):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model

    assert d_model % self.num_heads == 0

    self.depth = d_model // self.num_heads

    self.wq = tf.keras.layers.Dense(d_model)
    self.wk = tf.keras.layers.Dense(d_model)
    self.wv = tf.keras.layers.Dense(d_model)

    self.dense = tf.keras.layers.Dense(d_model)

  def split_heads(self, x, batch_size):
    """Split the last dimension into (num_heads, depth).
    Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
    """
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])

  def call(self, v, k, q, mask):
    batch_size = tf.shape(q)[0]

    # Here we multiply out inputs by the Q, K and V weight matrices so as to obtain Q, K, and V themselves  
    q = self.wq(q)  # (batch_size, seq_len, d_model)
    k = self.wk(k)  # (batch_size, seq_len, d_model)
    v = self.wv(v)  # (batch_size, seq_len, d_model)

    q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
    k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
    v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

    # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
    # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
    scaled_attention, attention_weights = scaled_dot_product_attention(
        q, k, v, mask)

    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

    concat_attention = tf.reshape(scaled_attention,
                                  (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

    output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

    return output, attention_weights

# !FEED FOWARD BLOCK
"""
Here we have the feed-foward block responsible for processing the selected information 
given by the attention block.
"""

def point_wise_feed_forward_network(d_model, dff):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
  ])

# ? ============================ ENCODER AND DECODER LAYERS ============================
"""
The Encoder Layer consists a Multi-head Attention Block followed by a Feed Foward NN,
to which are applied Normalization Layers and Dropout in order to avoid overfitting and 
gradient issues.
"""

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.multi_head_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_foward_nn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):

        attention_output, _ = self.multi_head_attention(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attention_output = self.dropout1(attention_output, training=training)
        out1 = self.layernorm1(x + attention_output)  # (batch_size, input_seq_len, d_model)

        feed_foward_nn_output = self.feed_foward_nn(out1)  # (batch_size, input_seq_len, d_model)
        feed_foward_nn_output = self.dropout2(feed_foward_nn_output, training=training)
        out2 = self.layernorm2(out1 + feed_foward_nn_output)  # (batch_size, input_seq_len, d_model)

        return out2


"""
The Decoder Layer consists of two Multi-head Attention Blocks, self-attention (expected output) and 
encoder-decoder attention (input-output), followed by a Feed Foward NN, to which are applied 
Normalization Layers and Dropout in order to avoid overfitting and gradient issues.
"""
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training,
                look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2

# ? ============================ ENCODER AND DECODER BLOCKS ============================
"""
The Encoder Block is formed by a Embedding Layer, followed by the Positional Encoding
Layer and n Encoder Layers. It's a wrap-up of the entire Encoder.
"""

class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding,
                                                self.d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                            for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):

        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)

"""
The Decoder Block is formed by a Embedding Layer, followed by the Positional Encoding
Layer and n Decoder Layers. It's a wrap-up of the entire Decoder.
"""

class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
                    maximum_position_encoding, rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate)
                            for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    # Here we remove the padding mask
    # def call(self, x, enc_output, training,
    #             look_ahead_mask, padding_mask):
    def call(self, x, enc_output, training,
                look_ahead_mask):

        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                    look_ahead_mask, padding_mask)

            attention_weights[f'decoder_layer{i+1}_block1'] = block1
            attention_weights[f'decoder_layer{i+1}_block2'] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights

# ? ============================ TRANSFORMER ============================
"""
Now we assemble everything into the Transformer class, which is comprised of 
Encoder and Decoder Blocks followed by a final Dense Layer responsible for 
giving our model outputs.
"""

class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                    target_vocab_size, pe_input, pe_target, rate=0.1):
        super().__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, dff,
                                    input_vocab_size, pe_input, rate)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                                target_vocab_size, pe_target, rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inputs, training):
        # Keras models prefer if you pass all your inputs in the first argument
        inp, tar = inputs

        # enc_padding_mask, look_ahead_mask, dec_padding_mask = self.create_masks(inp, tar)
        enc_padding_mask, look_ahead_mask = self.create_masks(inp, tar)

        enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        # dec_output, attention_weights = self.decoder(
        #     tar, enc_output, training, look_ahead_mask, dec_padding_mask)
        dec_output, attention_weights = self.decoder(tar, enc_output, training, look_ahead_mask)

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output, attention_weights

    def create_masks(self, inp, tar):
        # Encoder padding mask
        enc_padding_mask = create_padding_mask(inp)

        # Used in the 2nd attention block in the decoder.
        # This padding mask is used to mask the encoder outputs.
        # dec_padding_mask = create_padding_mask(inp)

        # Used in the 1st attention block in the decoder.
        # It is used to pad and mask future tokens in the input received by
        # the decoder.
        look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = create_padding_mask(tar)
        look_ahead_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

        return enc_padding_mask, look_ahead_mask#, dec_padding_mask

# ? ============================ TRAINING SCHEDULE ============================

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

# ? ============================ LOSS AND METRICS ============================
# The loss function used is the Sparse Categorical Crossentropy
def loss_function(real, pred):
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

# Here we use the standard accuracy
def accuracy_function(real, pred):
    accuracies = tf.equal(real, tf.argmax(pred, axis=2))

    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)

    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)

# ? ============================ TRAINING ============================



def main():
    parser = argparse.ArgumentParser(description="Initial sketch transformer - L. Shimabucoro")
    parser.add_argument('--dataset-dir', type=str, default='/store/lshimabucoro/projects/bumblebee/scratch/datasets/quickdraw_raw_345')
    parser.add_argument('--class-index-dict', type=str, default='/store/lshimabucoro/projects/bumblebee/scratch/datasets/quickdraw_raw_345/meta.json')
    args = parser.parse_args()


    # with open(args.class_index_dict, 'r') as json_file:
    #     class_idx_dict = json.load(json_file)['idx_to_classes']
    #     print(class_idx_dict)
    
    dataset = dataloader.parse_dataset('/store/lshimabucoro/projects/bumblebee/scratch/datasets/quickdraw_raw_345/valid000.records')

    batches = create_batches(dataset)
    for batch in batches:
        print(batch)
        break



if __name__ == "__main__":
    main()