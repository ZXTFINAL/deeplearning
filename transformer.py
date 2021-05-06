import tensorflow as tf
from tensorflow import keras
import numpy as np


def position_encoding(m_len, d_model):
    PE = np.zeros((m_len, d_model))
    for row in range(m_len):
        for col in range(d_model):
            if col % 2 == 0:
                PE[row, col] = np.sin(row/(10000**(col/d_model)))
            else:
                PE[row, col] = np.cos(row/(10000**((col-1)/d_model)))
    PE = tf.contant(PE, dtype=tf.float32)
    return PE


class muti_head_self_attention(keras.Model):
    def __init__(self, d_model, num_heads):
        super(muti_head_self_attention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_size = d_model//num_heads
        self.q = keras.layers.Dense(units=d_model, name="q")
        self.k = keras.layers.Dense(units=d_model, name="k")
        self.v = keras.layers.Dense(units=d_model, name="v")
        self.dense = keras.layers.Dense(d_model)

    def call(self, q, k, v, m):
        batch_size = tf.shape(q)[0]
        q = self.q(q)
        k = self.k(k)
        v = self.v(v)

        def _split_heads(x):
            x = tf.reshape(
                x, shape=[batch_size, -1, self.num_heads, self.head_size])
            return tf.transpose(x, perm=[0, 2, 1, 3])
        q = _split_heads(q)
        k = _split_heads(k)
        v = _split_heads(v)
        q_k = tf.multiply(q, tf.transpose(k, perm=[0, 1, 3, 2]))
        dk = tf.cast(q.shape[-1], tf.float32)
        q_k_score = q_k/tf.math.sqrt(dk)
        if m:
            q_k_score += (1-m)*-1e10
        a = tf.nn.softmax(q_k_score)
        context = tf.matmul(a, v)
        context = tf.transpose(context, perm=[0, 2, 1, 3])
        context = context.reshape(context, (batch_size, -1, self.d_model))
        output = self.dense(context)
        print(output)


class feed_forward_network(keras.Model):
    def __init__(self, dff_size, d_model):
        self.dense_a = keras.layers.Dense(units=dff_size, activation="relu")
        self.dense_b = keras.layers.Dense(units=d_model)

    def call(self, x):
        x = self.dense_b(self.dense_a(x))
        return x


class encoder_layer(keras.layers.layer):
    def __init__(self, d_model, num_heads, dff_size, dropout_rate=0.1):
        super(encoder_layer, self).__init__()

        self.attention_layer = muti_head_self_attention(d_model, num_heads)
        self.ffn_layer = feed_forward_network(dff_size, d_model)
        self.layernorm_a = keras.layers.LayerNormalization(epsilon=1e-5)
        self.layernorm_b = keras.layers.LayerNormalization(epsilon=1e-5)
        self.dropout_a = keras.layers.Dropout(dropout_rate)
        self.dropout_b = keras.layers.Dropout(dropout_rate)

    def call(self, x, training, mask):
        att_output = self.attention_layer(x, x, x, mask)
        att_do_output = self.dropout_a(att_output)
        att_do_ln_output = self.layernorm_a(x+att_do_output)
        ffn_output = self.ffn_layer(att_do_ln_output)
        ffn_do_output = self.dropout_b(ffn_output)
        ffn_do_ln_output = self.layernorm_b(att_do_ln_output+ffn_do_output)
        return ffn_do_ln_output


class encoder(keras.Model):
    def __init__(self, encoder_N, vocab_size, d_model, num_heads, dff_size, m_len):
        self.embedding = keras.layers.Embedding(vocab_size, d_model)
        self.position_encoding = position_encoding(m_len, d_model)
        self.encoder_layer_list = [encoder_layer(
            d_model, num_heads, dff_size) for _ in range(encoder_N)]

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)+self.position_encoding[:, :seq_len, :]

        for encoderlayer in range(self.encoder_layer_list):
            x = encoderlayer(x, training, mask)
        return x


class decoder(keras.Model):
    def __init__():
        self.embedding = keras.layers.Embedding(vovab_size, d_model)
        self.position_encoding = position_encoding(m_len, d_model)

    def call():
        pass


class transformer(keras.Model):
    def __init__():
        pass

    def call():
        pass
