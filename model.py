"""
################
### model.py ###
################

~ Will Bennett 12/07/2021

This store all of the models
"""
import tensorflow as tf
import numpy as np

def train_model(model, x, y , epochs, ckpt_name, ckpt_path='models'):
    ckpt_path = ckpt_path + '/' + ckpt_name +'.ckpt'
    # ckpt_dir = os.path.dirname(ckpt_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_path, save_weights_only=True, verbose=1)

    x = x.transpose()
    model.fit((x[0], x[1]), y,epochs=epochs, callbacks=[cp_callback])


def test_model(seq_len = 50, stock_vocab_size = 50, epoch= 100, embedidng_size=50):
    X1 = np.array(range(0, stock_vocab_size)).reshape((-1,1))
    X2 = np.array(range(0, seq_len)).reshape((-1,1))
    y = tf.random.uniform((seq_len, 5), minval=-0.01, maxval=0.01)
    
    embed = get_embedding_model(embedidng_size, stock_vocab_size, seq_len, 5)
    train_model(embed, X1, X2, y, epoch, 'test1')
    

def get_embedding_model(embedding_size, stock_vocab_size, time_vocab_size, output_size, rate=0.1):
    r"""
    args:
        embedding_size : int (the size of the embedding layer / output size),
        stock_vocab_size : int (the number of stocks in the index / list used),
        time_vocab_size : int (the number of days that can be used Max-min days)
    returns:
        tensorflow model

    this model is the original i used and was insiperied by the wiki book embedding tutorial
    - 
    """
    
    Stock = tf.keras.layers.Input(name = 'Stock_inp', shape=[1])
    Time = tf.keras.layers.Input(name = 'Time_inp', shape=[1])
    # Data = tf.keras.layers.Input(name = 'Data_inp', shape=[5])

    Stock_embedding = tf.keras.layers.Embedding(name = 'Stock_embedding', input_dim=stock_vocab_size, output_dim=embedding_size)(Stock)
    Time_embedding = tf.keras.layers.Embedding(name = 'Time_embedding', input_dim=time_vocab_size, output_dim=embedding_size)(Time)
    # Stock_embedding = tf.keras.layers.Dropout(rate)(Stock_embedding)
    # Time_embedding = tf.keras.layers.Dropout(rate)(Time_embedding)

    
    merged = tf.keras.layers.Dot(name = 'dot_product', normalize=True, axes=2)([Stock_embedding, Time_embedding])
    output = tf.keras.layers.Dense(output_size)(merged)

    model = tf.keras.Model(inputs = [Stock, Time], outputs = output)
    model.compile(optimizer = 'Adam', loss = 'mse')
    model.summary()
    return model

######## Tranformer models ##########


#Positional Encoding Functions
def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

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

#Masking functions
def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)

def create_masks(inp, tar):
    enc_padding_mask = create_padding_mask(inp)
    dec_padding_mask = create_padding_mask(inp)
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
    return enc_padding_mask, combined_mask, dec_padding_mask
    
#######  Layers #######

#Single dot product attention
def scaled_dot_product_attention(q, k, v, mask): 
    r"""Calculate the attention weights.
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

def sequence_attention():
    pass

#multihead attention
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
        r"""Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  
        k = self.wk(k)  
        v = self.wv(v)  

        q = self.split_heads(q, batch_size)  
        k = self.split_heads(k, batch_size) 
        v = self.split_heads(v, batch_size)  

      
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  

        concat_attention = tf.reshape(scaled_attention,
                                    (batch_size, -1, self.d_model)) 

        output = self.dense(concat_attention)  

        return output, attention_weights

#Feedforward layer
def point_wise_feed_forward_network(d_model, dff): 
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


class attention_layer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(attention_layer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, q, k, v, training, mask):
        attn_output, _ = self.mha(v,k,q, mask)
        attn_output = self.dropout(attn_output, training=training)
        out1 = self.layernorm(q + attn_output)
        return out1, _

class AttentionBlocks(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(AttentionBlocks, self).__init__()

        self.top_layer = attention_layer(d_model, num_heads, dff, rate)
        self.bottom_layer = attention_layer(d_model, num_heads, dff, rate)

    def call(self, top_q, top_k ,top_v, bottom_q, bottom_k, bottom_v, training, mask):
        top_x, top_attn = self.top_layer(top_q, top_k, top_v, training, mask)

        bottom_x, bottom_attn = self.bottom_layer(bottom_q, bottom_k, bottom_v, training, mask)

        return top_x, top_attn, bottom_x, bottom_attn

class look_up_layer(tf.keras.layers.Layer):
    def __init__(self, data_table):
        super(look_up_layer, self).__init__()
        self.data_table = data_table

    def call(self, stock, labels=None):
        stock_keys = self.data_table.keys()
        if labels == None:
            data = [self.data_table[label] for label in stock_keys if label != stock]
        else:
            data = [self.data_table[label] for label in labels if label != stock]
        data = np.array(data)
        return data, stock_keys

class curator(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, time_vocab_size, stock_vocab_size, output_size, data_table, pe_input, pe_target, rate=0.1):
        super(curator, self).__init__()

        self.stock_embedding = tf.keras.layers.Embedding(stock_vocab_size, d_model)
        self.time_embedding = tf.keras.layers.Embedding(stock_vocab_size, d_model)
        self.look_up = look_up_layer(data_table)
        self.output_layer = tf.keras.layers.Dense(output_size)

        self.time_vocab_size = time_vocab_size
        self.Nx = num_layers
        self.attn_blocks = [AttentionBlocks(d_model, num_heads, dff, rate) for i in range(0, self.Nx)]

    def call(self, stock_token, time_token, day_data, training, mask):
        stock_embed = self.stock_embedding(stock_token)
        time_embed = self.time_embedding(time_token)

        q = np.zeros((self.time_vocab_size, 5))
        start_index = len(day_data)
        q[:start_index] = day_data
        look_up_data, keys = self.look_up(stock_token)
        keys = self.stock_embedding(np.array(list(keys)))
        time_tokens_casted = tf.cast(time_embed, dtype=tf.float64)
        day_data = tf.concat([time_tokens_casted, day_data], axis=1)
        

        top_x, top_attn, bottom_x, bottom_attn = self.attn_blocks[0](day_data, keys, look_up_data, day_data, day_data, day_data, training, mask)
        
        for Nx in range(1,self.Nx):
            top_x, top_attn, bottom_x, bottom_attn = self.attn_blocks[Nx](bottom_x, keys, look_up_data, bottom_x, bottom_x, top_x, training, mask)

        merged = tf.keras.layers.Dot(axes=2)([top_x, bottom_x])
        output_layer = self.output(merged)
        return output_layer

def train_curator(model, x, y, epochs, filename):
    pass

def loss_function(real, pred, loss_object):
    # mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    return loss_
    # return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

def test_curator():
    seq_len = 50
    stock_tokens = np.array(range(0,seq_len))
    time_tokens = np.array(range(0, seq_len))
    X = tf.random.uniform((len(stock_tokens), seq_len, 5), dtype=tf.float64, minval=-5, maxval=5)
    y = tf.random.uniform((len(stock_tokens), 5), dtype=tf.float64, minval=-5, maxval=5)
    data_table = {stock_tokens[i] : X[i] for i in stock_tokens}

    model = curator(6, 512, 8, 2048, 50, 50, 5, data_table, 10000, 6000)
    loss_object = tf.keras.losses.MeanSquaredError()
    train_loss = tf.keras.metrics.Mean(name='train_loss')

    for i,stock in enumerate(stock_tokens):
        pred = model(stock.reshape((1)), time_tokens, X[i], False, None)
        print(pred)
        loss = loss_function(y, pred, loss_object)
        print(loss)


if __name__ == '__main__':
    test_model()