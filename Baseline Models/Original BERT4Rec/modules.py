# modules.py  (TF1, Python 3) 
import tensorflow as tf


def embedding(inputs, vocab_size, num_units, zero_pad=True, scale=True,
              l2_reg=0.0, scope="embedding", with_t=False, reuse=None):
    """Lookup embedding; trả về (outputs, table) như bản gốc."""
    with tf.variable_scope(scope, reuse=reuse):
        regularizer = tf.contrib.layers.l2_regularizer(l2_reg) if l2_reg > 0 else None
        lookup_table = tf.get_variable(
            'lookup_table',
            shape=[vocab_size, num_units],
            dtype=tf.float32,
            regularizer=regularizer,
            initializer=tf.random_normal_initializer(stddev=0.01)
        )
        if zero_pad:
            lookup_table = tf.concat(
                (tf.zeros([1, num_units], dtype=tf.float32),
                 lookup_table[1:, :]), axis=0)

        outputs = tf.nn.embedding_lookup(lookup_table, inputs)
        if scale:
            outputs *= num_units ** 0.5
        return outputs, lookup_table


def normalize(inputs, epsilon=1e-8, scope="ln", reuse=None):
    """Layer Normalization."""
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        params_shape = inputs.get_shape()[-1:]
        beta  = tf.get_variable('beta',  params_shape, initializer=tf.zeros_initializer())
        gamma = tf.get_variable('gamma', params_shape, initializer=tf.ones_initializer())
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        normalized = (inputs - mean) / tf.sqrt(variance + epsilon)
        outputs = gamma * normalized + beta
        return outputs


def multihead_attention(queries, keys,
                        num_units=None, num_heads=1,
                        dropout_rate=0.0, is_training=False,
                        causality=False, scope="multihead_attention", reuse=None):
    """Scaled dot-product multi-head attention (gần bản gốc)."""
    if num_units is None:
        num_units = queries.get_shape().as_list()[-1]

    with tf.variable_scope(scope, reuse=reuse):
        Q = tf.layers.dense(queries, num_units, activation=None, name='Q')
        K = tf.layers.dense(keys,    num_units, activation=None, name='K')
        V = tf.layers.dense(keys,    num_units, activation=None, name='V')

        def split_heads(x):
            b, t, c = tf.unstack(tf.shape(x))
            x = tf.reshape(x, [b, t, num_heads, num_units // num_heads])
            return tf.transpose(x, [0, 2, 1, 3])  # (B, h, T, d)

        Q_ = split_heads(Q)
        K_ = split_heads(K)
        V_ = split_heads(V)

        # Scaled dot-product
        scores = tf.matmul(Q_, K_, transpose_b=True)  # (B,h,Tq,Tk)
        scores /= (num_units // num_heads) ** 0.5

        # Key padding mask (keys == 0)
        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))  # (B,Tk)
        key_masks = tf.tile(tf.expand_dims(tf.expand_dims(key_masks, 1), 1),
                            [1, num_heads, tf.shape(queries)[1], 1])
        paddings = tf.ones_like(scores) * (-2 ** 9)
        scores = tf.where(tf.equal(key_masks, 0), paddings, scores)

        # Causality mask
        if causality:
            T_q = tf.shape(queries)[1]
            T_k = tf.shape(keys)[1]
            tril = tf.linalg.LinearOperatorLowerTriangular(
                tf.ones([T_q, T_k], dtype=tf.float32)).to_dense()
            masks = tf.tile(tf.expand_dims(tf.expand_dims(tril, 0), 0),
                            [tf.shape(queries)[0], num_heads, 1, 1])
            scores = tf.where(tf.equal(masks, 0), paddings, scores)

        weights = tf.nn.softmax(scores)
        weights = tf.layers.dropout(weights, rate=dropout_rate, training=is_training)

        context = tf.matmul(weights, V_)                           # (B,h,Tq,d)
        context = tf.transpose(context, [0, 2, 1, 3])              # (B,Tq,h,d)
        outputs = tf.reshape(context, [tf.shape(queries)[0], tf.shape(queries)[1], num_units])

        # Residual + LN
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=is_training)
        outputs += queries
        outputs = normalize(outputs, scope="ln")
        return outputs


def feedforward(inputs, num_units, scope="feedforward", reuse=None,
                dropout_rate=0.0, is_training=False):
    """Position-wise FFN: Conv1D(1) -> Dropout -> Conv1D(1) -> Dropout + Residual + LN."""
    with tf.variable_scope(scope, reuse=reuse):
        params = {"inputs": inputs, "filters": num_units[0],
                  "kernel_size": 1, "activation": tf.nn.relu, "use_bias": True}
        outputs = tf.layers.conv1d(**params)
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=is_training)

        params = {"inputs": outputs, "filters": num_units[1],
                  "kernel_size": 1, "activation": None, "use_bias": True}
        outputs = tf.layers.conv1d(**params)
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=is_training)

        # Residual + LN
        outputs += inputs
        outputs = normalize(outputs, scope="ln_ffn")
        return outputs
