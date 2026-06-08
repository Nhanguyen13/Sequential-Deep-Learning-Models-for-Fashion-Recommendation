# model.py  (TF1, Python 3; giữ logic gốc, chỉ sửa tối thiểu)
import tensorflow as tf
from modules import embedding, multihead_attention, feedforward, normalize


class Model(object):
    def __init__(self, usernum, itemnum, args, reuse=None):
        self.is_training = tf.placeholder(tf.bool, shape=())
        self.u          = tf.placeholder(tf.int32, shape=(None,))
        self.input_seq  = tf.placeholder(tf.int32, shape=(None, args.maxlen))
        self.pos        = tf.placeholder(tf.int32, shape=(None, args.maxlen))
        self.neg        = tf.placeholder(tf.int32, shape=(None, args.maxlen))

        pos = self.pos
        neg = self.neg
        # tf.to_float -> tf.cast(..., tf.float32) cho Python3/TF1
        mask = tf.expand_dims(tf.cast(tf.not_equal(self.input_seq, 0), tf.float32), -1)

        with tf.variable_scope("SASRec", reuse=reuse):
            # sequence embedding, item embedding table
            self.seq, item_emb_table = embedding(
                self.input_seq,
                vocab_size=itemnum + 1,
                num_units=args.hidden_units,
                zero_pad=True,
                scale=True,
                l2_reg=args.l2_emb,
                scope="input_embeddings",
                with_t=True,
                reuse=reuse
            )

            # Positional Encoding
            t, pos_emb_table = embedding(
                tf.tile(tf.expand_dims(tf.range(tf.shape(self.input_seq)[1]), 0),
                        [tf.shape(self.input_seq)[0], 1]),
                vocab_size=args.maxlen,
                num_units=args.hidden_units,
                zero_pad=False,
                scale=False,
                l2_reg=args.l2_emb,
                scope="dec_pos",
                reuse=reuse,
                with_t=True
            )
            self.seq += t

            # Dropout
            self.seq = tf.layers.dropout(
                self.seq, rate=args.dropout_rate,
                training=tf.convert_to_tensor(self.is_training)
            )
            self.seq *= mask

            # Build blocks
            for i in range(args.num_blocks):
                with tf.variable_scope("num_blocks_%d" % i):
                    # Self-attention  (đổi scope LN để tránh trùng biến)
                    self.seq = multihead_attention(
                        queries=normalize(self.seq, scope="ln_att_%d" % i),
                        keys=self.seq,
                        num_units=args.hidden_units,
                        num_heads=args.num_heads,
                        dropout_rate=args.dropout_rate,
                        is_training=self.is_training,
                        causality=True,
                        scope="self_attention"
                    )
                    # Feed forward  (đổi scope LN để tránh trùng biến)
                    self.seq = feedforward(
                        normalize(self.seq, scope="ln_ffn_in_%d" % i),
                        num_units=[args.hidden_units, args.hidden_units],
                        dropout_rate=args.dropout_rate,
                        is_training=self.is_training
                    )
                    self.seq *= mask

            # LN cuối cùng (đổi scope khác)
            self.seq = normalize(self.seq, scope="ln_out")

        # reshape & lookup
        pos = tf.reshape(pos, [tf.shape(self.input_seq)[0] * args.maxlen])
        neg = tf.reshape(neg, [tf.shape(self.input_seq)[0] * args.maxlen])
        pos_emb = tf.nn.embedding_lookup(item_emb_table, pos)
        neg_emb = tf.nn.embedding_lookup(item_emb_table, neg)
        seq_emb = tf.reshape(self.seq, [-1, args.hidden_units])

        # candidates for test
        self.test_item = tf.placeholder(tf.int32, shape=(None, None)) 
        test_item_emb  = tf.nn.embedding_lookup(item_emb_table, self.test_item) 
        self.test_logits = tf.matmul(seq_emb, tf.transpose(test_item_emb, perm=[0, 2, 1]))
        num_test_items = tf.shape(self.test_item)[1]
        self.test_logits = tf.reshape(self.test_logits, [tf.shape(self.input_seq)[0], args.maxlen, num_test_items])
        self.test_logits = self.test_logits[:, -1, :]


        # prediction layer
        self.pos_logits = tf.reduce_sum(pos_emb * seq_emb, -1)
        self.neg_logits = tf.reduce_sum(neg_emb * seq_emb, -1)

        # ignore padding items (0)
        istarget = tf.reshape(tf.cast(tf.not_equal(pos, 0), tf.float32),
                              [tf.shape(self.input_seq)[0] * args.maxlen])

        pos_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(self.pos_logits), logits=self.pos_logits)
        neg_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros_like(self.neg_logits), logits=self.neg_logits)
        
        self.loss = tf.reduce_sum(pos_loss * istarget + neg_loss * istarget) / tf.reduce_sum(istarget)
        
        #self.loss = tf.reduce_sum(
            #- tf.log(tf.sigmoid(self.pos_logits) + 1e-24) * istarget
            #- tf.log(1 - tf.sigmoid(self.neg_logits) + 1e-24) * istarget
        #) / tf.reduce_sum(istarget)

        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        if reg_losses:
            self.loss += tf.add_n(reg_losses)

        tf.summary.scalar('loss', self.loss)
        self.auc = tf.reduce_sum(((tf.sign(self.pos_logits - self.neg_logits) + 1) / 2) * istarget) / tf.reduce_sum(istarget)

        if reuse is None:
            tf.summary.scalar('auc', self.auc)
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=args.lr, beta2=0.98)
            self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)
        else:
            tf.summary.scalar('test_auc', self.auc)

        self.merged = tf.summary.merge_all()

    def predict(self, sess, u, seq, item_idx):
        return sess.run(self.test_logits, {
            self.u: u,
            self.input_seq: seq,
            self.test_item: [item_idx],
            self.is_training: False
        })


__all__ = ['Model']
