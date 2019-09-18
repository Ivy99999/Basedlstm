# -*- coding: utf-8 -*-
#author ivy_nie
'''-------------------------------------------RNN模型------------------------------------------------'''
import tensorflow as tf



class RnnConfig(object):
    def __init__(self,args):
        # 模型参数
        self.embedding_dim = args.embedding_dim  # 词向量维度
        self.seq_length = args.seq_length  # 序列长度
        self.num_classes = args.num_classes  # 类别数
        self.vocab_size = args.vocab_size  # 词汇表大小
        self.n_layers = args.n_layers  # 隐藏层层数
        self.n_neurons = args.n_neurons  # 隐藏层神经元数量
        self.rnn = args.rnn  # lstm或gru
        self.dropout_keep_prob = args.dropout_keep_prob  # dropout保留比例
        self.learning_rate = args.learning_rate  # 学习率
        self.batch_size = args.batch_size  # 每批训练大小
        self.n_epochs = args.n_epochs  # 总迭代轮次
        self.print_per_batch = args.print_per_batch  # 每多少轮输出一次结果
        self.save_per_batch = args.save_per_batch  # 每多少轮存入tensorboard
        self.input_x = tf.placeholder(tf.int32, [None, args.seq_length])
        self.input_y = tf.placeholder(tf.float32, [None, args.num_classes])
        self.dropout_prob = tf.placeholder(tf.float32, name="dropout_prob")
        self.rnnn()
    def rnnn(self):
        def lstm_cell():
            return tf.contrib.rnn.BasicLSTMCell(num_units=self.n_neurons, state_is_tuple=True)

        def gru_cell():
            return tf.contrib.rnn.GRUCell(num_units=self.n_neurons)

        def basic_cell():
            return tf.contrib.rnn.BasicRNNCell(num_units=self.n_neurons)

        def dropout_cell():
            if self.rnn == "lstm":
                print("+++++++++++++")
                cell = lstm_cell()
            else:
                if self.rnn == "gru":
                    cell = gru_cell()
                else:
                    cell = basic_cell()

            return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.dropout_prob)
        # 词向量映射
        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding", shape=[self.vocab_size, self.embedding_dim])
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)
        with tf.name_scope("rnn"):
            # 多层rnn网络
            cells = [dropout_cell() for layer in range(self.n_layers)]
            rnn_cells = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
            outputs, states = tf.nn.dynamic_rnn(rnn_cells, inputs=embedding_inputs, dtype=tf.float32)
            last_outputs = outputs[:, -1, :]  # 取最后一个时序输出结果
            # 全连接层，后接dropout以及relu激活
            fc = tf.layers.dense(last_outputs, self.n_neurons, name="fc1")
            fc = tf.contrib.layers.dropout(fc, self.dropout_prob)
            fc = tf.nn.relu(fc, name="relu")
            # 输出层
            self.logits = tf.layers.dense(fc, self.num_classes, name="softmax")
            self.y_pred = tf.argmax(self.logits, 1)
        # 训练
        with tf.name_scope("training_op"):
            # 损失函数，交叉熵
            xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(xentropy)

            # 优化
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.optim = optimizer.minimize(self.loss)
        # 计算准确率
        with tf.name_scope("accuracy"):
            correct = tf.equal(self.y_pred, tf.argmax(self.input_y, 1))
            self.acc = tf.reduce_mean(tf.cast(correct, tf.float32))
            print("ending")
