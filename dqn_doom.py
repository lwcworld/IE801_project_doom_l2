import tensorflow as tf
import numpy as np

class DQN:
    def __init__(self, session, input_size, output_size, name="main"):
        self.session = session
        self.input_size = input_size
        self.output_size = output_size
        self.net_name = name

        output_from_FEL = self._feature_extraction_layer()
        self._fully_connected_layer(input_data=output_from_FEL)

    def _feature_extraction_layer(self):
        with tf.variable_scope(self.net_name):
            # dropout (keep_prob) rate  0.7~0.5 on training, but should be 1 for testing
            self.training = tf.placeholder(tf.bool)

            # input place holders
            self.X = tf.placeholder(tf.float32, [None, self.input_size])

            # img 160x120x3 (black/white), Input Layer
            X_img = tf.reshape(self.X, [-1, 160, 120, 3]) ##

            # Convolutional Layer #1 (out : [40*40*32])
            conv1 = tf.layers.conv2d(inputs=X_img, bias_initializer=tf.zeros_initializer(), filters=32, kernel_size=[8,8], padding="SAME", activation=tf.nn.relu)
            # Pooling Layer #1 (out :
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[8, 8], padding="SAME", strides=4)
            dropout1 = tf.layers.dropout(inputs=pool1, rate=0.7, training=self.training)

            # Convolutional Layer #2 and Pooling Layer #2 (out : [20,20,64])
            conv2 = tf.layers.conv2d(inputs=dropout1, bias_initializer=tf.zeros_initializer(), filters=64, kernel_size=[4, 4], padding="SAME", activation=tf.nn.relu)
            # Pooling Layer #2 (out :
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[4, 4], padding="SAME", strides=2)
            dropout2 = tf.layers.dropout(inputs=pool2, rate=0.7, training=self.training)

            # Convolutional Layer #3 and Pooling Layer #2 (out : [10,10,64])
            conv3 = tf.layers.conv2d(inputs=dropout2, bias_initializer=tf.zeros_initializer(), filters=128, kernel_size=[3, 3], padding="SAME", activation=tf.nn.relu)
            # Pooling Layer #2 (out :
            pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[4, 4], padding="SAME", strides=2)
            dropout3 = tf.layers.dropout(inputs=pool3, rate=0.7, training=self.training)

            # Convolutional Layer #4 and Pooling Layer #2 (out : [4,4,512])
            conv4 = tf.layers.conv2d(inputs=dropout3, bias_initializer=tf.zeros_initializer(), filters=512, kernel_size=[7, 7], padding="SAME", activation=tf.nn.relu)
            # Pooling Layer #2 (out :
            pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[7, 7], padding="Valid", strides=1)
            dropout4 = tf.layers.dropout(inputs=pool4, rate=0.7, training=self.training)

            # Dense Layer with Relu
            # print(self.input_size)
            # flat = tf.reshape(pool4, [-1, 1*1*512])
            flat = tf.reshape(dropout4, [-1, 4*4*256])

        return flat # will be transfered to fully connected layer as a input

    def _fully_connected_layer(self, input_data, l_rate=0.01):
        with tf.variable_scope(self.net_name):
            dense1 = tf.layers.dense(inputs=input_data, units=32, activation=tf.nn.relu)
            dropout1 = tf.layers.dropout(inputs=dense1, rate=0.7, training=True)

            dense2 = tf.layers.dense(inputs=dropout1, units=24, activation=tf.nn.relu)
            dropout2 = tf.layers.dropout(inputs=dense2, rate=0.7, training=True)

            # dense3 = tf.layers.dense(inputs=dense2, units=16, activation=tf.nn.relu)
            # # dropout3 = tf.layers.dropout(inputs=dense3, rate=0.7, training=True)
            #
            # dense4 = tf.layers.dense(inputs=dense3, units=70, activation=tf.nn.relu)
            # # dropout4 = tf.layers.dropout(inputs=dense4, rate=0.7, training=True)

            # dense5 = tf.layers.dense(inputs=dropout4, units=230, activation=tf.nn.relu, bias_initializer=tf.zeros_initializer())
            # dropout5 = tf.layers.dropout(inputs=dense5, rate=0.7, training=True)

            dense_f = tf.layers.dense(dropout2, units=self.output_size)

            self._Qpred = dense_f

        # We need to define the parts of the network needed for learning a policy
        self._Y = tf.placeholder(shape=[None, self.output_size], dtype=tf.float32)
        # Loss function
        self._loss = tf.reduce_mean(tf.square(self._Y - self._Qpred))
        # Learning
        self._train = tf.train.AdamOptimizer(learning_rate=l_rate).minimize(self._loss)

    def predict(self, state, training=False):
        x = np.reshape(state, [-1, self.input_size])
        # return self.session.run(self._Qpred, feed_dict={self.X: x, self.training: training})
        Qpred_out = self.session.run(self._Qpred, feed_dict={self.X: x, self.training: training})
        return Qpred_out

    def update(self, x_stack, y_stack, training=False):
        return self.session.run([self._loss, self._train], feed_dict={self.X: x_stack, self._Y: y_stack, self.training: training})