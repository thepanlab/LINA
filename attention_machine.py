import tensorflow as tf


class AttentionNetwork(tf.keras.Model):
    def __init__(self, num_feature, list_reduc, l2_strength, dropout_act=False, classification=False,bn=True):
        super(AttentionNetwork, self).__init__()
        self.list_reduc = list_reduc
        self.num_feature = num_feature
        self.l2_strength = l2_strength
        self.reduction_layers = [tf.keras.layers.Dense(list_reduc[k], activation=tf.nn.leaky_relu)
                                 for k in range(len(list_reduc))
                                 ]
        self.dense = tf.keras.layers.Dense(num_feature, activation=None)
        self.bn = [tf.keras.layers.BatchNormalization()
                   for k in range(len(list_reduc))]
        self.scaling = tf.keras.layers.BatchNormalization()
        self.dropout_act = dropout_act
        self.classification = classification
        self.bn_act=bn
        if self.dropout_act is True:
            self.dropout_in = tf.keras.layers.Dropout(0.5)
            self.dropout = [tf.keras.layers.Dropout(
                0.5) for k in range(len(list_reduc))]
            self.dropout_H = tf.keras.layers.Dropout(0.5)
        self.logit = tf.keras.layers.Dense(
            1, activation=None, kernel_regularizer=tf.keras.regularizers.l2(self.l2_strength))

    def call(self, x):
        if self.dropout_act is True:
            dropout = self.dropout_in(x)
        X = x
        '''Reduction'''
        for i in range(len(self.list_reduc)):
            x = self.reduction_layers[i](x)
            if self.bn_act is True:
                x = self.bn[i](x)
            if self.dropout_act is True:
                x = self.dropout[i](x)

        '''Compute A'''
        scores = self.dense(x)
        A = self.scaling(scores)

        '''Sample jacobian'''
        dA_dx = []
        #for Ai in tf.split(A, self.num_feature, axis=1):
        #    dA_dx.append(tf.gradients(Ai, X))
        '''Get H'''
        H = tf.keras.layers.multiply([A, X])
        dH_dx = []
        # for Hi in tf.split(H, self.num_feature, axis=1):
        #    dH_dx.append(tf.gradients(Hi, X))
        '''Linear Layer'''
        if self.dropout_act is True:
            dp = self.dropout_H(H)
            logit = self.logit(dp)
        else:
            logit = self.logit(H)
        if self.classification is True:
            p = tf.nn.sigmoid(logit)
        else:
            p = logit
        dp_dx = tf.gradients(p, X)
        return p, logit, H, A, dA_dx, dp_dx, dH_dx
