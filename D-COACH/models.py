import tensorflow.contrib.layers as lays
import tensorflow as tf


def autoencoder(ae_inputs):
    pass

def fully_connected_layers(x, dim_a, fc_layers_neurons, loss_function_type):
    # Fully connected layer (in tf contrib folder for now)
    if len(x.shape) == 1:
        x = tf.reshape(x, shape=(x.shape[0],1))
    fc1 = tf.layers.dense(x, fc_layers_neurons, activation=tf.nn.tanh)
    fc2 = tf.layers.dense(fc1, fc_layers_neurons, activation=tf.nn.tanh)

    # Output layer, class prediction
    y = tf.layers.dense(fc2, dim_a, activation=tf.nn.tanh, name='action')

    y_ = tf.placeholder(tf.float32, [None, dim_a], name='label')

    # define the loss function
    if loss_function_type == 'cross_entropy':
        loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    elif loss_function_type == 'mean_squared':
        loss = 0.5 * tf.reduce_mean(tf.square(y - y_))
    else:
        loss = None
        print('No existing loss function was selected, please try mean_squared or cross_entropy')
        exit()

    return y, loss
