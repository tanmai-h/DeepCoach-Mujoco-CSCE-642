import tensorflow as tf
import numpy as np
from models import fully_connected_layers
from tools.functions import observation_to_gray, FastImagePlot
from agents.agent_base import AgentBase
from memory_buffer import MemoryBuffer

class Agent(AgentBase):
    def __init__(self, load_policy=False, learning_rate=0.001,
                 dim_a=3, fc_layers_neurons=100, loss_function_type='mean_squared',
                 policy_loc='./racing_car_m2/network', image_size=64, action_upper_limits='1,1',
                 action_lower_limits='-1,-1', e='1', show_ae_output=True, show_state=True, resize_observation=True,
                 ae_training_threshold=0.0011, ae_evaluation_frequency=40, observation_input_shape=(1,)):

        self.image_size = image_size
        self.observation_input_shape = observation_input_shape
        super(Agent, self).__init__(dim_a=dim_a, policy_loc=policy_loc, action_upper_limits=action_upper_limits,
                                    action_lower_limits=action_lower_limits, e=e, load_policy=load_policy,
                                    loss_function_type=loss_function_type, learning_rate=learning_rate,
                                    fc_layers_neurons=fc_layers_neurons)

        # High-dimensional state initialization
        self.resize_observation = resize_observation
        self.show_state = show_state
        self.show_ae_output = show_ae_output
        # Autoencoder training control variables
        self.ae_training = False
        self.ae_loss_history = MemoryBuffer(min_size=50, max_size=50)  # reuse memory buffer for the ae loss history
        self.ae_trainig_threshold = ae_training_threshold
        self.ae_evaluation_frequency = ae_evaluation_frequency
        self.mean_ae_loss = 1e7

    # @override
    def _build_network(self, dim_a, params):
        # Initialize graph
        print("HD_agent_Enhanced _build_network")
        with tf.variable_scope('base'):
            # Build autoencoder
            
            fc_inputs = tf.placeholder(tf.float32, [None] + list(self.observation_input_shape), name='input')
            # Build fully connected layers
            self.y, loss_policy = fully_connected_layers(fc_inputs, dim_a,
                                                         params['fc_layers_neurons'],
                                                         params['loss_function_type'])
            print('self.y=',self.y)
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'base')
        self.train_policy = tf.train.GradientDescentOptimizer(
            learning_rate=params['learning_rate']).minimize(loss_policy, var_list=variables)

        # self.train_ae = tf.train.AdamOptimizer(learning_rate=params['learning_rate']).minimize(self.loss_ae)

        # Initialize tensorflow
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        self.saver = tf.train.Saver()

    def _preprocess_observation(self, observation):
        pass

    def _batch_update_extra(self, state_batch, y_label_batch):
        # Calculate autoencoder loss and train if necessary
        pass
        # if self.ae_training:
        #     _, loss_ae = self.sess.run([self.train_ae, self.loss_ae], feed_dict={'base/input:0': state_batch})

        # else:
        #     loss_ae = self.sess.run(self.loss_ae, feed_dict={'base/input:0': state_batch})

        # # Append loss to loss buffer
        # self.ae_loss_history.add(loss_ae)

    def _evaluate_ae(self, t):
        # Check autoencoder mean loss in history and update ae_training flag
        pass
        # if t % self.ae_evaluation_frequency == 0:
        #     self.mean_ae_loss = np.array(self.ae_loss_history.buffer).mean()
        #     last_ae_training_state = self.ae_training

        #     if self.ae_loss_history.initialized() and self.mean_ae_loss < self.ae_trainig_threshold:
        #         self.ae_training = False
        #     else:
        #         self.ae_training = True

        #     # If flag changed, print
        #     if last_ae_training_state is not self.ae_training:
        #         print('\nTraining autoencoder:', self.ae_training, '\n')

    def _refresh_image_plots(self, t):
        if t % 4 == 0 and self.show_state:
            self.state_plot.refresh(self.high_dim_observation)

        # if (t+2) % 4 == 0 and self.show_ae_output:
        #     self.ae_output_plot.refresh(self.ae_output.eval(session=self.sess,
        #                                                     feed_dict={'base/input:0': self.high_dim_observation})[0])

    def time_step(self, t):
        self._evaluate_ae(t)
        self._refresh_image_plots(t)

    def new_episode(self):
        # print('\nTraining autoencoder:', self.ae_training)
        # print('Last autoencoder mean loss:', self.mean_ae_loss, '\n')
        pass

