#Imports
import tensorflow as tf
from tensorflow.models.rnn.rnn import *


#Input Params
input_dim = 2

#The Tensorflow Graph
graph = tf.Graph()

with graph.as_default():
    ##The Input Layer as a Placeholder
    #Since we will provide data sequentially, the 'batch size'
    #is 1.
    input_layer = tf.placeholder(tf.float32, [1, input_dim])

    ##The LSTM Layer-1
    #The LSTM Cell initialization
    lstm_layer1 = rnn_cell.BasicLSTMCell(input_dim)
    #The LSTM state's as a Variable initialized to zeroes
    lstm_state1 = tf.Variable(tf.zeros([1, lstm_layer1.state_size]))
    #Connect the input layer and initial LSTM state to the LSTM cell
    lstm_output1, lstm_state_output1 = lstm_layer1(input_layer, lstm_state1,
                                                  scope="LSTM1")
    #The LSTM state will get updated
    lstm_update_op1 = lstm_state1.assign(lstm_state_output1)

    ##The Regression-Output Layer1
    #The Weights and Biases matrices first
    output_W1 = tf.Variable(tf.truncated_normal([input_dim, input_dim]))
    output_b1 = tf.Variable(tf.zeros([input_dim]))
    #Compute the output
    final_output = tf.matmul(lstm_output1, output_W1) + output_b1

    #final_output = lstm_output1

    ##Input for correct output (for training)
    correct_output = tf.placeholder(tf.float32, [1, input_dim])

    ##Calculate the Sum-of-Squares Error
    error = tf.pow(tf.sub(final_output, correct_output), 2)

    ##The Optimizer
    #Adam works best
    train_step = tf.train.AdamOptimizer(0.006).minimize(error)

    ##Session
    sess = tf.Session()
    #Initialize all Variables
    sess.run(tf.initialize_all_variables())


##Producing training/testing inputs+output
from numpy import array, sin, cos, pi
from random import random

sin_angle = random()
cos_angle = random()


def get_sample():
    global sin_angle, cos_angle
    sin_angle += 2*pi/300.0
    cos_angle += 2*pi/300.0
    if sin_angle > 2*pi:
        sin_angle -= 2*pi
    if cos_angle > 2*pi:
        cos_angle -= 2*pi
    return array([1 + 100*array([sin(sin_angle), cos(cos_angle)])])


##Training

current_input = get_sample()

errors1 = []
errors2 = []
values1 = []
values2 = []
x_axis = []


for i in range(100000):
    next_input = get_sample()
    s1, _, outputs, current_error = sess.run([lstm_update_op1,
                                    train_step,
                                    final_output,
                                    error],
                                   feed_dict = {
                                       input_layer: current_input,
                                       correct_output: next_input})

    current_input = next_input
    errors1.append(outputs[0][0])
    errors2.append(outputs[0][1])
    values1.append(current_input[0][0])
    values2.append(current_input[0][1])
    x_axis.append(i)

import matplotlib.pyplot as plt
plt.plot(x_axis, errors1, 'r-', x_axis, values1, 'b-')
plt.show()
plt.plot(x_axis, errors2, 'r-', x_axis, values2, 'b-')
plt.show()

##Testing

#Flush LSTM state
with graph.as_default():
    sess.run(lstm_state1.assign(tf.zeros([1, lstm_layer1.state_size])))

errors1 = []
errors2 = []
values1 = []
values2 = []
x_axis = []


for i in range(5000):
    next_input = get_sample()
    _, outputs, current_error = sess.run([lstm_update_op1,
                                    final_output,
                                    error],
                                   feed_dict = {
                                       input_layer: current_input,
                                       correct_output: next_input})

    current_input = next_input
    errors1.append(outputs[0][0])
    errors2.append(outputs[0][1])
    values1.append(current_input[0][0])
    values2.append(current_input[0][1])
    x_axis.append(i)

import matplotlib.pyplot as plt
plt.plot(x_axis, errors1, 'r-', x_axis, values1, 'b-')
plt.show()
plt.plot(x_axis, errors2, 'r-', x_axis, values2, 'b-')
plt.show()
