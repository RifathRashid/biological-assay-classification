
# coding: utf-8

# In[1]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sklearn as sk
from sklearn import metrics
import numpy as np
import pandas as pd
import tensorflow as tf
import argparse
import os

# structure of code largely based on tensorflow tutorial for deep neural nets: https://www.tensorflow.org/get_started/mnist/pros
# see full code of tutorial here: https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/examples/tutorials/mnist/mnist_deep.py


# In[2]:


## Utility functions

def sign(x, threshold=0):
    y = x > threshold
    return y.astype(int)

def get_data_filenames(data_dir, data_file_ext, assay_name):
    '''
    Returns dictionary mapping 'train', 'test', and 'score' to the corresponding data filename
    '''
    return {subfolder: os.path.join(os.getcwd(), data_dir, subfolder, '') + assay_name + '.' + data_file_ext             for subfolder in ['train', 'test', 'score']}

def read_fingerprint(filename):
    '''
    Parameters
    - filename: str
        File must be tab-delimited as follows: smiles code, tox21_id, label, fingerprint
    
    Returns
    - (X, Y): tuple of np.arrays
        X is an array of features
        Y is a vector of labels
    '''
    X = []
    Y = []
    input_file = open(filename, 'r')
    
    for index, line in enumerate(input_file):
        # split line (1 data point) into smiles, fingerprint (features), and label
        split_line = line.strip().split('\t')
        # print(index)
        # smiles = split_line[0]
        fingerprint = [int(c) for c in split_line[3]]
        label = int(split_line[2])
        
        # append data point to train_x (features) and train_y (labels)
        X.append(fingerprint)
        Y.append(label)
    input_file.close()
    return (np.array(X), np.array(Y))

def read_features(filename):
    '''
    Parameters
    - filename: str
        File must be tab-delimited as follows: smiles code, cid, pubchem_fingerprint, 33 extra features (tab-delimited), label
    
    Returns
    - (X, Y): tuple of np.arrays
        X is an array of features
        Y is a vector of labels
    '''
    X = []
    Y = []
    input_file = open(filename, 'r')
    
    for index, line in enumerate(input_file):
        try:
            # split line (1 data point) into smiles, fingerprint (features), 33 extra featues, and label
            split_line = line.strip().split()
            fingerprint = [int(c) for c in split_line[2]]
            label = int(split_line[36])
            extra_features = split_line[3:36]
            fingerprint.extend(extra_features)

            # append data point to X (features) and Y (labels)
            X.append(fingerprint)
            Y.append(label)
        except:
            print('failed to parse data point %d' % index)
            continue
    input_file.close()
    return (np.array(X), np.array(Y))

## if running as main function

# construct parser
parser = argparse.ArgumentParser()
parser.add_argument('--run_id', type=str, default='', help='run id')
parser.add_argument('--rand_seed', type=int, default='848', help='graph-level random seed for tensorflow')
parser.add_argument('--assay_name', type=str, required=True, help='assay name, e.g. nr-ar, sr-are, ...')
parser.add_argument('--data_dir', type=str, required=True, help='name of directory to find train, test, and score data files')
parser.add_argument('--data_file_ext', type=str, default='data', help='file extension, exluduing the period (e.g. ''fp'', ''data'', etc)')
parser.add_argument('--loss_balance', action='store_true', help='adjust loss function to account for unbalanced dataset, default = false')
parser.add_argument('--kernel_reg_const', type=float, default=0.1, help='L2 kernel regularization constant')
parser.add_argument('--batch_size', type=int, default=1, help='batch size. default = 1 (SGD)')
parser.add_argument('--num_epochs', type=int, default=1, help='number of epochs (passes through entire training set)')
parser.add_argument('--node_array', nargs='*', required=True, help='sizes of hidden layers in the neural network. use 0 for a simple linear classifier')

# parse arguments
args = parser.parse_args()
run_id = args.run_id
rand_seed = args.rand_seed
assay_name = args.assay_name
data_dir = args.data_dir
data_file_ext = args.data_file_ext.lstrip('.')
loss_balance = args.loss_balance
kernel_reg_const = args.kernel_reg_const
batch_size = args.batch_size
num_epochs = args.num_epochs
node_array = np.array(args.node_array, dtype=int)

params={'run_id': run_id,
        'rand_seed': rand_seed,
        'assay_name': assay_name,
        'data_dir': data_dir,
        'data_file_ext': data_file_ext,
        'loss_balance': loss_balance,
        'kernel_reg_const': kernel_reg_const,
        'batch_size': batch_size,
        'num_epochs': num_epochs,
        'node_array': node_array}

# get data
filenames = get_data_filenames(data_dir, data_file_ext, assay_name)
X_train, Y_train = read_features(filenames['train'])
X_test, Y_test = read_features(filenames['test'])
num_features = X_train.shape[1]
# In[3]:


# ## if running inside iPython notebook

# # parameters
# run_id = 1
# rand_seed = 848
# assay_name = 'nr-ahr'
# data_dir = 'data_pcfp_ext'
# data_file_ext = 'features'
# loss_balance = True
# kernel_reg_const = 0.1
# batch_size = 50
# num_epochs = 3
# node_array = np.array([512, 256, 128])

# params={'run_id': run_id,
#         'rand_seed': rand_seed,
#         'assay_name': assay_name,
#         'data_dir': data_dir,
#         'data_file_ext': data_file_ext,
#         'loss_balance': loss_balance,
#         'kernel_reg_const': kernel_reg_const,
#         'batch_size': batch_size,
#         'num_epochs': num_epochs,
#         'node_array': node_array}

# # get data
# filenames = get_data_filenames(data_dir, data_file_ext, assay_name)
# X_train, Y_train = read_features(filenames['train'])
# X_test, Y_test = read_features(filenames['test'])
# num_features = X_train.shape[1]


# In[4]:


## Model - basic ##

# Notes
# Probability of classifying into the positive class = sigmoid(logit)
# logit can take on any real value

def deepnn_params(x, nodes, kernel_reg_const=0.1):
    """
    deepnn builds the graph for a deep net for learning the logit

    Args:
        x: input layer. type = tf.Tensor. size = (batch_size, num_features)
        nodes: a list of number of nodes in hidden layers. type = np.ndarray
        kernel_reg_const: L2 regularization weights. type = float

    Returns:
        y: a tensor of length batch_size with values equal to the logits
            of classifying an input data point into the positive class
    """
    # tensorflow dense layer example: https://www.tensorflow.org/tutorials/layers#dense_layer
    
    layers = []
    layers.append(x)
    
    num_hidden_layers = min(nodes.size,nodes[0])
    for i in range(num_hidden_layers):
        layers.append(tf.layers.dense(inputs=layers[i], units=nodes[i], activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(kernel_reg_const)))
    layers.append(tf.layers.dense(inputs=layers[num_hidden_layers], units=1, activation=None, kernel_regularizer=tf.contrib.layers.l2_regularizer(kernel_reg_const)))
    return tf.squeeze(layers[-1])

# sign tensorflow function
def sign_tf(x, threshold=0):
    return tf.cast(tf.greater_equal(x, threshold), tf.int32)

# input
x = tf.placeholder(tf.float32, [None, num_features])

# labels
y_labels = tf.placeholder(tf.float32, [None]) # domain: {0,1}

# loss weights for unbalanced data
q = tf.placeholder(tf.float32, None)

# Build the graph for the deep net
y_score = deepnn_params(x, node_array, kernel_reg_const)
y_prob = tf.sigmoid(y_score)

# Define loss and optimizer
# logistic loss, aka sigmoid cross entropy
# y * -log(sigmoid(x)) + (1 - y) * -log(1 - sigmoid(x)), where x is the logit and y is the label
loss_fn = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=y_labels, logits=y_score, pos_weight=q))
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss_fn)

correct_prediction = tf.equal(sign_tf(y_score), tf.cast(y_labels, tf.int32))
correct_prediction = tf.cast(correct_prediction, tf.float32)
accuracy = tf.reduce_mean(correct_prediction)


# In[5]:


## Train model ##

np.random.seed(rand_seed)

# calculate frequencies of positives, negatives in training set
# - https://stackoverflow.com/questions/35155655/loss-function-for-class-imbalanced-binary-classifier-in-tensor-flow
q_train = Y_train.size/np.sum(Y_train)
if not loss_balance:
    q_train = 1

# training parameters
num_batches_per_epoch = int(np.ceil(len(X_train) / batch_size))

sess = tf.InteractiveSession()
tf.set_random_seed(rand_seed)
sess.run(tf.global_variables_initializer())

# training loop
for epoch in range(num_epochs):
    # shuffle indices of training data
    shuffle_indices = np.arange(X_train.shape[0])
    np.random.shuffle(shuffle_indices)

    for i in range(num_batches_per_epoch):
        # get batch
        batch_indices = shuffle_indices[i*batch_size : (i+1)*batch_size]
        batch_x = X_train[batch_indices]
        batch_y = Y_train[batch_indices]

        # train on batch data
        sess.run(train_step, feed_dict={x: batch_x, y_labels: batch_y, q: q_train})


# In[6]:


## AUROC - sklearn

train_accuracy, train_loss = sess.run([accuracy, loss_fn], feed_dict={x: X_train, y_labels: Y_train, q: q_train})

# get normalized score, i.e. probability of classifying into positive class
y_prob_test = sess.run(y_prob, feed_dict={x: X_test})
test_accuracy = sess.run(accuracy, feed_dict={x: X_test, y_labels: Y_test})

fpr, tpr, thresholds = sk.metrics.roc_curve(Y_test, y_prob_test)
auc_roc = sk.metrics.auc(fpr, tpr)


# In[8]:


## save parameters, accuracy, and auc_roc
results_file = os.path.join(os.getcwd(), 'results','') + str(run_id) + '.results'

params['accuracy'] = test_accuracy
params['auc_roc'] = auc_roc
params['train_loss'] = train_loss
params['train_accuracy'] = train_accuracy
params['data_train_size'] = X_train.shape[0]
params['data_test_size'] = X_test.shape[0]

series = pd.Series(params)
df = pd.DataFrame(series)
df = df.T
df.to_csv(results_file, index=False)
# series.to_csv(results_file)

