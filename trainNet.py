#if no cuda, run 'sudo ldconfig /usr/local/cuda/lib64'
import pickle
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
import tensorflow as tf
import numpy as np
import gc

def slice_audio(song_array):
    slice_width = 200
    total_slices = 0
    for song in song_array:
        length_to_use = len(song) - (len(song) % slice_width)
        total_slices += length_to_use / slice_width

    group_slices = np.zeros((total_slices, slice_width, song_array[0].shape[1]), dtype=float)
    array_index = 0
    for song in song_array:
        length_to_use = len(song) - (len(song) % slice_width)
        song = song[0:length_to_use]
        split_parts = np.split(song, len(song) / slice_width)
        group_slices[array_index:array_index + len(split_parts)] = split_parts
        array_index += len(split_parts)
    return group_slices

def get_flattened_slices(cleansed_slices, one_hot_label):
    flattened_slices = []
    labels = []
    for slices_list in cleansed_slices:
        flattened_slices.append(slices_list.reshape(slices_list.shape[0], slices_list.shape[1], 1))
        labels.append(one_hot_label)
    return flattened_slices, labels

#########################################################################

edm_audio = []
ad_audio = []
extra_data = []
with open('edmAudio.pickle', 'rb') as file:
    edm_audio = pickle.load(file)
with open('adAudio.pickle', 'rb') as file:
    ad_audio = pickle.load(file)
with open('extraData.pickle', 'rb') as file:
    extra_data = pickle.load(file)

ad_slices = slice_audio(np.asarray(ad_audio))
edm_slices = slice_audio(np.asarray(edm_audio))

one_hot_label_ad = np.zeros(2)
one_hot_label_ad[0] = 1
flattened_ad_slices, ad_labels = get_flattened_slices(np.array(ad_slices), one_hot_label_ad)
one_hot_label_edm = np.zeros(2)
one_hot_label_edm[1] = 1
flattened_edm_slices, edm_labels = get_flattened_slices(np.asarray(edm_slices), one_hot_label_edm)

del edm_audio
del ad_audio
del ad_slices
del edm_slices
gc.collect()

normalized_edm_slices = np.asarray(flattened_edm_slices) / extra_data['edmMaximum']
normalized_ad_slices = np.asarray(flattened_ad_slices) / extra_data['adMaximum']

del flattened_edm_slices
del flattened_ad_slices
gc.collect()

allSlices = np.concatenate((normalized_ad_slices, normalized_edm_slices), axis=0)
allLabels = np.concatenate((ad_labels, edm_labels), axis=0)

##########################################################################

TOTAL_CLASSES = 2
BASE_LEARNING_RATE = 0.000005
NUM_TRAINING_STEPS = 100000
BATCH_SIZE = 32
SNAPSHOT_STEPS = 1000
CONVOLUTION1_DEPTH = 64
CONVOLUTION2_DEPTH = 32
CONVOLUTION3_DEPTH = 16
CONVOLUTION1_KERNEL_SIZE = 7
CONVOLUTION2_KERNEL_SIZE = 7
CONVOLUTION3_KERNEL_SIZE = 7

#86% with 32, 32, 7, 7

with tf.device('/gpu:0'):
    tflearn.config.init_graph(gpu_memory_fraction=0.9)

    # Building convolutional network
    network = input_data(shape=[None, allSlices[0].shape[0], allSlices[0].shape[1], 1], name='input')
    network = conv_2d(network, CONVOLUTION1_DEPTH, 1, activation='relu', regularizer="L2")
    network = max_pool_2d(network, CONVOLUTION1_KERNEL_SIZE)
    network = conv_2d(network, CONVOLUTION2_DEPTH, 1, activation='relu', regularizer="L2")
    network = max_pool_2d(network, CONVOLUTION2_KERNEL_SIZE)
    network = conv_2d(network, CONVOLUTION3_DEPTH, 1, activation='relu', regularizer="L2")
    network = max_pool_2d(network, CONVOLUTION3_KERNEL_SIZE)
    network = fully_connected(network, TOTAL_CLASSES, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=BASE_LEARNING_RATE,
                         loss='categorical_crossentropy', name='target')

    # Training the auto encoder
    model = tflearn.DNN(network, tensorboard_verbose=0)
    model.fit({'input': allSlices}, {'target': allLabels}, n_epoch=NUM_TRAINING_STEPS, batch_size=BATCH_SIZE, validation_set=0.1,
              snapshot_step=SNAPSHOT_STEPS, show_metric=True, run_id='classifier')