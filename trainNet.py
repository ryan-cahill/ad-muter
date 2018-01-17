#if no cuda, run 'sudo ldconfig /usr/local/cuda/lib64'
import pickle
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
import tensorflow as tf
import numpy as np
import gc

def sliceAudio(songArray):
    sliceWidth = 200
    totalSlices = 0
    for song in songArray:
        lengthToUse = len(song) - (len(song) % sliceWidth)
        totalSlices += lengthToUse / sliceWidth

    groupSlices = np.zeros((totalSlices, sliceWidth, songArray[0].shape[1]), dtype=float)
    arrayIndex = 0
    for song in songArray:
        lengthToUse = len(song) - (len(song) % sliceWidth)
        song = song[0:lengthToUse]
        splitParts = np.split(song, len(song) / sliceWidth)
        groupSlices[arrayIndex:arrayIndex + len(splitParts)] = splitParts
        arrayIndex += len(splitParts)
    return groupSlices

def getFlattenedSlices(cleansedSlices, oneHotLabel):
    flattenedSlices = []
    labels = []
    for slicesList in cleansedSlices:
        flattenedSlices.append(slicesList.reshape(slicesList.shape[0], slicesList.shape[1], 1))
        labels.append(oneHotLabel)
    return flattenedSlices, labels

#########################################################################

edmAudio = []
adAudio = []
extraData = []
with open('edmAudio.pickle', 'rb') as file:
    edmAudio = pickle.load(file)
with open('adAudio.pickle', 'rb') as file:
    adAudio = pickle.load(file)
with open('extraData.pickle', 'rb') as file:
    extraData = pickle.load(file)

ad_slices = sliceAudio(np.asarray(adAudio))
edm_slices = sliceAudio(np.asarray(edmAudio))

oneHotLabelAd = np.zeros(2)
oneHotLabelAd[0] = 1
flattenedAdSlices, adLabels = getFlattenedSlices(np.array(ad_slices), oneHotLabelAd)
oneHotLabelEdm = np.zeros(2)
oneHotLabelEdm[1] = 1
flattenedEdmSlices, edmLabels = getFlattenedSlices(np.asarray(edm_slices), oneHotLabelEdm)

del edmAudio
del adAudio
del ad_slices
del edm_slices
gc.collect()

normalizedEdmSlices = np.asarray(flattenedEdmSlices) / extraData['edmMaximum']
normalizedAdSlices = np.asarray(flattenedAdSlices) / extraData['adMaximum']

del flattenedEdmSlices
del flattenedAdSlices
gc.collect()

allSlices = np.concatenate((normalizedAdSlices, normalizedEdmSlices), axis=0)
allLabels = np.concatenate((adLabels, edmLabels), axis=0)

##########################################################################

TOTAL_CLASSES = 2
BASE_LEARNING_RATE = 0.000005
NUM_TRAINING_STEPS = 10000
BATCH_SIZE = 64
SNAPSHOT_STEPS = 100
CONVOLUTION1_DEPTH = 32
CONVOLUTION2_DEPTH = 32
#CONVOLUTION3_DEPTH = 128
CONVOLUTION1_KERNEL_SIZE = 9
CONVOLUTION2_KERNEL_SIZE = 5
#CONVOLUTION3_KERNEL_SIZE = 5

#86% with 32, 32, 7, 7

with tf.device('/gpu:0'):
    tflearn.config.init_graph(gpu_memory_fraction=0.9)

    # Building convolutional network
    network = input_data(shape=[None, allSlices[0].shape[0], allSlices[0].shape[1], 1], name='input')
    network = conv_2d(network, CONVOLUTION1_DEPTH, 1, activation='relu', regularizer="L2")
    network = max_pool_2d(network, CONVOLUTION1_KERNEL_SIZE)
    network = conv_2d(network, CONVOLUTION2_DEPTH, 1, activation='relu', regularizer="L2")
    network = max_pool_2d(network, CONVOLUTION2_KERNEL_SIZE)
    #network = conv_2d(network, CONVOLUTION3_DEPTH, 1, activation='relu', regularizer="L2")
    #network = max_pool_2d(network, CONVOLUTION3_KERNEL_SIZE)
    network = fully_connected(network, TOTAL_CLASSES, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=BASE_LEARNING_RATE,
                         loss='categorical_crossentropy', name='target')

    # Training the auto encoder
    model = tflearn.DNN(network, tensorboard_verbose=0)
    model.fit({'input': allSlices}, {'target': allLabels}, n_epoch=NUM_TRAINING_STEPS, batch_size=BATCH_SIZE, validation_set=0.1,
              snapshot_step=SNAPSHOT_STEPS, show_metric=True, run_id='classifier')