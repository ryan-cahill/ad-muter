import h5py
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression

musicDataSetExtracted = h5py.File("musicData.hdf5", "r")
extractedSlices = musicDataSetExtracted["allSlices"][:]
extractedLabels = musicDataSetExtracted["allLabels"][:]
musicDataSetExtracted.close()

##########################################################################

TOTAL_CLASSES = 2
BASE_LEARNING_RATE = 0.0005
NUM_TRAINING_STEPS = 10000
BATCH_SIZE = 128
SNAPSHOT_STEPS = 100
CONVOLUTION1_DEPTH = 64
CONVOLUTION2_DEPTH = 128
CONVOLUTION3_DEPTH = 128
CONVOLUTION1_KERNEL_SIZE = 3
CONVOLUTION2_KERNEL_SIZE = 3
CONVOLUTION3_KERNEL_SIZE = 3
#cv2.imwrite("image0.png", extractedImages[0]) #for testing and validation of h5 file

tflearn.config.init_graph(gpu_memory_fraction=0.5)

# Building convolutional network
network = input_data(shape=[None, extractedSlices[0].shape[0], extractedSlices[0].shape[1], 1], name='input')
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
model.fit({'input': extractedSlices}, {'target': extractedLabels}, n_epoch=NUM_TRAINING_STEPS, batch_size=BATCH_SIZE,
          snapshot_step=SNAPSHOT_STEPS, show_metric=True, run_id='classifier')