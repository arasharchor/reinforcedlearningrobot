import numpy as np
import matplotlib.pyplot as plt
import sys
caffe_root = '~/caffe_new/caffe/'
sys.path.insert(0, caffe_root + 'python')
import caffe

import os

# '/home/ubuntu/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel')
#if os.path.isfile('/home/ubuntu/catkin_ws/src/dlrobot/src/bvlc_alexnet/bvlc_reference_caffenet.caffemodel'):
##    print 'CaffeNet found.'
#else:
#    print 'Downloading pre-trained CaffeNet model...'
#   !..scripts/download_model_binary.py ../models/bvlc_reference_caffenet


caffe.set_mode_gpu()
print "GPU mode set!"


#model_def = '/home/ubuntu/caffe/models/bvlc_reference_caffenet/deploy.prototxt'
#model_weights = '/home/ubuntu/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
#model_def = '/home/ubuntu/catkin_ws/src/dlrobot/src/bvlc_alexnet/deploy.prototxt'
#model_weights = '/home/ubuntu/catkin_ws/src/dlrobot/src/bvlc_alexnet/bvlc_reference_caffenet.caffemodel'
model_def = '/home/ubuntu/catkin_ws/src/dlrobot/src/bvlc_reference_caffenet/deploy.prototxt'
#model_weights = '/home/ubuntu/catkin_ws/src/dlrobot/src/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
model_weights = '/media/ubuntu/BOOT/Snap/Snap_iter_10000.caffemodel'
net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

print "Model loaded!"

# load the mean ImageNet image (as distributed with Caffe) for subtraction
mu = np.load('/home/ubuntu/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
print 'mean-subtracted values:', zip('BGR', mu)

# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

print "Data transformed"

# set the size of the input (we can skip this if we're happy
#  with the default; we can also change it later, e.g., for different batch sizes)
net.blobs['data'].reshape(12,        # batch size
                          3,         # 3-channel (BGR) images
                          128, 128)  # image size is 227x227

imagenr = 'colorImg_'

for i in range(80):
    image = caffe.io.load_image('/home/ubuntu/catkin_ws/src/dlrobot/src/training/img_database/colorImg.png')
    transformed_image = transformer.preprocess('data', image)
    #plt.imshow(image)
    print "imgnr_%s loaded" % i


    # copy the image data into the memory allocated for the net
    net.blobs['data'].data[...] = transformed_image

    ### perform classification
    output = net.forward()

    output_prob = output['prob'][0]  # the output probability vector for the first image in the batch

    print 'predicted class is:', output_prob.argmax()

    # load ImageNet labels

    labels_file = '/home/ubuntu/catkin_ws/src/dlrobot/src/DL_forward.txt'
    #labels_file = '/home/ubuntu/caffe/data/ilsvrc12/synset_words.txt'

    #if not os.path.exists(labels_file):
    #    !../data/ilsvrc12/get_ilsvrc_aux.sh

    labels = np.loadtxt(labels_file, str, delimiter='\t')

    print 'output label:', labels[output_prob.argmax()]

    # sort top five predictions from softmax output
    top_inds = output_prob.argsort()[::-1][:5]  # reverse sort and take five largest items

    print 'probabilities and labels:'
    zip(output_prob[top_inds], labels[top_inds])
