import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge, CvBridgeError
import sixaxis
from geometry_msgs.msg import Twist
import subprocess
import time
import sys
import numpy as np
import caffe
import datetime
import signal

import os

caffe_root = '~/caffe'
sys.path.insert(0, caffe_root + 'python')

class camera_node:
    
    def __init__(self):
        rospy.init_node('camera_node')

        self.speed = 0.5
        self.angular_speed = 1
        self.log_data = False

        self.classify = False
        self.action = 5
        global callbackCounter
        callbackCounter = 0
        self.imageData = 0

        self.image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.callback)
        self.bridge = CvBridge()
        self.imgnr = 0
        self.imgskip = 0
        self.alpha = 1/10.0
        self.oldx = 0

        sixaxis.init("/dev/input/js0")
        self.pub = rospy.Publisher('mobile_base/commands/velocity', Twist)
        self.command = Twist()
        self.log_path = '/media/ubuntu/9016-4EF8/transferLearning_data/controlFile.log'
        self.logfile = open(self.log_path, 'a')
	self.classyfile = open('/media/ubuntu/9016-4EF8/transferLearning_data/classifierFile.log', 'w')
        #self.logfile.write("x; z; image")

        self.imageSaved = 0
        self.callbackCounter = 0

        #if os.path.isfile('/home/ubuntu/catkin_ws/src/dlrobot/src/bvlc_alexnet/bvlc_reference_caffenet.caffemodel'):
        #    print 'CaffeNet found.'
        #else:
        #    print 'Downloading pre-trained CaffeNet model...'
        #   !..scripts/download_model_binary.py ../models/bvlc_reference_caffenet


        caffe.set_mode_gpu()
	print 'loading weights'
        #model_def = '/home/ubuntu/caffe/models/bvlc_reference_caffenet/deploy.prototxt'
        #model_weights = '/home/ubuntu/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
        model_def = '/home/ubuntu/catkin_ws/src/dlrobot/src/bvlc_reference_caffenet/deploy.prototxt'
        
        model_weights = '/home/ubuntu/catkin_ws/src/dlrobot/src/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
	#model_weights = '/media/ubuntu/BOOT/Snap/Snap_iter_10000.caffemodel'
        self.net = caffe.Net(model_def,      # defines the structure of the model
                        model_weights,  # contains the trained weights
                        caffe.TEST)     # use test mode (e.g., don't perform dropout)
	print 'weights loaded'

        # load the mean ImageNet image (as distributed with Caffe) for subtraction
        mu = np.load('/home/ubuntu/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy')
        mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
        print 'mean-subtracted values:', zip('BGR', mu)
	
        # create transformer for the input called 'data'
        self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})

        self.transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
        self.transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
        self.transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
        self.transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

        # set the size of the input (we can skip this if we're happy
        #  with the default; we can also change it later, e.g., for different batch sizes)
        self.net.blobs['data'].reshape(128,        # batch size
                                  3,         # 3-channel (BGR) images
                                  128, 128)  # image size is 227x227
        print self.transformer
        print "tranformer loaded"
	self.counter = 0


    def __del__(self):
        self.logfile.close()

    def callback(self,data):

        self.callbackCounter += 1
        #print callbackCounter
        self.imageData = data

        self.imageSaved = 0

    def run(self):

	
        #print "callback image received"
        try:
            state = sixaxis.get_state()
            x = state['lefty'] / 125.0  # Speed  [-1 to 1]
            z = state['rightx'] * 3.14 / 100.0  # turn

            if self.log_data == True:
                self.imgskip += 1
                if self.callbackCounter % 10 == 0 and self.imageSaved == 0:
                    print "imgskip"
                    cv_image = self.bridge.imgmsg_to_cv2(self.imageData, "bgr8")

                    timestamp = datetime.datetime.now()
                    timestampminutes = timestamp.minute
                    timestampseconds = timestamp.second
                    timestampmili = timestamp.microsecond
                    timestamphour = timestamp.hour
                    timestampdate = timestamp.date()


                    theTime = "%s_%s_%s_%s_%s" % (timestampdate,timestamphour,timestampminutes,timestampseconds,timestampmili)


                    #filename = 'img_database/colorImg_%s.png' % theTime
                    filename = '/media/ubuntu/9016-4EF8/transferLearning_data/imageCaptured/colorImg_%s.png' % theTime
                    cv2.imwrite(filename, cv_image)
                    self.imgnr += 1

                    #print theTime
                    buf = "%f; %f; colorImg_%s.png\n" % (x, z,theTime)
                    self.logfile.write(buf)
                    #print filename
                    #self.imgskip = 0
                    self.imageSaved = 1

            if self.classify == True:
                print "inside classifier"
                self.imgskip = 1
		
                if self.imgskip  == 1:
                    t = time.time()
                    cv_image = self.bridge.imgmsg_to_cv2(self.imageData,"bgr8")
                    filename = 'img_database/colorImg.png'
		    self.counter += 1
                    cv2.imwrite(filename, cv_image)
		    cv2.imwrite('/media/ubuntu/9016-4EF8/transferLearning_data/classifierImages/img_%s.png' %self.counter,cv_image)

                    image = caffe.io.load_image('/home/ubuntu/catkin_ws/src/dlrobot/src/img_database/colorImg.png')
                    
                    #print time.time() -t

                    #image = caffe.io.load_image(cvimg)

                    transformed_image = self.transformer.preprocess('data', image)
                    #plt.imshow(image)
                    print "image transformed"
                    #print time.time() -t

                    # copy the image data into the memory allocated for the net
                    self.net.blobs['data'].data[...] = transformed_image
                    print "image loaded into memory"
                    ### perform classification
                    output = self.net.forward()
                    print "forward done"
                    #print time.time() -t
                    output_prob = output['prob'][0]  # the output probability vector for the first image in the batch

                    print 'predicted class is:', output_prob.argmax()
                    print output_prob.argmax()
                    self.action = output_prob.argmax()
		    self.classyfile.write('img_%s_%s \n' %(self.counter, self.action))


            if state['start'] == True:
                self.log_data = True
                self.classify = False
                subprocess.Popen(["espeak", "-v", "mb-en1", "Logging enabled!"])
                self.logfile = open(self.log_path, 'a') # append to file!
                time.sleep(1)

            if state['cross'] == True:
                print "cross pressed"
                self.log_data = False
                self.classify = False
                subprocess.Popen(["espeak", "-v", "mb-en1", "Logging disabled"])
		self.action = 3
                self.logfile.close()
		self.classyfile.close()
                time.sleep(1)
            if state['triangle'] == True:
                subprocess.Popen(["espeak", "-v", "mb-en1","Move Bitch, Get out the way!"])
                time.sleep(1)

            if state['square'] == True:
                print "square pressed"
                self.classify = True
                self.log_data = False
		self.classyfile =open('/media/ubuntu/9016-4EF8/transferLearning_data/classifierImages/classifierFile.log', 'a')
                subprocess.Popen(["espeak", "-v", "mb-en1", "Autonomous driving enabled"])
                time.sleep(1)

            #if state['trig2'] & state['circle'] == True:
            #    subprocess.Popen(["espeak", "-v", "mb-en1", "goodbye! Ha ha ha!"])
            #    self.image_sub.unregister()
            #    self.logfile.close()
            #    time.sleep(1)
            #    raise SystemExit, "Please shutdown!"


            if self.action == 0:
                self.command.linear.x = 0.4
                self.command.angular.z = 0
            elif self.action == 2:
                self.command.linear.x = 0
                self.command.angular.z = -1.4
		subprocess.Popen(["espeak", "-v", "mb-en1", "Right"])
            elif self.action == 1:
                self.command.linear.x = 0
                self.command.angular.z = 1.4
		subprocess.Popen(["espeak", "-v", "mb-en1", "Left"])

            if not self.classify:
                x = self.alpha * x + (1-self.alpha)*self.oldx
                self.oldx = x

                if x > 0.1:
                    z = z + -0.03
                self.command.linear.x = x

                self.command.angular.z = z
            self.pub.publish(self.command)

	    
            #print self.command.linear.x, self.command.angular.z
        except CvBridgeError as e:
            print e


def main(args):
    ic = camera_node()

    while 1:
        try:
        #if callbackCounter == 5:
         #   callbackCounter = 0

            ic.run()
        #rospy.spin()

        except (KeyboardInterrupt,SystemExit):
            print "Shutting down"
            raise
        except rospy.ROSInterruptException:
            print "error in the code"


        # cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
