#!/usr/bin/python
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
import time

import os

def main(args):

    while 1:
        try:
            state = sixaxis.get_state()
            x = state['lefty'] / 125.0  # Speed  [-1 to 1]
            z = state['rightx'] * 3.14 / 100.0  # turn
            if x > 0.5:
                print('x' + x)

        except (KeyboardInterrupt,SystemExit):
            print "Shutting down"
            raise

if __name__ == '__main__':
    main(sys.argv)
