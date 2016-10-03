import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import sys
from os.path import expanduser

caffe_root = '~/caffe'
sys.path.insert(0, caffe_root + 'python')


class camera_node:
    def __init__(self):
        rospy.init_node('camera_node')
        self.image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.callback)
        self.bridge = CvBridge()
        self.logfile = open('controls.log', 'w')
        self.logfile.write("x; z; image")

        self.imgnr = 0
        print "camera_node.py running..."
        home = expanduser("~")

    def __del__(self):
        self.logfile.close()

    def callback(self, data):
        cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        filename = 'collected_data/colorImg_%s.png' % self.imgnr
        cv2.imwrite(filename, cv_image)
        self.imgnr += 1
        buf = "%f; %f; colorImg_%s\n" % (0, 0, self.imgnr)
        self.logfile.write(buf)
        print filename


def main(args):
    ic = camera_node()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting down"


if __name__ == '__main__':
    main(sys.argv)
