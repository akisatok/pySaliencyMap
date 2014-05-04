#-------------------------------------------------------------------------------
# Name:        main
# Purpose:     Testing the package pySaliencyMap
#
# Author:      Akisato Kimura <akisato@ieee.org>
#
# Created:     04/05/2014
# Copyright:   (c) Akisato Kimura 2014-
# Licence:     All rights reserved
#-------------------------------------------------------------------------------

import cv2
import numpy as np
import pySaliencyMap

# main
if __name__ == '__main__':
    img = cv2.imread('test.jpg')
    imgsize = img.shape
    img_width  = imgsize[1]
    img_height = imgsize[0]
    sm = pySaliencyMap.pySaliencyMap(img_width, img_height)
    map = sm.SMGetSM(img)
    cv2.imshow("input",  img)
    cv2.imshow("output", map)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


