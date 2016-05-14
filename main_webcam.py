#-------------------------------------------------------------------------------
# Name:        main_webcam
# Purpose:     Testing the package pySaliencyMap with your own webcams
#
# Author:      Akisato Kimura <akisato@ieee.org>
#
# Created:     May 14, 2016
# Copyright:   (c) Akisato Kimura 2016-
# Licence:     All rights reserved
#-------------------------------------------------------------------------------

import cv2
import matplotlib.pyplot as plt
import pySaliencyMap

# main
if __name__ == '__main__':
    # set up webcams
    capture = cv2.VideoCapture(0)
    # repeat until pressing a key "q"
    while(True):
        # capture
        retval, frame = capture.read()
        # initialize
        frame_size = frame.shape
        frame_width  = frame_size[1]
        frame_height = frame_size[0]
        sm = pySaliencyMap.pySaliencyMap(frame_width, frame_height)
        # computation
        saliency_map = sm.SMGetSM(frame)
#        binarized_map = sm.SMGetBinarizedSM(frame)
#        salient_region = sm.SMGetSalientRegion(frame)
        # visualize
        cv2.imshow('Input image', cv2.flip(frame, 1))
        cv2.imshow('Saliency map', cv2.flip(saliency_map, 1))
#        cv2.imshow('Binalized saliency map', cv2.flip(binarized_map, 1))
#        cv2.imshow('Salient region', cv2.flip(salient_region, 1))
        # exit if the key "q" is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
