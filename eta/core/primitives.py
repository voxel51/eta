'''
Implementations of computer vision primitive algorithms.

Copyright 2018, Voxel51, LLC
voxel51.com

Kunyi Lu, kunyi@voxel51.com
Brian Moore, brian@voxel51.com
'''
# pragma pylint: disable=redefined-builtin
# pragma pylint: disable=unused-wildcard-import
# pragma pylint: disable=wildcard-import
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from builtins import *
# pragma pylint: enable=redefined-builtin
# pragma pylint: enable=unused-wildcard-import
# pragma pylint: enable=wildcard-import

import cv2
import numpy as np

import eta.core.video as etav


class DenseOpticalFlow(object):
    '''Base class for dense optical flow methods.'''

    def __init__(self, _flow):
        '''Initializes the base DenseOpticalFlow object.

        Args:
            _flow: a function that accepts the previous and current frames and
                returns an m x n x 2 array containing the dense optical flow
                field for the current frame expressed in Cartesian (x, y)
                coordinates
        '''
        self._flow = _flow
        self._prev_frame = None

    def process(
            self, input_path, cart_path=None, polar_path=None, vid_path=None):
        '''Performs dense optical flow on the given video.

        Args:
            input_path: the input video path
            cart_path: an optional path to write the per-frame arrays
                describing the flow fields in Cartesian (x, y) coordinates
            polar_path: an optional path to write the per-frame arrays
                describing the flow fields in polar (magnitude, angle)
                coordinates
            vid_path: an optional path to write a video that visualizes the
                magnitude and angle of the flow fields as the value (V) and
                hue (H), respectively, of per-frame HSV images
        '''
        with etav.VideoProcessor(input_path, out_vidpath=vid_path) as p:
            for img in p:
                # Compute flow
                curr_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                flow_cart = self._flow(self._prev_frame, curr_frame)
                self._prev_frame = curr_frame

                if cart_path:
                    # Write Cartesian fields
                    np.save(cart_path % p.frame_number, flow_cart)

                if not polar_path and not vid_path:
                    continue;

                # Convert to polar coordinates
                mag, ang = cv2.cartToPolar(
                    flow_cart[..., 0], flow_cart[..., 1])
                flow_polar = np.dstack((mag, ang))

                if polar_path:
                    # Write polar fields
                    np.save(polar_path % p.frame_number, flow_polar)

                if vid_path:
                    # Write flow visualization frame
                    p.write(_polar_flow_to_img(mag, ang))


def _polar_flow_to_img(mag, ang):
    hsv = np.zeros(mag.shape + (3,), dtype=mag.dtype)
    hsv[..., 0] = (90.0 / np.pi) * ang
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


class FarnebackDenseOpticalFlow(DenseOpticalFlow):
    '''A class that computes dense optical flow on a video using Gunnar
    Farneback’s algorithm.
    '''

    def __init__(
            pyramid_scale=0.5,
            pyramid_levels=3,
            window_size=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.1,
            use_gaussian_filter=False):
        '''Constructs a FarnebackDenseOpticalFlow object.

        Args:
            pyramid_scale (0.5): the image scale (<1) to build pyramids for
                each image
            pyramid_levels (3): number of pyramid layers including the initial
                image
            window_size (15): averaging window size
            iterations (3): number of iterations to perform at each pyramid
                level
            poly_n (5): size of the pixel neighborhood used to find polynomial
                expansion in each pixel
            poly_sigma (1.1): standard deviation of the Gaussian that is used
                to smooth derivatives
            use_gaussian_filter (False): whether to use a Gaussian filter
                instead of a box filer
        '''
        flags = cv2.OPTFLOW_FARNEBACK_GAUSSIAN if use_gaussian_filter else 0

        def _flow(prev, curr):
            return cv2.calcOpticalFlowFarneback(
                prev, curr, pyr_scale=pyramid_scale, levels=pyramid_levels,
                winsize=window_size, iterations=iterations, poly_n=poly_n,
                poly_sigma=poly_sigma, flags=flags)

        super(FarnebackDenseOpticalFlow, self).__init__(_flow)


class BackgroundSubtractor(object):
    '''A class for processing adaptive background subtraction to a video.'''

    def __init__(self, fgbg):
        '''Initiate the parameters for BackgroundSubtraction class

        Args:
            fgbg: an instance of a certain background subtractor
        '''
        self.fgbg = fgbg

    def process(self, input_path, output_format, npy_path=None, vid_path=None):
        '''Process the video using the self.fgbg.
        
        Args:
            input_path: the path of the video to be processed
            output_format: a list of output format from ["npy", "vid"]
            npy_path: the output path for npy format foreground
            vid_path: the output path for processed video
        '''
        with etav.VideoProcessor(input_path, out_vidpath=vid_path) as processor:
            for img in processor:
                fgmask = self.fgbg.apply(img)
                img[np.where(fgmask == 0)] = 0
                if "npy" in output_format:
                    np.save(npy_path % processor.frame_number, img)
                if "vid" in output_format:
                    processor.write(img)


class MOGBackgroundSubtractor(BackgroundSubtractor):
    '''A class for processing adaptive background subtraction to a video
       using Gaussian mixtrue-base algorithm.'''

    def __init__(self, history=500, threshold=16, detect_shadows=True, **kwargs):
        '''Initialize variable used by MOGBackgroundSubtractor class.

        Args:
            history: length of the history
            threshold: threshold on the squared Mahalanobis distance between pixel and
                       the model to decide whether a pixel is well described by the
                       background model.
            detect_shadows: if true, the algorithm will detect shadows and mark them
            **kwargs: valid keyword arguments for BackgroundSubtractor
        '''
        fgbg = cv2.createBackgroundSubtractorMOG2(history=history,
                                                  varThreshold=threshold,
                                                  detectShadows=detect_shadows)
        super(MOGBackgroundSubtractor, self).__init__(fgbg, **kwargs)


class KNNBackgroundSubtractor(BackgroundSubtractor):
    '''A class for processing adaptive background subtraction to a video
       using KNN algorithm.'''

    def __init__(self, history=500, threshold=400.0, detect_shadows=True, **kwargs):
        '''Initialize variable used by MOGBackgroundSubtractor class.

        Args:
            history: length of the history
            threshold: threshold on the squared distance between pixel and
                       the sample to decide whether a pixel is close to that sample.
            detect_shadows: if true, the algorithm will detect shadows and mark them
            **kwargs: valid keyword arguments for BackgroundSubtractor
        '''
        fgbg = cv2.createBackgroundSubtractorKNN(history=history,
                                                 dist2Threshold=threshold,
                                                 detectShadows=detect_shadows)
        super(KNNBackgroundSubtractor, self).__init__(fgbg, **kwargs)


class EdgeDetector(object):
    '''A class for processing edge detection to a video'''

    def __init__(self, detector):
        '''Initiate the parameters for EdgeDetection class

        Args:
            detector: an instance of a certain edge detector
        '''
        self.detector = detector

    def process(self, input_path, im_path=None, vid_path=None):
        '''Detect edges using self.detector.
        
        Args:
            input_path: the path of the video to be processed
            im_path: the output path for each frame image
            vid_path: the output path for processed video
        '''
        with etav.VideoProcessor(input_path, out_impath=im_path, out_vidpath=vid_path) as processor:
            for img in processor:
                edge = self.detector(img)
                edge_bgr = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)
                processor.write(edge_bgr)


class CannyEdgeDetector(EdgeDetector):

    def __init__(self, threshold_1=100, threshold_2=100, aperture_size=3, l2_gradient=False, **kwargs):
        '''Initialize variable used by CannyEdgeDetector class.

        Args:
            threshold_1: first threshold for the hysteresis procedure
            threshold_2: second threshold for the hysteresis procedure
            aperture_size: aperture size for the Sobel operator
            l2_gradient: a flag, indicating whether a more accurate L2 norm 
                         should be used to calculate the image gradient magnitude
            **kwargs: valid keyword arguments for EdgeDetector
        '''
        detector = (lambda img: cv2.Canny(img,
                                          threshold1=threshold_1,
                                          threshold2=threshold_2,
                                          apertureSize=aperture_size,
                                          L2gradient=l2_gradient))
        super(CannyEdgeDetector, self).__init__(detector)


class FeaturePointDetector(object):
    '''A class for detecting feature points in a video.'''

    def __init__(self, detector):
        '''Initiate the parameters for FeaturePointDetector class

        Args:
            detector: an instance of a certain feature point detector
        '''
        self.detector = detector

    def process(self, input_path, output_format, coor_path=None, vid_path=None):
        '''Detect feature points using self.detector.
        
        Args:
            input_path: the path of the video to be processed
            output_format: a list of output format from ["point_coor", "vid"]
            coor_path: the output path for coordinates of feature points in each frame
            vid_path: the output path for processed video
        '''
        with etav.VideoProcessor(input_path, out_vidpath=vid_path) as processor:
            point_coor = []
            for img in processor:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                corner = self.detector.detect(gray, None)
                img = cv2.drawKeypoints(gray, corner, img, color=(255,0,0))
                if "point_coor" in output_format:
                    for i in corner:
                        point_coor.append(i.pt)
                    np.save(coor_path % processor.frame_number, point_coor)
                if "vid" in output_format:
                    processor.write(img)


class HarrisFeaturePointDetector(FeaturePointDetector):

    def __init__(self, block_size=2, aperture_size=3, k=0.04, **kwargs):
        '''Initialize variable used by HarrisFeaturePointDetector class.

        Args:
            block_size: the size of neighbourhood considered for corner detection
            aperture_size: aperture parameter of Sobel derivative used
            k: Harris detector free parameter
            **kwargs: valid keyword arguments for FeaturePointDetector
        '''
        detector = lambda img: cv2.cornerHarris(img,
                                                blockSize=block_size,
                                                ksize=aperture_size,
                                                k=k)
        super(HarrisFeaturePointDetector, self).__init__(detector)

    def process(self, input_path, im_path=None, vid_path=None):
        '''Detect feature points using self.detector.

        Args:
            input_path: the path of the video to be processed
            im_path: the output path for each frame image
            vid_path: the output path for processed video
        '''
        with etav.VideoProcessor(input_path, out_impath=im_path, out_vidpath=vid_path) as processor:
            for img in processor:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray = np.float32(gray)
                corner = self.detector(gray)                
                img[corner > 0.01 * corner.max()] = [0, 0, 255]
                processor.write(img)


class FASTFeaturePointDetector(FeaturePointDetector):

    def __init__(self, threshold=1, nonmaxSuppression=True, type=cv2.FAST_FEATURE_DETECTOR_TYPE_9_16, **kwargs):
        '''Initialize variable used by FASTFeaturePointDetector class.

        Args:
            threshold: threshold on difference between intensity of the
                       central pixel and pixels of a circle around this pixel
            nonmaxSuppression: if true, non-maximum suppression is applied to 
                               detected corners (keypoints).
            type: one of the three neighborhoods as defined in the paper
            **kwargs: valid keyword arguments for FeaturePointDetector
        '''
        detector = cv2.FastFeatureDetector_create()
        super(FASTFeaturePointDetector, self).__init__(detector)


class FeaturePointDescriptor(object):
    '''A class for detecting feature points and computing its 
       feature vector in a video.'''

    def __init__(self, descriptor):
        '''Initiate the parameters for FeaturePointDescriptor class

        Args:
            descriptor: an instance of a certain feature point descriptor
        '''
        self.descriptor = descriptor

    def process(self, input_path, output_format, coor_path=None, vid_path=None):
        '''Detect feature points using self.detector.
        
        Args:
            input_path: the path of the video to be processed
            output_format: a list of output format from ["point_coor", "vid"]
            coor_path: the output path for coordinates of feature points in each frame
            vid_path: the output path for processed video
        '''
        with etav.VideoProcessor(input_path, out_vidpath=vid_path) as processor:
            point_coor = []
            for img in processor:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                key_point, des = self.descriptor.detectAndCompute(gray, None)
                img = cv2.drawKeypoints(gray, key_point, img, color=(255,0,0))
                if "point_coor" in output_format:
                    for i,j in zip(key_point, des):
                        point_coor.append([i.pt, j])
                    np.save(coor_path % processor.frame_number, point_coor)
                if "vid" in output_format:
                    processor.write(img)


class ORBFeaturePointDescriptor(FeaturePointDescriptor):

    def __init__(self, num_features=1000, **kwargs):
        '''Initialize variable used by ORBFeaturePointDetector class.

        Args:
            num_features: maximum number of features to be retained
            **kwargs: valid keyword arguments for FeaturePointDetector
        '''
        descriptor = cv2.ORB_create(num_features)
        super(ORBFeaturePointDescriptor, self).__init__(descriptor)
