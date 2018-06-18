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

import eta.core.utils as etau
import eta.core.video as etav


class DenseOpticalFlow(object):
    '''Base class for dense optical flow methods.'''

    def process(
            self, input_path, cart_path=None, polar_path=None, vid_path=None):
        '''Performs dense optical flow on the given video.

        Args:
            input_path: the input video path
            cart_path: an optional path to write the per-frame arrays as .npy
                files describing the flow fields in Cartesian (x, y)
                coordinates
            polar_path: an optional path to write the per-frame arrays as .npy
                files describing the flow fields in polar (magnitude, angle)
                coordinates
            vid_path: an optional path to write a video that visualizes the
                magnitude and angle of the flow fields as the value (V) and
                hue (H), respectively, of per-frame HSV images
        '''
        # Ensure output directories exist
        if cart_path:
            etau.ensure_basedir(cart_path)
        if polar_path:
            etau.ensure_basedir(polar_path)
        # VideoProcessor ensures that the output video directory exists

        self._reset()
        with etav.VideoProcessor(input_path, out_single_vidpath=vid_path) as p:
            for img in p:
                # Compute optical flow
                flow_cart = self._process_frame(img)

                if cart_path:
                    # Write Cartesian fields
                    np.save(cart_path % p.frame_number, flow_cart)

                if not polar_path and not vid_path:
                    continue

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

    def _process_frame(self, img):
        '''Processes the next frame.

        Args:
            img: the next frame

        Returns:
            flow: the optical flow for the frame in Cartesian coordinates
        '''
        raise NotImplementedError("subclass must implement _process_frame()")

    def _reset(self):
        '''Prepares the object to start processing a new video.'''
        pass


def _polar_flow_to_img(mag, ang):
    hsv = np.zeros(mag.shape + (3,), dtype=mag.dtype)
    hsv[..., 0] = (89.5 / np.pi) * ang  # [0, 179]
    hsv[..., 1] = 255
    #hsv[..., 2] = np.minimum(255 * mag, 255)  # [0, 255]
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # [0, 255]
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


class FarnebackDenseOpticalFlow(DenseOpticalFlow):
    '''Computes dense optical flow on a video using Farneback's method.

    This class is a wrapper around the OpenCV `calcOpticalFlowFarneback`
    function.
    '''

    def __init__(
            self,
            pyramid_scale=0.5,
            pyramid_levels=3,
            window_size=15,
            iterations=3,
            poly_n=7,
            poly_sigma=1.5,
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
            poly_n (7): size of the pixel neighborhood used to find polynomial
                expansion in each pixel
            poly_sigma (1.5): standard deviation of the Gaussian that is used
                to smooth derivatives
            use_gaussian_filter (False): whether to use a Gaussian filter
                instead of a box filer
        '''
        self.pyramid_scale = pyramid_scale
        self.pyramid_levels = pyramid_levels
        self.window_size = window_size
        self.iterations = iterations
        self.poly_n = poly_n
        self.poly_sigma = poly_sigma
        self.use_gaussian_filter = use_gaussian_filter

        self._prev_frame = None
        self._flags = (
            cv2.OPTFLOW_FARNEBACK_GAUSSIAN if use_gaussian_filter else 0)

    def _process_frame(self, img):
        curr_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if self._prev_frame is None:
            # There is no previous frame for the first frame, so we set
            # it to the current frame, which implies that the flow for
            # the first frame will always be zero
            self._prev_frame = curr_frame

        # works in OpenCV 3 and OpenCV 2
        flow_cart = cv2.calcOpticalFlowFarneback(
            self._prev_frame, curr_frame, flow=None,
            pyr_scale=self.pyramid_scale, levels=self.pyramid_levels,
            winsize=self.window_size, iterations=self.iterations,
            poly_n=self.poly_n, poly_sigma=self.poly_sigma,
            flags=self._flags)
        self._prev_frame = curr_frame

        return flow_cart

    def _reset(self):
        self._prev_frame = None


class BackgroundSubtractor(object):
    '''Base class for background subtraction methods.'''

    def process(
            self, input_path, fgmask_path=None, fgvid_path=None,
            bgvid_path=None):
        '''Performs background subtraction on the given video.

        Args:
            input_path: the input video path
            fgmask_path: an optional path to write the per-frame foreground
                masks as .npy files
            fgvid_path: an optional path to write the foreground-only video
            bgvid_path: an optional path to write the background video
        '''
        # Ensure output directories exist
        if fgmask_path:
            etau.ensure_basedir(fgmask_path)
        # VideoWriters ensure that the output video directories exist

        r = etav.FFmpegVideoReader(input_path)
        try:
            if fgvid_path:
                fgw = etav.FFmpegVideoWriter(
                    fgvid_path, r.frame_rate, r.frame_size)
            if bgvid_path:
                bgw = etav.FFmpegVideoWriter(
                    bgvid_path, r.frame_rate, r.frame_size)

            self._reset()
            for img in r:
                fgmask, bgimg = self._process_frame(img)

                if fgmask_path:
                    # Write foreground mask
                    np.save(fgmask_path % r.frame_number, fgmask)

                if fgvid_path:
                    # Write foreground-only video
                    img[np.where(fgmask == 0)] = 0
                    fgw.write(img)

                if bgvid_path:
                    # Write background video
                    bgw.write(bgimg)
        finally:
            if fgvid_path:
                fgw.close()
            if bgvid_path:
                bgw.close()

    def _process_frame(self, img):
        '''Processes the next frame.

        Args:
            img: the next frame

        Returns:
            fgmask: the foreground mask
            bgimg: the background-only image
        '''
        raise NotImplementedError("subclass must implement _process_frame()")

    def _reset(self):
        '''Prepares the object to start processing a new video.'''
        pass


class BackgroundSubtractorError(Exception):
    '''Error raised when an error is encountered while performing background
    subtraction.
    '''
    pass


class MOG2BackgroundSubtractor(BackgroundSubtractor):
    '''Performs background subtraction on a video using Gaussian mixture-based
    foreground-background segmentation.

    This class is a wrapper around the OpenCV `BackgroundSubtractorMOG2` class.

    This model is only supported when using OpenCV 3.
    '''

    def __init__(
            self, history=500, threshold=16.0, learning_rate=-1,
            detect_shadows=False):
        '''Initializes an MOG2BackgroundSubtractor object.

        Args:
            history (500): the number of previous frames that affect the
                background model
            threshold (16.0): threshold on the squared Mahalanobis distance
                between pixel and the model to decide whether a pixel is well
                described by the background model
            learning_rate (-1): a value between 0 and 1 that indicates how fast
                the background model is learnt, where 0 means that the
                background model is not updated at all and 1 means that the
                background model is completely reinitialized from the last
                frame. If a negative value is provided, an automatically chosen
                learning rate will be used
            detect_shadows (True): whether to detect and mark shadows
        '''
        self.history = history
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.detect_shadows = detect_shadows
        self._fgbg = None

    def _process_frame(self, img):
        fgmask = self._fgbg.apply(img, None, self.learning_rate)
        bgimg = self._fgbg.getBackgroundImage()
        return fgmask, bgimg

    def _reset(self):
        try:
            # OpenCV 3
            self._fgbg = cv2.createBackgroundSubtractorMOG2(
                history=self.history, varThreshold=self.threshold,
                detectShadows=self.detect_shadows)
        except AttributeError:
            # OpenCV 2
            #
            # Note that OpenCV 2 does have a BackgroundSubtractorMOG2 class,
            # but background subtractors in OpenCV 2 don't support the
            # getBackgroundImage method, so they are not suitable for our
            # interface here
            #
            raise BackgroundSubtractorError(
                "BackgroundSubtractorMOG2 is not supported in OpenCV 2")


class KNNBackgroundSubtractor(BackgroundSubtractor):
    '''Performs background subtraction on a video using K-nearest
    neighbors-based foreground-background segmentation.

    This class is a wrapper around the OpenCV `BackgroundSubtractorKNN` class.

    This model is only supported when using OpenCV 3.
    '''

    def __init__(
            self, history=500, threshold=400.0, learning_rate=-1,
            detect_shadows=False):
        '''Initializes an KNNBackgroundSubtractor object.

        Args:
            history (500): length of the history
            threshold (400.0): threshold on the squared distance between pixel
                and the sample to decide whether a pixel is close to that
                sample
            learning_rate (-1): a value between 0 and 1 that indicates how fast
                the background model is learnt, where 0 means that the
                background model is not updated at all and 1 means that the
                background model is completely reinitialized from the last
                frame. If a negative value is provided, an automatically chosen
                learning rate will be used
            detect_shadows (True): whether to detect and mark shadows
        '''
        self.history = history
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.detect_shadows = detect_shadows
        self._fgbg = None

    def _process_frame(self, img):
        fgmask = self._fgbg.apply(img, None, self.learning_rate)
        bgimg = self._fgbg.getBackgroundImage()
        return fgmask, bgimg

    def _reset(self):
        try:
            # OpenCV 3
            self._fgbg = cv2.createBackgroundSubtractorKNN(
                history=self.history, dist2Threshold=self.threshold,
                detectShadows=self.detect_shadows)
        except AttributeError:
            # OpenCV 2
            raise BackgroundSubtractorError(
                "KNNBackgroundSubtractor is not supported in OpenCV 2")


class EdgeDetector(object):
    '''Base class for edge detection methods.'''

    def process(self, input_path, masks_path=None, vid_path=None):
        '''Detect edges using self.detector.

        Args:
            input_path: the input video path
            masks_path: an optional path to write the per-frame edge masks as
                .npy files
            vid_path: an optional path to write the edges video
        '''
        # Ensure output directories exist
        if masks_path:
            etau.ensure_basedir(masks_path)
        # VideoProcessor ensures that the output video directory exists

        self._reset()
        with etav.VideoProcessor(input_path, out_single_vidpath=vid_path) as p:
            for img in p:
                # Compute edges
                edges = self._process_frame(img)

                if masks_path:
                    # Write edges mask
                    np.save(masks_path % p.frame_number, edges.astype(np.bool))

                if vid_path:
                    # Write edges video
                    p.write(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR))

    def _process_frame(self, img):
        '''Processes the next frame.

        Args:
            img: the next frame

        Returns:
            edges: the edges image
        '''
        raise NotImplementedError("subclass must implement _process_frame()")

    def _reset(self):
        '''Prepares the object to start processing a new video.'''
        pass


class CannyEdgeDetector(EdgeDetector):
    '''The Canny edge detector.

    This class is a wrapper around the OpenCV `Canny` method.
    '''

    def __init__(
            self, threshold1=200, threshold2=50, aperture_size=3,
            l2_gradient=False):
        '''Creates a new CannyEdgeDetector object.

        Args:
            threshold1 (200): the edge threshold
            threshold2 (50): the hysteresis threshold
            aperture_size (3): aperture size for the Sobel operator
            l2_gradient (False): whether to use a more accurate L2 norm to
                calculate the image gradient magnitudes
        '''
        self.threshold1 = threshold1
        self.threshold2 = threshold2
        self.aperture_size = aperture_size
        self.l2_gradient = l2_gradient

    def _process_frame(self, img):
        # works in OpenCV 3 and OpenCV 2
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.Canny(
            gray, threshold1=self.threshold1, threshold2=self.threshold2,
            apertureSize=self.aperture_size, L2gradient=self.l2_gradient)


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
        with etav.VideoProcessor(input_path, out_single_vidpath=vid_path) as processor:
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
        with etav.VideoProcessor(input_path, out_impath=im_path, out_single_vidpath=vid_path) as processor:
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
        with etav.VideoProcessor(input_path, out_single_vidpath=vid_path) as processor:
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
