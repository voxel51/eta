"""
Implementations of computer vision primitive algorithms.

Copyright 2017-2022, Voxel51, Inc.
voxel51.com
"""
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

import eta.core.image as etai
import eta.core.utils as etau
import eta.core.video as etav


class DenseOpticalFlow(object):
    """Base class for dense optical flow methods."""

    def __enter__(self):
        self.reset()
        return self

    def __exit__(self, *args):
        pass

    def process_video(
        self, input_path, cart_path=None, polar_path=None, video_path=None
    ):
        """Performs dense optical flow on the given video.

        Args:
            input_path: the input video path
            cart_path: an optional path to write the per-frame arrays as .npy
                files describing the flow fields in Cartesian (x, y)
                coordinates
            polar_path: an optional path to write the per-frame arrays as .npy
                files describing the flow fields in polar (magnitude, angle)
                coordinates
            video_path: an optional path to write a video that visualizes the
                magnitude and angle of the flow fields as the value (V) and
                hue (H), respectively, of per-frame HSV images
        """
        # Ensure output directories exist
        if cart_path:
            etau.ensure_basedir(cart_path)
        if polar_path:
            etau.ensure_basedir(polar_path)
        # VideoProcessor ensures that the output video directory exists

        self.reset()
        with etav.VideoProcessor(input_path, out_video_path=video_path) as p:
            for img in p:
                # Compute optical flow
                flow_cart = self.process_frame(img)

                if cart_path:
                    # Write Cartesian fields
                    np.save(cart_path % p.frame_number, flow_cart)

                if not polar_path and not video_path:
                    continue

                # Convert to polar coordinates
                flow_polar = cart_to_polar(flow_cart)

                if polar_path:
                    # Write polar fields
                    np.save(polar_path % p.frame_number, flow_polar)

                if video_path:
                    # Write flow visualization frame
                    p.write(polar_to_img(flow_polar))

    def process_frame(self, img):
        """Computes the dense optical flow field for the next frame.

        Args:
            img: an m x n x 3 image

        Returns:
            an m x n x 2 array containing the optical flow vectors
                in Cartesian (x, y) format
        """
        raise NotImplementedError("subclass must implement process_frame()")

    def reset(self):
        """Prepares the object to start processing a new video."""
        pass


def cart_to_polar(cart):
    """Converts the Cartesian vectors to polar coordinates.

    Args:
        cart: an m x n x 2 array describing vectors in Cartesian (x, y) format

    Returns:
        an m x n x 2 array describing vectors in polar
            (magnitude, angle) format
    """
    mag, ang = cv2.cartToPolar(cart[..., 0], cart[..., 1])
    return np.dstack((mag, ang))


def polar_to_img(polar):
    """Converts the polar coordinates into an image whose hue encodes the
    angle and value encodes the magnitude.

    Args:
        polar: an m x n x 2 array describing vectors in polar
            (magnitude, angle) format

    Returns:
        an image whose HSV colors encode the input polar coordinates
    """
    mag = polar[..., 0]
    ang = polar[..., 1]
    hsv = np.zeros(mag.shape + (3,), dtype=mag.dtype)
    hsv[..., 0] = (89.5 / np.pi) * ang  # [0, 179]
    hsv[..., 1] = 255
    # hsv[..., 2] = np.minimum(255 * mag, 255)  # [0, 255]
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # [0, 255]
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


class FarnebackDenseOpticalFlow(DenseOpticalFlow):
    """Computes dense optical flow on a video using Farneback's method.

    This class is a wrapper around the OpenCV `calcOpticalFlowFarneback`
    function.
    """

    def __init__(
        self,
        pyramid_scale=0.5,
        pyramid_levels=3,
        window_size=15,
        iterations=3,
        poly_n=7,
        poly_sigma=1.5,
        use_gaussian_filter=False,
    ):
        """Creates a FarnebackDenseOpticalFlow instance.

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
        """
        self.pyramid_scale = pyramid_scale
        self.pyramid_levels = pyramid_levels
        self.window_size = window_size
        self.iterations = iterations
        self.poly_n = poly_n
        self.poly_sigma = poly_sigma
        self.use_gaussian_filter = use_gaussian_filter

        self._prev_frame = None
        self._flags = (
            cv2.OPTFLOW_FARNEBACK_GAUSSIAN if use_gaussian_filter else 0
        )

    def process_frame(self, img):
        curr_frame = etai.rgb_to_gray(img)
        if self._prev_frame is None:
            # There is no previous frame for the first frame, so we set
            # it to the current frame, which implies that the flow for
            # the first frame will always be zero
            self._prev_frame = curr_frame

        # works in OpenCV 3 and OpenCV 2
        flow_cart = cv2.calcOpticalFlowFarneback(
            self._prev_frame,
            curr_frame,
            flow=None,
            pyr_scale=self.pyramid_scale,
            levels=self.pyramid_levels,
            winsize=self.window_size,
            iterations=self.iterations,
            poly_n=self.poly_n,
            poly_sigma=self.poly_sigma,
            flags=self._flags,
        )
        self._prev_frame = curr_frame

        return flow_cart

    def reset(self):
        self._prev_frame = None


class BackgroundSubtractor(object):
    """Base class for background subtraction methods."""

    def __enter__(self):
        self.reset()
        return self

    def __exit__(self, *args):
        pass

    def process_video(
        self,
        input_path,
        fgmask_path=None,
        fgvideo_path=None,
        bgvideo_path=None,
    ):
        """Performs background subtraction on the given video.

        Args:
            input_path: the input video path
            fgmask_path: an optional path to write the per-frame foreground
                masks (as boolean arrays) in .npy files
            fgvideo_path: an optional path to write the foreground-only video
            bgvideo_path: an optional path to write the background video
        """
        # Ensure output directories exist
        if fgmask_path:
            etau.ensure_basedir(fgmask_path)
        # VideoWriters ensure that the output video directories exist

        r = etav.FFmpegVideoReader(input_path)
        try:
            if fgvideo_path:
                fgw = etav.FFmpegVideoWriter(
                    fgvideo_path, r.frame_rate, r.frame_size
                )
            if bgvideo_path:
                bgw = etav.FFmpegVideoWriter(
                    bgvideo_path, r.frame_rate, r.frame_size
                )

            self.reset()
            for img in r:
                fgmask, bgimg = self.process_frame(img)

                if fgmask_path:
                    # Write foreground mask
                    fgmask_bool = fgmask.astype(np.bool)
                    np.save(fgmask_path % r.frame_number, fgmask_bool)

                if fgvideo_path:
                    # Write foreground-only video
                    fgw.write(apply_mask(img, fgmask))

                if bgvideo_path:
                    # Write background video
                    bgw.write(bgimg)
        finally:
            if fgvideo_path:
                fgw.close()
            if bgvideo_path:
                bgw.close()

    def process_frame(self, img):
        """Performs background subtraction on the next frame.

        Args:
            img: an image

        Returns:
            fgmask: the foreground mask
            bgimg: the background-only image
        """
        raise NotImplementedError("subclass must implement process_frame()")

    def reset(self):
        """Prepares the object to start processing a new video."""
        pass


def apply_mask(img, mask):
    """Applies the mask to the image.

    Args:
        img: an image
        mask: a mask image

    Returns:
        a copy of the input image with pixels outside the mask set to 0
    """
    mimg = img.copy()
    mimg[np.where(mask == 0)] = 0
    return mimg


class BackgroundSubtractorError(Exception):
    """Error raised when an error is encountered while performing background
    subtraction.
    """

    pass


class MOG2BackgroundSubtractor(BackgroundSubtractor):
    """Performs background subtraction on a video using Gaussian mixture-based
    foreground-background segmentation.

    This class is a wrapper around the OpenCV `BackgroundSubtractorMOG2` class.

    This model is only supported when using OpenCV 3.
    """

    def __init__(
        self,
        history=500,
        threshold=16.0,
        learning_rate=-1,
        detect_shadows=False,
    ):
        """Creates an MOG2BackgroundSubtractor instance.

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
        """
        self.history = history
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.detect_shadows = detect_shadows
        self._fgbg = None

    def process_frame(self, img):
        # We pass in an RGB image b/c this algo is invariant to channel order
        fgmask = self._fgbg.apply(img, None, self.learning_rate)
        bgimg = self._fgbg.getBackgroundImage()
        return fgmask, bgimg

    def reset(self):
        try:
            # OpenCV 3
            self._fgbg = cv2.createBackgroundSubtractorMOG2(
                history=self.history,
                varThreshold=self.threshold,
                detectShadows=self.detect_shadows,
            )
        except AttributeError:
            # OpenCV 2
            #
            # Note that OpenCV 2 does have a BackgroundSubtractorMOG2 class,
            # but background subtractors in OpenCV 2 don't support the
            # getBackgroundImage method, so they are not suitable for our
            # interface here
            #
            raise BackgroundSubtractorError(
                "BackgroundSubtractorMOG2 is not supported in OpenCV 2"
            )


class KNNBackgroundSubtractor(BackgroundSubtractor):
    """Performs background subtraction on a video using K-nearest
    neighbors-based foreground-background segmentation.

    This class is a wrapper around the OpenCV `BackgroundSubtractorKNN` class.

    This model is only supported when using OpenCV 3.
    """

    def __init__(
        self,
        history=500,
        threshold=400.0,
        learning_rate=-1,
        detect_shadows=False,
    ):
        """Creates a KNNBackgroundSubtractor instance.

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
        """
        self.history = history
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.detect_shadows = detect_shadows
        self._fgbg = None

    def process_frame(self, img):
        # We pass in an RGB image b/c this algo is invariant to channel order
        fgmask = self._fgbg.apply(img, None, self.learning_rate)
        bgimg = self._fgbg.getBackgroundImage()
        return fgmask, bgimg

    def reset(self):
        try:
            # OpenCV 3
            self._fgbg = cv2.createBackgroundSubtractorKNN(
                history=self.history,
                dist2Threshold=self.threshold,
                detectShadows=self.detect_shadows,
            )
        except AttributeError:
            # OpenCV 2
            raise BackgroundSubtractorError(
                "KNNBackgroundSubtractor is not supported in OpenCV 2"
            )


class EdgeDetector(object):
    """Base class for edge detection methods."""

    def __enter__(self):
        self.reset()
        return self

    def __exit__(self, *args):
        pass

    def process_video(self, input_path, masks_path=None, video_path=None):
        """Detect edges using self.detector.

        Args:
            input_path: the input video path
            masks_path: an optional path to write the per-frame edge masks (as
                boolean arrays) in .npy files
            video_path: an optional path to write the edges video
        """
        # Ensure output directories exist
        if masks_path:
            etau.ensure_basedir(masks_path)
        # VideoProcessor ensures that the output video directory exists

        self.reset()
        with etav.VideoProcessor(input_path, out_video_path=video_path) as p:
            for img in p:
                # Compute edges
                edges = self.process_frame(img)

                if masks_path:
                    # Write edges mask
                    edges_bool = edges.astype(np.bool)
                    np.save(masks_path % p.frame_number, edges_bool)

                if video_path:
                    # Write edges video
                    p.write(cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB))

    def process_frame(self, img):
        """Performs edge detection on the next frame.

        Args:
            img: an image

        Returns:
            the edges mask
        """
        raise NotImplementedError("subclass must implement process_frame()")

    def reset(self):
        """Prepares the object to start processing a new video."""
        pass


class CannyEdgeDetector(EdgeDetector):
    """The Canny edge detector.

    This class is a wrapper around the OpenCV `Canny` method.
    """

    def __init__(
        self, threshold1=200, threshold2=50, aperture_size=3, l2_gradient=False
    ):
        """Creates a CannyEdgeDetector instance.

        Args:
            threshold1 (200): the edge threshold
            threshold2 (50): the hysteresis threshold
            aperture_size (3): aperture size for the Sobel operator
            l2_gradient (False): whether to use a more accurate L2 norm to
                calculate the image gradient magnitudes
        """
        self.threshold1 = threshold1
        self.threshold2 = threshold2
        self.aperture_size = aperture_size
        self.l2_gradient = l2_gradient

    def process_frame(self, img):
        # works in OpenCV 3 and OpenCV 2
        gray = etai.rgb_to_gray(img)
        return cv2.Canny(
            gray,
            threshold1=self.threshold1,
            threshold2=self.threshold2,
            apertureSize=self.aperture_size,
            L2gradient=self.l2_gradient,
        )


class FeaturePointDetector(object):
    """Base class for feature point detection methods."""

    KEYPOINT_RGB_COLOR = (0, 255, 0)  # RGB

    def __enter__(self):
        self.reset()
        return self

    def __exit__(self, *args):
        pass

    def process_video(self, input_path, coords_path=None, video_path=None):
        """Detect feature points using self.detector.

        Args:
            input_path: the input video path
            masks_path: an optional path to write the per-frame feature points
                as .npy files
            video_path: an optional path to write the feature points video
        """
        # Ensure output directories exist
        if coords_path:
            etau.ensure_basedir(coords_path)
        # VideoProcessor ensures that the output video directory exists

        self.reset()
        with etav.VideoProcessor(input_path, out_video_path=video_path) as p:
            for img in p:
                # Compute feature points
                keypoints = self.process_frame(img)

                if coords_path:
                    # Write feature points to disk
                    pts = _unpack_keypoints(keypoints)
                    np.save(coords_path % p.frame_number, pts)

                if video_path:
                    # Write feature points video
                    # We pass in an RGB image b/c this function is invariant to
                    # channel order
                    img = cv2.drawKeypoints(
                        img, keypoints, None, color=self.KEYPOINT_RGB_COLOR
                    )
                    p.write(img)

    def process_frame(self, img):
        """Detects feature points in the next frame.

        Args:
            img: an image

        Returns:
            a list of `cv2.KeyPoint`s describing the detected features
        """
        raise NotImplementedError("subclass must implement process_frame()")

    def reset(self):
        """Prepares the object to start processing a new video."""
        pass


class HarrisFeaturePointDetector(FeaturePointDetector):
    """Detects Harris corners.

    This class is a wrapper around the OpenCV `cornerHarris` method.
    """

    def __init__(self, threshold=0.01, block_size=3, aperture_size=3, k=0.04):
        """Creates a HarrisEdgeDetector instance.

        Args:
            threshold (0.01): threshold (relative to the maximum detector
                response) to declare a corner
            block_size (3): the size of neighborhood used for the Harris
                operator
            aperture_size (3): aperture size for the Sobel derivatives
            k (0.04): Harris detector free parameter
        """
        self.threshold = threshold
        self.block_size = block_size
        self.aperture_size = aperture_size
        self.k = k

    def process_frame(self, img):
        # Works in OpenCV 3 and OpenCV 2
        gray = np.float32(etai.rgb_to_gray(img))
        response = cv2.cornerHarris(
            gray, blockSize=self.block_size, ksize=self.aperture_size, k=self.k
        )
        response = cv2.dilate(response, None)
        corners = response > self.threshold * response.max()
        return _pack_keypoints(np.argwhere(corners))


class FASTFeaturePointDetector(FeaturePointDetector):
    """Detects feature points using the FAST method.

    This class is a wrapper around the OpenCV `FastFeatureDetector` class.
    """

    def __init__(self, threshold=1, non_max_suppression=True):
        """Creates a FastFeatureDetector instance.

        Args:
            threshold (1): threshold on difference between intensity of the
                central pixel and pixels of a circle around this pixel
            non_max_suppression (True): whether to apply non-maximum
                suppression to the detected keypoints
        """
        self.threshold = threshold
        self.non_max_suppression = non_max_suppression
        try:
            # OpenCV 3
            self._detector = cv2.FastFeatureDetector(
                threshold=self.threshold,
                nonmaxSuppression=self.non_max_suppression,
            )
        except AttributeError:
            # OpenCV 2
            self._detector = cv2.FastFeatureDetector_create(
                threshold=self.threshold,
                nonmaxSuppression=self.non_max_suppression,
            )

    def process_frame(self, img):
        # We pass in an RGB image b/c this algo is invariant to channel order
        return self._detector.detect(img, None)


class ORBFeaturePointDetector(FeaturePointDetector):
    """Detects feature points using the ORB (Oriented FAST and rotated BRIEF
    features) method.

    This class is a wrapper around the OpenCV `ORB` class.
    """

    def __init__(self, max_num_features=500, score_type=cv2.ORB_HARRIS_SCORE):
        """Creates a ORBFeaturePointDetector instance.

        Args:
            max_num_features (500): the maximum number of features to retain
            score_type (cv2.ORB_HARRIS_SCORE): the algorithm used to rank
                features. The choices are `cv2.ORB_HARRIS_SCORE` and
                `cv2.FAST_SCORE`
        """
        self.max_num_features = max_num_features
        self.score_type = score_type
        try:
            # OpenCV 3
            self._detector = cv2.ORB_create(
                nfeatures=self.max_num_features, scoreType=self.score_type
            )
        except AttributeError:
            # OpenCV 2
            self._detector = cv2.ORB(
                nfeatures=self.max_num_features, scoreType=self.score_type
            )

    def process_frame(self, img):
        # We pass in an RGB image b/c this algo is invariant to channel order
        return self._detector.detect(img, None)


def _pack_keypoints(pts):
    """Pack the points into a list of `cv2.KeyPoint`s.

    Args:
        pts: an n x 2 array of [row, col] coordinates

    Returns:
        a list of `cv2.KeyPoint`s
    """
    return [cv2.KeyPoint(x[1], x[0], 1) for x in pts]


def _unpack_keypoints(keypoints):
    """Unpack the keypoints into an array of coordinates.

    Args:
        keypoints: a list of `cv2.KeyPoint`s

    Returns:
        an n x 2 array of [row, col] coordinates
    """
    return np.array([[kp.pt[1], kp.pt[0]] for kp in keypoints])
