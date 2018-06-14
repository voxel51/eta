'''
Core low-level primitive operations.

Copyright 2018, Voxel51, LLC
voxel51.com

Kunyi Lu, kunyi@voxel51.com
'''
import cv2
import numpy as np


import eta.core.video as etav


class DenseOpticalFlow(object):
    '''A class for processing dense optical flow to a video.'''

    def __init__(self, 
                 initial_flow=None,
                 pyramid_scale=None,
                 pyramid_levels=None,
                 window_size=None,
                 iterations=None,
                 poly_n=None,
                 poly_sigma=None,
                 flag=None):
        '''Initiate the parameters for DenseOpticalFlow class

        Args:
            initial_flow: initialization for optical flow
            pyramid_scale: the image scale(<1) to build pyramids for each image
            pyramid_levels: number of pyramid layers including the initial image
            window_size: averaging window size
            iterations: number of iterations the algorithm does at each pyramid level
            poly_n: size of the pixel neighborhood used to find polynomial expansion in each pixel
            poly_sigma: standard deviation of the Gaussian that is used to smooth derivatives
                        used as a basis for the polynomial expansion
            flag: operation flags including cv2.OPTFLOW_USE_INITIAL_FLOW and
                  cv2.OPTFLOW_FARNEBACK_GAUSSIAN
        '''
        if pyramid_scale is None:
            pyramid_scale = 0.5

        if pyramid_levels is None:
            pyramid_levels = 3

        if window_size is None:
            window_size = 15

        if iterations is None:
            iterations = 3

        if poly_n is None:
            poly_n = 5

        if poly_sigma is None:
            poly_sigma = 1.2

        if flag is None:
            flag = 0

        self.initial_flow = initial_flow
        self.pyramid_scale = pyramid_scale
        self.pyramid_levels = pyramid_levels
        self.window_size = window_size
        self.iterations = iterations
        self.poly_n = poly_n
        self.poly_sigma = poly_sigma
        self.flag = flag

    def process(self,
                input_path,
                output_format,
                cartesian_path=None,
                polar_path=None,
                vid_path=None,):
        '''Compute dense optical flow by Gunnar Farneback algorithm.
        
        Args:
            input_path: the path of the video to be processed
            output_format: a list of output format from ["cartesian", "polar", "vid"]
            cartesian_path: the path for cartesian format output
            polar_path: the path for polar format output
            vid_path: the output path for processed video
        '''

        with etav.VideoProcessor(input_path, out_vidpath=vid_path) as processor:
            for img in processor:
                current_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                if processor.is_new_frame_range is False:
                    opt_flow_cart = cv2.calcOpticalFlowFarneback(previous_frame,
                                                                 current_frame,
                                                                 self.initial_flow,
                                                                 self.pyramid_scale,
                                                                 self.pyramid_levels,
                                                                 self.window_size,
                                                                 self.iterations,
                                                                 self.poly_n,
                                                                 self.poly_sigma,
                                                                 self.flag)                
                    mag, ang = cv2.cartToPolar(opt_flow_cart[...,0], opt_flow_cart[...,1])
                    opt_flow_polar = np.dstack((mag, ang))
                    if "cartesian" in output_format:
                        np.save(cartesian_path % processor.frame_number, opt_flow_cart)
                    if "polar" in output_format:
                        np.save(polar_path % processor.frame_number, opt_flow_polar)
                    hsv = np.zeros_like(img)
                    hsv[..., 1] = 255
                    hsv[...,0] = ang * 180 / np.pi / 2
                    hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                    if "vid" in output_format:
                        processor.write(bgr)
                previous_frame = current_frame


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
