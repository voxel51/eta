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
                 input_path,
                 output_path,
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
            input_path: the path of the video to be processed
            output_path: output path to save the flow of each frame
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

        self.input_path = input_path
        self.output_path = output_path
        self.initial_flow = initial_flow
        self.pyramid_scale = pyramid_scale
        self.pyramid_levels = pyramid_levels
        self.window_size = window_size
        self.iterations = iterations
        self.poly_n = poly_n
        self.poly_sigma = poly_sigma
        self.flag = flag

    def compute_flow(self):
        '''Compute dense optical flow by Gunnar Farneback algorithm.'''

        with etav.VideoProcessor(self.input_path) as processor:
            for img in processor:
                current_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                if processor.is_new_frame_range is False:
                    opt_flow = cv2.calcOpticalFlowFarneback(previous_frame,
                                                            current_frame,
                                                            self.initial_flow,
                                                            self.pyramid_scale,
                                                            self.pyramid_levels,
                                                            self.window_size,
                                                            self.iterations,
                                                            self.poly_n,
                                                            self.poly_sigma,
                                                            self.flag)                    
                    np.save(self.output_path % processor.frame_number, opt_flow)
                previous_frame = current_frame


class BackgroundSubtraction(object):
    '''A class for processing adaptive background subtraction to a video.'''

    def __init__(self, input_path, output_path):
        '''Initiate the parameters for BackgroundSubtraction class

        Args:
            input_path: the path of the video to be processed
            output_path: output path to save the flow of each frame
        '''
        self.input_path = input_path
        self.output_path = output_path

    def background_subtractor_KNN(self, history=500, threshold=400.0, detect_shadows=True):
        '''Compute background subtractor by KNN algorithm.
        
        Args:
            history: length of the history
            threshold: threshold on the squared distance between pixel and
                       the sample to decide whether a pixel is close to that sample.
            detect_shadows: if true, the algorithm will detect shadows and mark them
        '''
        fgbg = cv2.createBackgroundSubtractorKNN(history=history,
                                                 dist2Threshold=threshold,
                                                 detectShadows=detect_shadows)
        with etav.VideoProcessor(self.input_path, out_impath=self.output_path) as processor:
            for img in processor:
                fgmask = fgbg.apply(img)
                processor.write(fgmask)

    def background_subtractor_MOG(self, history=500, threshold=16, detect_shadows=True):
        '''Compute background subtractor by Gaussian mixtrue-base algorithm.
        
        Args:
            history: length of the history
            threshold: threshold on the squared Mahalanobis distance between pixel and
                       the model to decide whether a pixel is well described by the
                       background model.
            detect_shadows: if true, the algorithm will detect shadows and mark them
        '''
        fgbg = cv2.createBackgroundSubtractorMOG2(history=history,
                                                  varThreshold=threshold,
                                                  detectShadows=detect_shadows)
        with etav.VideoProcessor(self.input_path, out_impath=self.output_path) as processor:
            for img in processor:
                fgmask = fgbg.apply(img)
                processor.write(fgmask)
                
    def background_subtractor_GMG(self, initialization_frame=120, threshold=0.8):
        '''Compute background subtractor by GMG algorithm.
        
        Args:
            initialization_frame: number of frames used to initialize the background models
            threshold: threshold to decide which is marked foreground, else background.
        '''
        fgbg = cv2.bgsegm.createBackgroundSubtractorGMG(initializationFrames=initialization_frame,
                                                        decisionThreshold=threshold)
        with etav.VideoProcessor(self.input_path, out_impath=self.output_path) as processor:
            for img in processor:
                fgmask = fgbg.apply(img)
                processor.write(fgmask)

