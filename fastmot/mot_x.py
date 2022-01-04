from types import SimpleNamespace
from enum import Enum
import logging
import numpy as np
import cv2
import time

from .feature_extractor import FeatureExtractor
from .tracker import MultiTracker
from .utils import Profiler
from .utils.visualization import Visualizer
from .utils.numba import find_split_indices
from .detector_X import YOLOXDetector


LOGGER = logging.getLogger(__name__)


class DetectorType(Enum):
    SSD = 0
    YOLO = 1
    PUBLIC = 2


class MOT_X:
    def __init__(self, size,
                 detector_type='YOLO',
                 detector_frame_skip=5,
                 class_ids=(1,2,3,4,6,8),
                 ssd_detector_cfg=None,
                 yolo_detector_cfg=None,
                 public_detector_cfg=None,
                 feature_extractor_cfgs=None,
                 tracker_cfg=None,
                 visualizer_cfg=None,
                 draw=False):
        """Top level module that integrates detection, feature extraction,
        and tracking together.

        Parameters
        ----------
        size : tuple
            Width and height of each frame.
        detector_type : {'SSD', 'YOLO', 'public'}, optional
            Type of detector to use.
        detector_frame_skip : int, optional
            Number of frames to skip for the detector.
        class_ids : sequence, optional
            Class IDs to track. Note class ID starts at zero.
        ssd_detector_cfg : SimpleNamespace, optional
            SSD detector configuration.
        yolo_detector_cfg : SimpleNamespace, optional
            YOLO detector configuration.
        public_detector_cfg : SimpleNamespace, optional
            Public detector configuration.
        feature_extractor_cfgs : List[SimpleNamespace], optional
            Feature extractor configurations for all classes.
            Each configuration corresponds to the class at the same index in sorted `class_ids`.
        tracker_cfg : SimpleNamespace, optional
            Tracker configuration.
        visualizer_cfg : SimpleNamespace, optional
            Visualization configuration.
        draw : bool, optional
            Draw visualizations.
        """
        self.size = size
        self.detector_type = DetectorType[detector_type.upper()]
        assert detector_frame_skip >= 1
        self.detector_frame_skip = detector_frame_skip
        self.class_ids = tuple(np.unique(class_ids))
        self.draw = draw
	
        """
        class_ids: [0, 2, 3, 4, 6, 8]
        0: u'__background__',
        2: u'bicycle',
        3: u'car',
        4: u'motorcycle',
        6: u'bus',
        8: u'truck'
        """
        if ssd_detector_cfg is None:
            ssd_detector_cfg = SimpleNamespace()
        if yolo_detector_cfg is None:
            yolo_detector_cfg = SimpleNamespace()
        if public_detector_cfg is None:
            public_detector_cfg = SimpleNamespace()
        if feature_extractor_cfgs is None:
            feature_extractor_cfgs = (SimpleNamespace(),)
        if tracker_cfg is None:
            tracker_cfg = SimpleNamespace()
        if visualizer_cfg is None:
            visualizer_cfg = SimpleNamespace()
        if len(feature_extractor_cfgs) != len(class_ids):
            raise ValueError('Number of feature extractors must match length of class IDs')

        LOGGER.info('Loading detector model...')
        self.detector = YOLOXDetector()
        
        

        LOGGER.info('Loading feature extractor models...')
        self.extractors = [FeatureExtractor(**vars(cfg)) for cfg in feature_extractor_cfgs]
        self.tracker = MultiTracker(self.size, self.extractors[0].metric, **vars(tracker_cfg))
        self.visualizer = Visualizer(**vars(visualizer_cfg))
        self.frame_count = 0

    def visible_tracks(self):
        """Retrieve visible tracks from the tracker

        Returns
        -------
        Iterator[Track]
            Confirmed and active tracks from the tracker.
        """
        return (track for track in self.tracker.tracks.values()
                if track.confirmed and track.active
               )

    def reset(self, cap_dt):
        """Resets multiple object tracker. Must be called before `step`.

        Parameters
        ----------
        cap_dt : float
            Time interval in seconds between each frame.
        """
        self.frame_count = 0
        self.tracker.reset(cap_dt)

    def step(self, frame):
        """Runs multiple object tracker on the next frame.

        Parameters
        ----------
        frame : ndarray
            The next frame.
        """
        detections = []
        if self.frame_count == 0:
            # Two below line is for YOLO-X only:
            output_of_detect_async, scale = self.detector.detect_async(frame)
            detections = self.detector.postprocess(output_of_detect_async, scale)
            
            #print(f'detections: {detections}, type: {type(detections)}, shape: {detections.shape}')    
            self.tracker.init(frame, detections)
        elif self.frame_count % self.detector_frame_skip == 0:
        
            #timestamp1 = time.time() #timestamp for measuring execution time of the system
            
            with Profiler('preproc'):
                # Below line is for YOLO-X only:
                output_of_detect_async, scale = self.detector.detect_async(frame)
                
            #timestamp2 = time.time() #timestamp for measuring execution time of the system
            
            with Profiler('detect'):
                with Profiler('track'):
                    self.tracker.compute_flow(frame)
                # Below line is for YOLO-X only:
                detections = self.detector.postprocess(output_of_detect_async, scale)

            #print(f'detections: {detections}, type: {type(detections)}, shape: {detections.shape}')    
            #timestamp3 = time.time() #timestamp for measuring execution time of the system
            
            with Profiler('extract'):
                
                timestamp6 = time.time() #timestamp for measuring execution time of the system
                
                cls_bboxes = np.split(detections.tlbr, find_split_indices(detections.label))
                for extractor, bboxes in zip(self.extractors, cls_bboxes):
                    extractor.extract_async(frame, bboxes)
                    
                timestamp7 = time.time() #timestamp for measuring execution time of the system
                
                with Profiler('track', aggregate=True):
                    self.tracker.apply_kalman()
                    
                timestamp8 = time.time() #timestamp for measuring execution time of the system
                    
                embeddings = []
                for extractor in self.extractors:
                    embeddings.append(extractor.postprocess())
                    
                timestamp9 = time.time() #timestamp for measuring execution time of the system
                
                embeddings = np.concatenate(embeddings) if len(embeddings) > 1 else embeddings[0]
                
                
                
            timestamp4 = time.time() #timestamp for measuring execution time of the system
                            
            with Profiler('assoc'):
                self.tracker.update(self.frame_count, detections, embeddings)
                
            #timestamp5 = time.time() #timestamp for measuring execution time of the system
            #total_time = timestamp5 - timestamp1
            
            """
            #print out execution time of the system
            print("\nPreprocessing time: " + str(round((timestamp2 - timestamp1),4)) + 
                  " = " + str(round(100*(timestamp2 - timestamp1)/total_time,4)) + " % total time" +
                  "\nDectecting time: " + str(round((timestamp3 - timestamp2),4)) +
                  " = " + str(round(100*(timestamp3 - timestamp2)/total_time,4)) + " % total time" +
                  "\nFeature extracting time: " + str(round((timestamp4 - timestamp3),4)) +
                  " = " + str(round(100*(timestamp4 - timestamp3)/total_time,4)) + " % total time" +
                  "\nTracking time: " + str(round((timestamp5 - timestamp4),4)) +
                  " = " + str(round(100*(timestamp5 - timestamp4)/total_time,4)) + " % total time" +
                  "\nTotal time: " + str(round(total_time,4)) +"\n")
            """
            
            
            #print out execution time of extractor
            """
            extract_async_time = timestamp7 - timestamp6
            apply_kalman_time = timestamp8 - timestamp7
            extractor_postprocess_time = timestamp9 - timestamp8
            extractor_total_time = timestamp4 - timestamp6
            print(f'\nExtract async time: {extract_async_time:6.3f} seconds = {(100*extract_async_time/extractor_total_time):6.3f} % total extractor time')
            print(f'Apply kalman time: {apply_kalman_time:6.3f} seconds = {(100*apply_kalman_time/extractor_total_time):6.3f} % total extractor time')
            print(f'Extractor postprocess time: {extractor_postprocess_time:6.3f} seconds = {(100*extractor_postprocess_time/extractor_total_time):6.3f} % total extractor time')
            print(f'\nExtractor total time: {extractor_total_time:6.3f} seconds = 100 % total extractor time\n')
            """
            
            
            
        else:
            with Profiler('track'):
                self.tracker.track(frame)
	
        if self.draw:
            self._draw(frame, detections)
        self.frame_count += 1

    @staticmethod
    def print_timing_info():
        # Calculate timing info
        Preprocess_time = Profiler.get_avg_millis('preproc')
        Detect_time = Profiler.get_avg_millis('detect')
        Feature_extract_time = Profiler.get_avg_millis('extract')
        Track_time = Profiler.get_avg_millis('track')
        Association_time = Profiler.get_avg_millis('assoc')
        Total_time = Preprocess_time + Detect_time + Feature_extract_time + Track_time + Association_time
        # Logger into console
        LOGGER.info('=============================Timing Stats=============================')
        LOGGER.info(f"{'preprocess time:':<37}{Preprocess_time:>6.3f} ms = {(100*Preprocess_time/Total_time):6.3f} % total time")
        LOGGER.info(f"{'detect/flow time:':<37}{Detect_time:>6.3f} ms = {(100*Detect_time/Total_time):6.3f} % total time")
        LOGGER.info(f"{'feature extract/kalman filter time:':<37}"
                     f"{Feature_extract_time:>6.3f} ms = {(100*Feature_extract_time/Total_time):6.3f} % total time")
        LOGGER.info(f"{'track time:':<37}{Track_time:>6.3f} ms = {(100*Track_time/Total_time):6.3f} % total time")
        LOGGER.info(f"{'association time:':<37}{Association_time:>6.3f} ms = {(100*Association_time/Total_time):6.3f} % total time")
        LOGGER.info('--------------------------------Sum Up--------------------------------')
        LOGGER.info(f"{'total time:':<37}{Total_time:>6.3f} ms = 100 % total time")
        

    def _draw(self, frame, detections):
        visible_tracks = list(self.visible_tracks())
        #print(visible_tracks)
        self.visualizer.render(frame, visible_tracks, detections, self.tracker.klt_bboxes.values(),
                               self.tracker.flow.prev_bg_keypoints, self.tracker.flow.bg_keypoints)
        cv2.putText(frame, f'visible: {len(visible_tracks)}', (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2, cv2.LINE_AA)