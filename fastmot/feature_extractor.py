from multiprocessing.pool import ThreadPool
import numpy as np
import numba as nb
import cv2
import time

from . import models
from .utils import TRTInference
from .utils.rect import multi_crop


class FeatureExtractor:
    def __init__(self, model='VehicleID_Baseline', batch_size=16):
        
        #timestamp1 = time.time() #timestamp for measuring execution time of the system
        
        """A feature extractor for ReID embeddings.

        Parameters
        ----------
        model : str, optional
            ReID model to use.
            Must be the name of a class that inherits `models.ReID`.
        batch_size : int, optional
            Batch size for inference.
        """
        self.model = models.ReID.get_model(model)
        assert batch_size >= 1
        self.batch_size = batch_size

        self.feature_dim = self.model.OUTPUT_LAYOUT
        self.backend = TRTInference(self.model, self.batch_size)
        self.inp_handle = self.backend.input.host.reshape(self.batch_size, *self.model.INPUT_SHAPE)
        self.pool = ThreadPool()

        self.embeddings = []
        self.last_num_features = 0
        
        #timestamp2 = time.time() #timestamp for measuring execution time of the system
        
        #print(f'Extractor init time: {(timestamp2-timestamp1):6.3f} seconds')

    def __del__(self):
        self.pool.close()
        self.pool.join()

    def __call__(self, frame, tlbrs):
        """Extract feature embeddings from bounding boxes synchronously."""
        self.extract_async(frame, tlbrs)
        return self.postprocess()

    @property
    def metric(self):
        return self.model.METRIC

    def extract_async(self, frame, tlbrs):
        """Extract feature embeddings from bounding boxes asynchronously."""
        imgs = multi_crop(frame, tlbrs)
        """
        # Displaying the image 
        cv2.imshow("window_name", imgs)
        cv2.waitKey(0)
        cv2.imshow
        """
        #print(tlbrs)
        #"""
        self.embeddings, cur_imgs = [], []
        # pipeline inference and preprocessing the next batch in parallel
        for offset in range(0, len(imgs), self.batch_size):
            cur_imgs = imgs[offset:offset + self.batch_size]
            self.pool.starmap(self._preprocess, enumerate(cur_imgs))
            #self.pool.starmap(self._preprocess(offset,cur_imgs), enumerate(cur_imgs))
            if offset > 0:
                embedding_out = self.backend.synchronize()[0]
                self.embeddings.append(embedding_out)
            timestamp1 = time.time() #timestamp for measuring execution time of the system
            self.backend.infer_async()
            timestamp2 = time.time() #timestamp for measuring execution time of the system
            #print("\nFeature Extractor's backend.infer_async: " + str(round((timestamp2 - timestamp1),4)))
        self.last_num_features = len(cur_imgs)

    def postprocess(self):
        """Synchronizes, applies postprocessing, and returns a NxM matrix of N
        extracted embeddings with dimension M.
        This API should be called after `extract_async`.
        """
        #timestamp1 = time.time() #timestamp for measuring execution time of the system
        
        if self.last_num_features == 0:
            return np.empty((0, self.feature_dim))
        
        #timestamp2 = time.time() #timestamp for measuring execution time of the system
        
        embedding_out = self.backend.synchronize()[0][:self.last_num_features * self.feature_dim]
        
        #timestamp3 = time.time() #timestamp for measuring execution time of the system
        
        self.embeddings.append(embedding_out)
        
        #timestamp4 = time.time() #timestamp for measuring execution time of the system
        
        embeddings = np.concatenate(self.embeddings).reshape(-1, self.feature_dim)
        
        #timestamp5 = time.time() #timestamp for measuring execution time of the system
        
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        #timestamp6 = time.time() #timestamp for measuring execution time of the system
        """
        print(f'\nreturn np.empty time: {(timestamp2-timestamp1):6.3f} seconds')
        print(f'\nbackend.synchronize time: {(timestamp3-timestamp2):6.3f} seconds')
        print(f'\nappend(embedding_out) time: {(timestamp4-timestamp3):6.3f} seconds')
        print(f'\nnp.concatenate time: {(timestamp5-timestamp4):6.3f} seconds')
        print(f'\nnp.linalg.norm time: {(timestamp6-timestamp5):6.3f} seconds\n')
        """
        return embeddings

    def null_embeddings(self, detections):
        """Returns a NxM matrix of N identical embeddings with dimension M.
        This API effectively disables feature extraction.
        """
        embeddings = np.ones((len(detections), self.feature_dim))
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings

    def _preprocess(self, idx, img):
        img = cv2.resize(img, self.model.INPUT_SHAPE[:0:-1])
        self._normalize(img, self.inp_handle[idx])

    @staticmethod
    @nb.njit(fastmath=True, nogil=True, cache=True)
    def _normalize(img, out):
        # BGR to RGB
        rgb = img[..., ::-1]
        # HWC -> CHW
        chw = rgb.transpose(2, 0, 1)
        # Normalize using ImageNet's mean and std
        out[0, ...] = (chw[0, ...] / 255. - 0.485) / 0.229
        out[1, ...] = (chw[1, ...] / 255. - 0.456) / 0.224
        out[2, ...] = (chw[2, ...] / 255. - 0.406) / 0.225
