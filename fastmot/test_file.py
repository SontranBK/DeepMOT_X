from types import SimpleNamespace
from enum import Enum
import logging
from collections import defaultdict
from pathlib import Path
import configparser
import abc
import numpy as np
import numba as nb
import cupy as cp
import cupyx.scipy.ndimage
import cv2
import time
#from common import *

import models
from utils import TRTInference
from utils.rect import as_tlbr, aspect_ratio, to_tlbr, get_size, area
from utils.rect import enclosing, multi_crop, iom, diou_nms
from utils.numba import find_split_indices


#extra lib for YOLOX
import tensorrt as trt
import ctypes
import math
import random
from PIL import Image
import pycuda.driver as cuda
import sys
sys.path.append('.')
sys.path.append('/home/minh/fastMOT_X/FastMOT/fastmot/utils')
import common 


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
INPUT_SHAPE = (3, 416, 416)
STRIDES = [8, 16, 32]
NMS_THRESH = 0.4
IOU_THRESH = 0.7
NUM_CLASSES = 80


DET_DTYPE = np.dtype(
    [('tlbr', float, 4),
     ('label', int),
     ('conf', float)],
    align=True
)


class YOLOXDetector:

########################################################################################################
#   Step 0: Load engine => create stream => create context from stream
########################################################################################################

    def __init__(self):
        #load engine 
        engine_path = '/home/minh/Car_tracking_Jetson/YOLOX/demo/TensorRT/cpp/model_trt.engine'
        runtime = trt.Runtime(TRT_LOGGER)
        engine = common.deserialize_engine_from_file(engine_path, runtime)

        #allocate buffers and create a stream.
        self.inputs, self.outputs, self.bindings, self.stream = common.allocate_buffers(engine)
        self.context = engine.create_execution_context()
        
        #generate the grids, strides to store output from GPU (do_inference function)
        self.grids, self.strides, self.total_cells = self.generate_grids_and_stride_numpy(STRIDES)
        
########################################################################################################
#   Step 1: Preprocess image
########################################################################################################

    def _preprocess(self,image, target_size):
        ih, iw    = target_size
        h,  w, _  = image.shape
        # same as height, width = img.shape[:2]
        #Smart resize
        scale = min(iw/w, ih/h)
        # same as ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        image_paded = np.full(shape=[ih, iw, 3], fill_value=114)
        nw, nh  = int(scale * w), int(scale * h)
        image_resized = cv2.resize(image, (nw, nh), cv2.INTER_LINEAR)
        # and img, _ = self.preproc(img, None, self.test_size)

        #Padding    
        image_paded[:nh,:nw,:] = image_resized
        #padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img


        # same as img = torch.from_numpy(img).unsqueeze(0)
        # and img = img.float()
        #Normalize
        #image_paded = image_paded / 255.

        image_paded = image_paded.transpose(2,0, 1)

        # same as: padded_img = padded_img.transpose(swap)
        
        return image_paded, scale
    
########################################################################################################
#   Step 2: Do inference

#   Function detect_async = step 0 + step 1 + step 2
########################################################################################################

    def detect_async(self,frame):
        #input
        image, scale = self._preprocess(frame, (INPUT_SHAPE[1], INPUT_SHAPE[2]))

        # Copy to the pagelocked input buffer
        np.copyto(dst=self.inputs[0].host, src=image.ravel()) #only 1 input
        #print("img after being ravelled: "+str(image.ravel()))

        #inference
        
        [output_of_detect_async] = common.do_inference(self.context, bindings=self.bindings, inputs=self.inputs, outputs=self.outputs, stream=self.stream)
        
        #print(f'outputs from model (GPU): {output}, type: {type(output)}, shape: {output.shape}')    

        return output_of_detect_async, scale
        
        
        
        #out = post_process(output_of_detect_async, grids, strides, total_cells, NMS_THRESH, IOU_THRESH, scale)
    
    
########################################################################################################
#   Step 3: Generate_grids_and_stride
########################################################################################################

    
    def generate_grids_and_stride_numpy(self,strides: []):

        heights = [INPUT_SHAPE[1] // stride for stride in strides]
        widths = [INPUT_SHAPE[2] // stride for stride in strides]

        total_cells = int(0)
        grids =[]
        strides_array = []

        for height, width, stride in zip(heights, widths, strides):
            total_cells += int(height*width)
            yv, xv = np.meshgrid(np.arange(height), np.arange(width)) #[(32,32), (32,32)]          
            grid = np.stack([xv, yv], -1) # (32,32,2)
            grid = np.reshape(grid, (-1,2)) # (32*32, 2)
            grids.append(grid) 
            strides_array.append(np.full((grid.shape[-2], 1), stride)) # (32*32, 1)
            

        grids = np.concatenate(grids, axis=-2) #(32^2+16^2+8^2, 2)       
        grids[:,[1,0]]=grids[:,[0,1]]
        strides_array = np.concatenate(strides_array, axis=-2) #(32^2+16^2+8^2, 1)

        assert total_cells == len(strides_array)

        #print("grids: "+str(grids)+" data type: "+str(type(grids)))

        return grids, strides_array, total_cells
    
    
        
        
    
########################################################################################################
#   Step 4: Post process
########################################################################################################

    def postprocess(self,predictions: np.ndarray, scale: float):
    #def postprocess(self,predictions, scale):
        #print(f'generate grids: {self.grids}, strides: {self.strides}, total cells: {self.total_cells}') 
        outputs = self.generate_yolox_proposals_numpy(predictions, self.grids, self.strides, self.total_cells)
        #print("prediction before nms: "+str(outputs[1999]))
        #print("shape of output before nms: "+str(outputs.shape))
        #print(f"Number of boxes before nms: {len(outputs)}")
        outputs = self.nms(outputs, NMS_THRESH, IOU_THRESH)
        #print(f"Number of boxes after nms: {len(outputs)}")    
        outputs = self.rescaled_box_corners(outputs, scale)


        detections = []
        for i in range(len(outputs)):
            tlbr = np.empty(4)
            tlbr[0] = int(outputs[i,0])
            tlbr[1] = int(outputs[i,2])
            tlbr[2] = int(outputs[i,1])
            tlbr[3] = int(outputs[i,3])                 
            label = int(outputs[i, 5])
            conf = outputs[i,4]
            detections.append((tlbr, label, conf))

        #print(f'detections before np.fromiter: {detections}, type: {type(detections)}, shape: {len(detections)}')

        detections = np.fromiter(detections, DET_DTYPE, len(detections)).view(np.recarray)

        #print(f'detections: {detections}, type: {type(detections)}, shape: {detections.shape}')            

        #np.set_printoptions(precision=3,suppress=True) #nicely format the output
        #print(f'output: {np.around(outputs,decimals=2)}, type: {type(outputs)}, shape: {outputs.shape}')  

        return detections
    
########################################################################################################
#   (Inside Step 4) Step 4.1: generate_yolox_proposals
########################################################################################################

    def generate_yolox_proposals_numpy(self,feat_blob: np.ndarray, grids: np.ndarray, strides: np.ndarray, total_cells: int)->np.ndarray:
        matrix_blob = np.reshape(feat_blob, (total_cells, 5+NUM_CLASSES))

        #print("shape of model's output: "+str(np.shape(matrix_blob)))
        #print("model's output: "+str(matrix_blob))

        box_objectness = matrix_blob[:, 4].reshape((total_cells), 1) # box objectness #use sigmoid here?
        class_conf = np.max(matrix_blob[:, 5:], axis=-1, keepdims=True)
        class_ids = np.argmax(matrix_blob[:, 5:], axis=-1).reshape((total_cells, 1))
        outputs = np.empty(shape=(len(matrix_blob), 6))

        #print("grids: "+str(grids))

        outputs[:, :2] = (matrix_blob[:, :2] + grids) * strides # (x_center, y_center)
        outputs[:, 2:4] = np.exp(matrix_blob[:, 2:4]) * strides # (w, h)

        outputs[:, 4:5] = class_conf * box_objectness
        #outputs[:, 4:5] = class_conf
        outputs[:, 5:] = class_ids # class_ids

        #print(f'Max of class conf: {class_conf.max()},Max of box objectness: {box_objectness.max()}')

        return outputs

########################################################################################################
#   (Inside Step 4) Step 4.2: iou and nms
########################################################################################################

    def iou(self,lbox, rbox):
        inter_box = np.array([max(lbox[0]-lbox[2]/2., rbox[0]-rbox[2]/2.), min(lbox[0]+lbox[2]/2., rbox[0]+rbox[2]/2.), max(lbox[1]-lbox[3]/2., rbox[1]-rbox[3]/2.), min(lbox[1]+lbox[3]/2., rbox[1]+rbox[3]/2.)])

        if (inter_box[2] > inter_box[3]) or (inter_box[0] > inter_box[1]):
            return 0.
        interBoxS =(inter_box[1]-inter_box[0])*(inter_box[3]-inter_box[2])

        return interBoxS/(lbox[2]*lbox[3] + rbox[2]*rbox[3] -interBoxS)

    def nms(self,predictions: np.ndarray, conf_threshold: float, iou_threshold: float)->np.ndarray:  
        #Filter out low-confidence box
        mask = predictions[:, 4]>conf_threshold
        #print(predictions[:, 4].max())
        predictions = predictions[mask]
        
        #print("prediction before nms: "+str(predictions))
        
        valids = []
        
        for class_id in np.unique(predictions[:, 5]):
            #print(class_id)    
            #dtypes = [('x', 'float'), ()]
            #Group boxes with the same class into the same group for compare IOU
            valid = predictions[predictions[:, 5]==class_id]
            #Sort boxes in each group w.r.t confidence point
            valid = valid[np.flip(np.argsort(valid[:, 4]))]
            #print(valid)
            accept_idx = []
            delete_idx = []
            for i, v in enumerate(valid):
               if i not in delete_idx: accept_idx.append(i) 
               for j, w in enumerate(valid[i+1:]):
                   #IOU = iou(v[:4], w[:4])
                   #print(IOU)  
                   if self.iou(v[:4], w[:4]) > iou_threshold: delete_idx.append(i+1+j)
                   
            #print(accept_idx)
            #print(delete_idx)
            for i in accept_idx:
                valids.append(valid[i])
        

        return np.vstack(valids)
        #print(len(valids))
    
    
    def new_nms(bboxes, iou_threshold, sigma=0.3, method='nms'):
        """
        :param bboxes: (xmin, ymin, xmax, ymax, score, class)
        Note: soft-nms, https://arxiv.org/pdf/1704.04503.pdf
              https://github.com/bharatsingh430/soft-nms
        """
        classes_in_img = list(set(bboxes[:, 5]))
        best_bboxes = []

        for cls in classes_in_img:
            cls_mask = (bboxes[:, 5] == cls)
            cls_bboxes = bboxes[cls_mask]

            while len(cls_bboxes) > 0:
                max_ind = np.argmax(cls_bboxes[:, 4])
                best_bbox = cls_bboxes[max_ind]
                best_bboxes.append(best_bbox)
                cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
                iou = bbox_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
                weight = np.ones((len(iou),), dtype=np.float32)

                assert method in ['nms', 'soft-nms']

                if method == 'nms':
                    iou_mask = iou > iou_threshold
                    weight[iou_mask] = 0.0

                if method == 'soft-nms':
                    weight = np.exp(-(1.0 * iou ** 2 / sigma))

                cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
                score_mask = cls_bboxes[:, 4] > 0.
                cls_bboxes = cls_bboxes[score_mask]

        return best_bboxes



########################################################################################################
#   (Inside Step 4) Step 4.3: rescaled_box_corners
########################################################################################################

    def rescaled_box_corners(self,outputs: np.ndarray, scale: float)->np.ndarray:
        x_center, y_center, w, h = outputs[:, 0], outputs[:, 1], outputs[:, 2], outputs[:, 3]
        
        x0 = (x_center - w/2) / scale
        x1 = (x_center + w/2) / scale
        y0 = (y_center - h/2) / scale
        y1 = (y_center + h/2)  / scale

        out = np.copy(outputs)
        out[:, 0] = x0; out[:, 1] = x1; out[:, 2] = y0; out[:, 3] = y1

        return out
    
    
    
    

    def visualize(self,out,image):

        for i in range(len(out)):          
      
            # Start coordinate, represents the top left corner of rectangle
            start_point = (int(out[i, 0]), int(out[i, 2]))
            
            print (start_point)
            # Ending coordinate
            # represents the bottom right corner of rectangle
            end_point = (int(out[i, 1]), int(out[i, 3]))
            print (end_point)
      
            # Blue color in BGR
            color = (255, 0, 0)
      
            # Line thickness of 2 px
            thickness = 2
      
            # Using cv2.rectangle() method
            # Draw a rectangle with blue line borders of thickness of 2 px
            image = cv2.rectangle(image, start_point, end_point, color, thickness)
            
            
        # Window name in which image is displayed
        window_name = 'Image'
      
        # Displaying the image 
        cv2.imshow(window_name, image)
        cv2.waitKey(0)
    
    
def main():
    
    #for image
    #im_path = '/home/minh/Car_tracking_Jetson/YOLOX/assets/highway1.jpg'
    #frame = cv2.imread(im_path)
    
    #for video
    video_path = "/home/minh/Car_tracking_Jetson/YOLOX/Test_data/video_data/cars.mp4"	
    cap = cv2.VideoCapture(video_path)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #print(video_length)


    detector = YOLOXDetector()
    
    timing_sum = 0
    
    for i in range(video_length):
        ret, frame = cap.read()
        timestamp1 = time.time() #timestamp for measuring execution time of the system
        
        output_of_detect_async, scale = detector.detect_async(frame)
        detections = detector.postprocess(output_of_detect_async, scale)
        
        timestamp2 = time.time() #timestamp for measuring execution time of the system

        #detector.visualize(detections,frame)
        
        #print(f'detections: {detections}, type: {type(detections)}, shape: {detections.shape}')
        
        #np.set_printoptions(precision=3,suppress=True) #nicely format the output
        timing_sum = timing_sum + (timestamp2 - timestamp1)
        print(f'Frame no.: {i}')
        
    avg_timing_info = timing_sum / video_length
    print(f'Avg detecting time: {avg_timing_info:6.5f}, Avg FPS: {1/avg_timing_info:6.5f}')


if __name__=="__main__":
    main()
