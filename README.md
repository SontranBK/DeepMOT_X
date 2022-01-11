# DeepMOT_X
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FGeekAlexis%2FDeepMOT_X&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) [![DOI](https://zenodo.org/badge/237143671.svg)](https://zenodo.org/badge/latestdoi/237143671)

<img src="assets/cars_x.gif" width="400"/> <img src="assets/Highway_x.gif" width="400"/>

## News
  - (11 Jan 2022) **v1.0.0-alpha release**. First version contains YOLOX and DeepSORT on Jetson Nano
  - (11 Jan 2022) **Vehicles counting feature added**. Count vehicles when they pass through a pre-drawn line

## Description
DeepMOT_X is a custom multiple object tracker that implements:
  - YOLOX detector
  - Deep SORT + OSNet ReID
  - KLT tracker
  - Camera motion compensation
  - Many video in out feature
  - Vehicles counting  

Two-stage trackers like Deep SORT run detection and feature extraction sequentially, which often becomes a bottleneck. DeepMOT_X significantly speeds up the entire system to run in **real-time** even on Jetson. Motion compensation improves tracking for scenes with moving camera, where Deep SORT and FairMOT fail.

To achieve faster processing, DeepMOT_X only runs the detector and feature extractor every N frames, while KLT fills in the gaps efficiently. DeepMOT_X also re-identifies objects that moved out of frame to keep the same IDs.


## Performance
### Results on MOT20 train set
| Detector Skip | MOTA | IDF1 | HOTA | MOTP | MT | ML |
|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| N = 1 | 66.8% | 56.4% | 45.0% | 79.3% | 912 | 274 |
| N = 5 | 65.1% | 57.1% | 44.3% | 77.9% | 860 | 317 |

### FPS on MOT17 sequences
| Sequence | Density | FPS |
|:-------|:-------:|:-------:|
| MOT17-13 | 5 - 30  | 42 |
| MOT17-04 | 30 - 50  | 26 |
| MOT17-03 | 50 - 80  | 18 |

DeepMOT_X has MOTA scores close to **state-of-the-art** trackers from the MOT Challenge. Increasing N shows small impact on MOTA. Tracking speed can reach up to **42 FPS** depending on the number of objects. Lighter models are recommended for a more constrained device like Jetson Nano. FPS is expected to be in the range of **50 - 150** on desktop CPU/GPU.

## Requirements
- CUDA >= 10
- cuDNN >= 7
- TensorRT >= 7
- OpenCV >= 3.3
- Numpy >= 1.17
- Scipy >= 1.5
- Numba == 0.48
- CuPy == 9.2
- TensorFlow < 2.0 (for SSD support)

### Install for x86 Ubuntu
Make sure to have [nvidia-docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) installed. The image requires NVIDIA Driver version >= 450 for Ubuntu 18.04 and >= 465.19.01 for Ubuntu 20.04. Build and run the docker image:
  ```bash
  # Add --build-arg TRT_IMAGE_VERSION=21.05 for Ubuntu 20.04
  # Add --build-arg CUPY_NVCC_GENERATE_CODE=... to speed up build for your GPU, e.g. "arch=compute_75,code=sm_75"
  docker build -t DeepMOT_X:latest .
  
  # Run xhost local:root first if you cannot visualize inside the container
  docker run --gpus all --rm -it -v $(pwd):/usr/src/app/DeepMOT_X -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=unix$DISPLAY -e TZ=$(cat /etc/timezone) DeepMOT_X:latest
  ```
### Install for Jetson Nano/TX2/Xavier NX/Xavier
Make sure to have [JetPack >= 4.4](https://developer.nvidia.com/embedded/jetpack) installed and run the script:
  ```bash
  ./scripts/install_jetson.sh
  ```
### Download models
Pretrained OSNet, SSD, and my YOLOv4 ONNX model are included.
  ```bash
  ./scripts/download_models.sh
  ```
### Build YOLOv4 TensorRT plugin
  ```bash
  cd DeepMOT_X/plugins
  make
  ```
### Download VOC dataset for INT8 calibration
Only required for SSD (not supported on Ubuntu 20.04)
  ```bash
  ./scripts/download_data.sh
  ```

## Usage
```bash
  python3 app_x.py --input-uri ... --mot
```
- Image sequence: `--input-uri %06d.jpg`
- Video file: `--input-uri file.mp4`
- USB webcam: `--input-uri /dev/video0`
- MIPI CSI camera: `--input-uri csi://0`
- RTSP stream: `--input-uri rtsp://<user>:<password>@<ip>:<port>/<path>`
- HTTP stream: `--input-uri http://<user>:<password>@<ip>:<port>/<path>`

Use `--show` to visualize, `--output-uri` to save output, and `--txt` for MOT compliant results.

Show help message for all options:
```bash
  python3 app_x.py -h
```
Note that the first run will be slow due to Numba compilation. To use the FFMPEG backend on x86, set `WITH_GSTREAMER = False` [here](https://github.com/GeekAlexis/DeepMOT_X/blob/3a4cad87743c226cf603a70b3f15961b9baf6873/DeepMOT_X/videoio.py#L11)
<details>
<summary> More options can be configured in cfg/mot.json </summary>

  - Set `resolution` and `frame_rate` that corresponds to the source data or camera configuration (optional). They are required for image sequence, camera sources, and saving txt results. List all configurations for a USB/CSI camera:
    ```bash
    v4l2-ctl -d /dev/video0 --list-formats-ext
    ```
  - To swap network, modify `model` under a detector. For example, you can choose from `SSDInceptionV2`, `SSDMobileNetV1`, or `SSDMobileNetV2` for SSD.
  - If more accuracy is desired and FPS is not an issue, lower `detector_frame_skip`. Similarly, raise `detector_frame_skip` to speed up tracking at the cost of accuracy. You may also want to change `max_age` such that `max_age` × `detector_frame_skip` ≈ 30
  - Modify `visualizer_cfg` to toggle drawing options.
  - All parameters are documented in the API.

</details>

 ## Track custom classes
DeepMOT_X can be easily extended to a custom class (e.g. vehicle). You need to train both YOLO and a ReID network on your object class. Check [Darknet](https://github.com/AlexeyAB/darknet) for training YOLO and [fast-reid](https://github.com/JDAI-CV/fast-reid) for training ReID. After training, convert weights to ONNX format. The TensorRT plugin adapted from [tensorrt_demos](https://github.com/jkjung-avt/tensorrt_demos/) is only compatible with Darknet.

DeepMOT_X also supports multi-class tracking. It is recommended to train a ReID network for each class to extract features separately.
### Convert YOLO to ONNX
1. Install ONNX version 1.4.1 (not the latest version)
    ```bash
    pip3 install onnx==1.4.1
    ```
2. Convert using your custom cfg and weights
    ```bash
    ./scripts/yolo2onnx.py --config yolov4.cfg --weights yolov4.weights
    ```
### Add custom YOLOv3/v4
1. Subclass `DeepMOT_X.models.YOLO` like here: https://github.com/GeekAlexis/DeepMOT_X/blob/32c217a7d289f15a3bb0c1820982df947c82a650/DeepMOT_X/models/yolo.py#L100-L109
    ```
    ENGINE_PATH : Path
        Path to TensorRT engine.
        If not found, TensorRT engine will be converted from the ONNX model
        at runtime and cached for later use.
    MODEL_PATH : Path
        Path to ONNX model.
    NUM_CLASSES : int
        Total number of trained classes.
    LETTERBOX : bool
        Keep aspect ratio when resizing.
    NEW_COORDS : bool
        new_coords Darknet parameter for each yolo layer.
    INPUT_SHAPE : tuple
        Input size in the format `(channel, height, width)`.
    LAYER_FACTORS : List[int]
        Scale factors with respect to the input size for each yolo layer.
    SCALES : List[float]
        scale_x_y Darknet parameter for each yolo layer.
    ANCHORS : List[List[int]]
        Anchors grouped by each yolo layer.
    ```
    Note anchors may not follow the same order in the Darknet cfg file. You need to mask out the anchors for each yolo layer using the indices in `mask` in Darknet cfg.
    Unlike YOLOv4, the anchors are usually in reverse for YOLOv3 and YOLOv3/v4-tiny
2. Set class labels to your object classes with `DeepMOT_X.models.set_label_map`
3. Modify cfg/mot.json: set `model` in `yolo_detector_cfg` to the added Python class name and set `class_ids` of interest. You may want to play with `conf_thresh` based on model performance
### Add custom ReID
1. Subclass `DeepMOT_X.models.ReID` like here: https://github.com/GeekAlexis/DeepMOT_X/blob/32c217a7d289f15a3bb0c1820982df947c82a650/DeepMOT_X/models/reid.py#L50-L55
    ```
    ENGINE_PATH : Path
        Path to TensorRT engine.
        If not found, TensorRT engine will be converted from the ONNX model
        at runtime and cached for later use.
    MODEL_PATH : Path
        Path to ONNX model.
    INPUT_SHAPE : tuple
        Input size in the format `(channel, height, width)`.
    OUTPUT_LAYOUT : int
        Feature dimension output by the model.
    METRIC : {'euclidean', 'cosine'}
        Distance metric used to match features.
    ```
2. Modify cfg/mot.json: set `model` in `feature_extractor_cfgs` to the added Python class name. For more than one class, add more feature extractor configurations to the list `feature_extractor_cfgs`. You may want to play with `max_assoc_cost` and `max_reid_cost` based on model performance

 ## Citation
 If you find this repo useful in your project or research, please star and consider citing it:
 ```bibtex
@software{sontranbk_2022,
  author       = {Tran Son, Nguyen Minh},
  title        = {{DeepMOT_X: High-Performance Multiple Object Tracking Based on Deep SORT, YOLOX and KLT}},
}
```
