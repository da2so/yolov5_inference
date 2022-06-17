# YOLOv5 Converting && Inference tools

The codes are mainly based on [yolov5](https://github.com/ultralytics/yolov5). 
This provides:
- Converting yolov5(Torch version) model into ONNX, OpenVINO, TF, TFLite models
- Inference images with Torch, ONNX, OpenVINO, TF, TFLite models



# How to Use

## Set up

```bash
git clone https://github.com/da2so/yolov5_inference.git  # clone
cd yolov5_inference
pip install -r requirements.txt 
```

## Converting with inferencing

Converting yolov5s torch model into ONNX:

```bash
python3 yolov5_convert.py --include onnx  --imgsz 640 --save_dir test --model_name yolov5s
```
Converting yolov5s torch model into ONNX and OpenVINO:

```bash
python3 yolov5_convert.py --include onnx openvino  --imgsz 640 --save_dir test --model_name yolov5s
```

The converted models are saved in 'save_dir' directory. Inferencing is automatically executed and results are saved in sub directory(e.g. save_dir/onnx_result) of 'save_dir'.