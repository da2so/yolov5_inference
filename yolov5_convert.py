import argparse
import cv2
from pathlib import Path
from hypothesis import infer
from loguru import logger
from typing import Any

import torch 

from config import download_model
from yolov5.experimental import attempt_load, Detect
from yolov5.dataloader import LoadImages
from yolov5.utils import select_device, check_img_size, file_size, time_sync, \
                        non_max_suppression, scale_coords
from yolov5.plots import Annotator, colors
from convert import *


def main(opt: argparse.ArgumentParser) -> None:
    model_name, imgsz, bs, save_dir, include, device, half, inplace, train, dynamic, int8, img_dir= \
        opt.model_name, opt.imgsz, opt.batch_size, opt.save_dir, opt.include, \
        opt.device, opt.half, opt.inplace, opt.train, opt.dynamic, opt.int8, opt.img_dir


    save_dir = Path(save_dir)
    save_path = save_dir / (model_name + '.pt')
    save_dir.mkdir(parents=True, exist_ok=True) 
    download_model(model_name=model_name, save_path=save_path)

    include = [x.lower() for x in include]  # to lowercase
    include.append('pytorch') # add pytorch for default

    # Load PyTorch model
    device = select_device(device)
    if half:
        assert device.type != 'cpu', '--half only compatible with GPU export, i.e. use --device 0'
        assert not dynamic, '--half not compatible with --dynamic, i.e. use either --half or --dynamic but not both'
    
    model = attempt_load(save_path, device=device, inplace=True, fuse=True)  # load FP32 model
    nc, names, stride = model.nc, model.names, model.stride

    # Input
    gs = int(max(model.stride))  # grid size (max stride)
    imgsz = [check_img_size(x, gs) for x in imgsz]  # verify img_size are gs-multiples
    im = torch.zeros(bs, 3, *imgsz).to(device)  # image size(1,3,320,192) BCHW iDetection

     # Update model
    if half:
        im, model = im.half(), model.half()  # to FP16
        data_type = 'fp16'
    elif int8:
        data_type = 'int8'
    else:
        data_type = 'fp32'
    
    model.train() if train else model.eval()  # training mode = no Detect() layer grid construction
    for k, m in model.named_modules():
        if isinstance(m, Detect):
            m.inplace = inplace
            m.onnx_dynamic = dynamic
            m.export = True

    for _ in range(2):
        y = model(im)  # dry runs
    shape = tuple(y[0].shape)  # model output shape
    logger.info(f"\nPyTorch starting from {model_name} with output shape {shape} ({file_size(save_path):.1f} MB)")

    dataset = LoadImages(img_dir, img_size=imgsz, stride=32, auto=False)
    bs = 1  # batch_size

    for fi in include:
        if fi == 'pytorch':
            inference_func = torch_inference
            model_path = save_path
            save_inf_dir = save_dir / 'torch_result' 
        if fi == 'onnx':
            model_path = Path(save_path).with_suffix('.onnx')
            torch2onnx(model=model, im=im, save_path=model_path, train=train, dynamic=dynamic)
            inference_func = onnx_inference
            save_inf_dir = save_dir / 'onnx_result' 



        if fi in ['pytorch', 'onnx']:
            save_inf_dir.mkdir(parents=True, exist_ok=True) 
            logger.info(f'Loading {model_path} for {fi} Runtime inference...')
            inference(dataset=dataset, 
                        inference_func=inference_func, 
                        model_path=model_path, 
                        device=device, 
                        data_type=data_type, 
                        names=names,
                        save_dir=save_inf_dir)


def inference(dataset, 
            inference_func, 
            model_path, 
            device, 
            data_type,
            names,
            save_dir, 
            conf_thres: float = 0.25,
            iou_thres: float = 0.45,
            agnostic_nms: bool = False,
            max_det: int = 1000,
            classes: Any = None,
            **kwargs):

    dt, seen = [0.0, 0.0, 0.0], 0
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if data_type == 'fp16' else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1
        # Inference
        pred = inference_func(model_path=model_path, im=im, device=device, data_type=data_type)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1

            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            s += '%gx%g ' % im.shape[2:]  # print string

            annotator = Annotator(im0, line_width=3, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                
                 # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = f'{names[c]} {conf:.2f}'
                    annotator.box_label(xyxy, label, color=colors(c, True))

            # Stream results
            im0 = annotator.result()

            # Save results (image with detections)
            cv2.imwrite(save_path, im0)
          
        # Print time (inference-only)
        logger.info(f'{s}Done. ({t3 - t2:.3f}s)')
    
def parse_opt() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='yolov5s', help='yolov5s, yolov5n, yolov5m, yolov5l')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640, 640], help='image (h, w)')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--img_dir', type=str, default='./data', help='test image directory')
    parser.add_argument('--save_dir', type=str, default='./test', help='save directory')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--half', action='store_true', help='FP16 half-precision export')
    parser.add_argument('--inplace', action='store_true', help='set YOLOv5 Detect() inplace=True')
    parser.add_argument('--train', action='store_true', help='model.train() mode')
    parser.add_argument('--dynamic', action='store_true', help='ONNX/TF: dynamic axes')
    parser.add_argument('--int8', action='store_true', help='CoreML/TF INT8 quantization')
    parser.add_argument('--include',
                    nargs='+',
                    default=['openvino','onnx'],
                    help='onnx, openvino, saved_model, pb, tflite')

    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand

    logger.info(vars(opt))
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
    