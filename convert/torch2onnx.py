from loguru import logger
from pathlib import Path
import numpy as np

import onnx
import onnxruntime
import torch

from yolov5.utils import file_size
from yolov5.experimental import attempt_load


def torch2onnx(model, im, save_path, train, dynamic):
    try:
        logger.info(f'ONNX: starting export with onnx {onnx.__version__}...')
        torch.onnx.export(model, im, save_path, verbose=False, opset_version=12,
                            training=torch.onnx.TrainingMode.TRAINING if train else torch.onnx.TrainingMode.EVAL,
                            do_constant_folding=not train,
                            input_names=['images'],
                            output_names=['output'],
                            dynamic_axes={'images': {0: 'batch'},  # shape(1,3,640,640)
                                        'output': {
                                            0: 'batch',
                                            }  # shape(1,25200,85)
                                        }if dynamic else None)
        # Checks
        model_onnx = onnx.load(save_path)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model
    
        # Metadata
        d = {'stride': int(max(model.stride)), 'names': model.names}
        for k, v in d.items():
            meta = model_onnx.metadata_props.add()
            meta.key, meta.value = k, str(v)
        onnx.save(model_onnx, save_path)

        logger.info(f'ONNX: export success, saved as {save_path} ({file_size(save_path):.1f} MB)')

    except Exception as e:
        logger.info(f'ONNX: export failure: {e}')


def onnx_load(model_path, device, **kwargs):
    cuda = torch.cuda.is_available()
    if device == 'cpu':
        cuda = False
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
    session = onnxruntime.InferenceSession(str(model_path), providers=providers)
    meta = session.get_modelmeta().custom_metadata_map  # metadata
    if 'stride' in meta:
        stride, names = int(meta['stride']), eval(meta['names'])
    return session

def onnx_inference(model, im, device, **kwargs):
    session = model
    im = im.cpu().numpy()  # torch to numpy
    y = session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: im})[0]

    if isinstance(y, np.ndarray):
        y = torch.tensor(y, device=device)
    return y

def torch_load(model_path, device, **kwargs):
    model = attempt_load(model_path if isinstance(model_path, list) else model_path, device=device)
    return model

def torch_inference(model, im, device, **kwargs):
    stride = max(int(model.stride.max()), 32)  # model stride
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    model.half() if kwargs['data_type'] == 'fp16' else model.float()
    y = model(im)[0]
    return y 