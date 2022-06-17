import subprocess
import os
from tkinter import W
import yaml
from loguru import logger
from pathlib import Path
import numpy as np 

import openvino.inference_engine as ie
from openvino.runtime import Core
import torch

from yolov5.utils import file_size


def onnx2openvino(model, onnx_path, save_path, data_type):
    # YOLOv5 OpenVINO export
    try:

        logger.info(f'OpenVINO: starting export with openvino {ie.__version__}...')
        f = str(save_path).replace('.pt', f'_openvino_model{os.sep}')

        cmd = f"mo --input_model {onnx_path} --output_dir {f} --data_type {'FP16' if data_type=='fp16' else 'FP32'}"
        subprocess.check_output(cmd.split())  # export
        with open(Path(f) / save_path.with_suffix('.yaml').name, 'w') as g:
            yaml.dump({'stride': int(max(model.stride)), 'names': model.names}, g)  # add metadata.yaml

        logger.info(f'OpenVINO: export success, saved as {f} ({file_size(f):.1f} MB)')
        return f
    except Exception as e:
        logger.info(f'\nOpenVINO: export failure: {e}')


def openvino_inference(model_path, im, device, **kwargs):
    ie = Core()
    if not Path(model_path).is_file():  # if not *.xml
        w = next(Path(model_path).glob('*.xml'))  # get *.xml file from *_openvino_model dir
    network = ie.read_model(model=w, weights=Path(w).with_suffix('.bin'))
    executable_network = ie.compile_model(model=network, device_name="CPU")
    output_layer = next(iter(executable_network.outputs))
    meta = Path(model_path).with_suffix('.yaml')

    im = im.cpu().numpy()  # FP32
    y = executable_network([im])[output_layer]

    if isinstance(y, np.ndarray):
        y = torch.tensor(y, device=device)
    return y