import subprocess
import os
import yaml
from loguru import logger
from pathlib import Path
import numpy as np 

import torch
import tensorflow as tf 

from yolov5.utils import file_size

def openvino2tensorflow(model_name, openvino_path, save_dir):
    try:
        logger.info(f'TF: starting starting export with TF {tf.__version__}...')
        if not Path(openvino_path).is_file():
            openvino_path = next(Path(openvino_path).glob('*.xml'))  # get *.xml file from *_openvino_model dir

        saved_model_path = str(save_dir / 'saved_model.pb')
        cmd = f'openvino2tensorflow --model_path {openvino_path} --model_output_path {save_dir}\
                --output_saved_model  --weight_replacement_config ./data/convert/replace_{model_name}.json'
        
        subprocess.check_output(cmd.split())        
        logger.info(f'TF: export success, saved_model saved as {saved_model_path} ({file_size(saved_model_path):.1f} MB)')

        return saved_model_path
    
    except Exception as e:
        logger.info(f'\TF: export failure: {e}')



def tensorflow_load(model_path, **kwargs):
    model = tf.saved_model.load(model_path)
    model = model.signatures['serving_default']

    return model

def tensorflow_inference(model, im, device, **kwargs):
    im = im.permute(0, 2, 3, 1).cpu().numpy()  # torch BCHW to numpy BHWC shape(1,320,192,3)
    im = tf.constant(im)
    y = model(im)['Identity'].numpy()

    if isinstance(y, np.ndarray):
        y = torch.tensor(y, device=device)
    return y 