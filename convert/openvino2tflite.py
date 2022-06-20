import subprocess
import os
import yaml
from loguru import logger
from pathlib import Path
import numpy as np 

import torch
import tensorflow as tf 

from yolov5.utils import file_size

def openvino2tflite(model_name, openvino_path, save_dir):
    try:
        logger.info(f'TFLite: starting starting export with TF {tf.__version__}...')
        if not Path(openvino_path).is_file():
            openvino_path = next(Path(openvino_path).glob('*.xml'))  # get *.xml file from *_openvino_model dir

        saved_model_path = str(save_dir / 'model_float32.tflite')
        cmd = f'openvino2tensorflow --model_path {openvino_path} --model_output_path {save_dir}\
                --output_no_quant_float32_tflite  --weight_replacement_config ./data/convert/replace_{model_name}.json'
        
        subprocess.check_output(cmd.split())        
        logger.info(f'TFLite: export success, saved_model saved as {saved_model_path} ({file_size(saved_model_path):.1f} MB)')

        return saved_model_path
    
    except Exception as e:
        logger.info(f'\TFLite: export failure: {e}')



def tflite_load(model_path, **kwargs):
    Interpreter, load_delegate = tf.lite.Interpreter, tf.lite.experimental.load_delegate
    interpreter = Interpreter(model_path=model_path)  # load TFLite model
    return interpreter


def tflite_inference(model, im, device, **kwargs):
    interpreter = model
    interpreter.allocate_tensors()  # allocate
    input_details = interpreter.get_input_details()  # inputs
    output_details = interpreter.get_output_details()  # outputs
  
    im = im.permute(0, 2, 3, 1).cpu().numpy()  # torch BCHW to numpy BHWC shape(1,320,192,3)
    input, output = input_details[0], output_details[0]
    int8 = input['dtype'] == np.uint8  # is TFLite quantized uint8 model
    if int8:
        scale, zero_point = input['quantization']
        im = (im / scale + zero_point).astype(np.uint8)  # de-scale
    interpreter.set_tensor(input['index'], im)
    interpreter.invoke()
    y = interpreter.get_tensor(output['index'])
    if int8:
        scale, zero_point = output['quantization']
        y = (y.astype(np.float32) - zero_point) * scale  # re-scale

    if isinstance(y, np.ndarray):
        y = torch.tensor(y, device=device)

    return y