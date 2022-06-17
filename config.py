import wget
from loguru import logger
from pathlib import Path
from dataclasses import dataclass


@dataclass
class YOLOv5:
    BASE_URL  = "https://github.com/ultralytics/yolov5/releases/download/v6.1/"
    yolov5n = BASE_URL + 'yolov5n.pt'
    yolov5s = BASE_URL + 'yolov5s.pt'
    yolov5m = BASE_URL + 'yolov5m.pt'
    yolov5l = BASE_URL + 'yolov5l.pt'


def download_model(model_name: str, 
                save_path: str) -> None:
    
    model = dict(yolov5n=YOLOv5.yolov5n,
                yolov5s=YOLOv5.yolov5s,
                yolov5m=YOLOv5.yolov5m,
                yolov5l=YOLOv5.yolov5l)
    try:
        if not save_path.is_file():
            wget.download(model[model_name], str(save_path))
            logger.info(f'{model_name} model is downloaded safely in {save_path}')
        else:
            logger.info(f'model is already downloaded in {save_path}')
    except:
        logger.error(f'model can not downloaded')