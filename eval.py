import os
import sys
import time
import numpy as np
import skimage.io
import keras.backend
# Download and install the Python COCO tools from https://github.com/waleedka/coco
# That's a fork from the original https://github.com/pdollar/coco with a bug
# fix for Python 3.
# I submitted a pull request https://github.com/cocodataset/cocoapi/pull/50
# If the PR is merged then use the original repo.
# Note: Edit PythonAPI/Makefile and replace "python" with "python3".
#  
# A quick one liner to install the library 
# !pip install git+https://github.com/waleedka/coco.git#subdirectory=PythonAPI

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

import coco #a slightly modified version

from mrcnn.evaluate import build_coco_results, evaluate_coco
from mrcnn.dataset import MappingChallengeDataset
from mrcnn import visualize


import zipfile
import urllib.request
import shutil
import glob
import tqdm
import random

ROOT_DIR = os.getcwd()

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils


PRETRAINED_MODEL_PATH = os.path.join(ROOT_DIR,"final_model.h5")
LOGS_DIRECTORY = os.path.join(ROOT_DIR, "logs")
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
IMAGE_DIR = os.path.join(ROOT_DIR, "data", "test_images")


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 5
    NUM_CLASSES = 41  # 1 Background + 1 Building
    IMAGE_MAX_DIM=320
    IMAGE_MIN_DIM=320
    NAME = "crowdai-mapping-challenge"
config = InferenceConfig()
config.display()

K = keras.backend.backend()
if K=='tensorflow':
    keras.backend.set_image_dim_ordering('tf')

model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

model_path = PRETRAINED_MODEL_PATH

# or if you want to use the latest trained model, you can use : 
# model_path = model.find_last()[1]

model.load_weights(model_path, by_name=True)

class_names = ['BG', 'bread-wholemeal', ' potatoes-steamed', ' broccoli', ' butter', ' hard-cheese', ' water', ' banana', ' wine-white', ' bread-white', ' apple', ' pizza-margherita-baked', ' salad-leaf-salad-green', ' zucchini', ' water-mineral', ' coffee-with-caffeine', ' avocado', ' tomato', ' dark-chocolate', ' white-coffee-with-caffeine', ' egg', ' mixed-salad-chopped-without-sauce', ' sweet-pepper', ' mixed-vegetables', ' mayonnaise', ' rice', ' chips-french-fries', ' carrot', ' tomato-sauce', ' cucumber', ' wine-red', ' cheese', ' strawberries', ' espresso-with-caffeine', ' tea', ' chicken', ' jam', ' leaf-spinach', ' pasta-spaghetti', ' french-beans', ' bread-whole-wheat'] # In our case, we have 1 class for the background, and 1 class for building
file_names = next(os.walk(IMAGE_DIR))[2]

TRAIN_ANNOTATIONS_PATH = "data/train/annotation.json"
coco = COCO(TRAIN_ANNOTATIONS_PATH)
category_ids = coco.loadCats(coco.getCatIds())
id_category = [_["id"] for _ in category_ids]
id_category = [0] + id_category





def evaluate(IMAGE_DIR, PREDICTION_FILE):
    files = glob.glob(os.path.join(IMAGE_DIR, "*.jpg"))
    ALL_FILES=[]
    _buffer = []
    for _idx, _file in enumerate(files):
        if len(_buffer) == config.IMAGES_PER_GPU * config.GPU_COUNT:
            ALL_FILES.append(_buffer)
            _buffer = []
        else:
            _buffer.append(_file)

    if len(_buffer) > 0:
        ALL_FILES.append(_buffer)


    _final_object = []
    for files in tqdm.tqdm(ALL_FILES):
        images = [skimage.io.imread(x) for x in files]
        if(len(images)!= config.IMAGES_PER_GPU):
            images = images + [images[-1]]*(config.BATCH_SIZE - len(images))
        predoctions = model.detect(images, verbose=0)
        for _idx, r in enumerate(predoctions):
            if(_idx < len(files)):
                _file = files[_idx]
                image_id = int(_file.split("/")[-1].replace(".jpg",""))
                for _idx, class_id in enumerate(r["class_ids"]):
                    if class_id > 0:
                        mask = r["masks"].astype(np.uint8)[:, :, _idx]
                        bbox = np.around(r["rois"][_idx], 1)
                        bbox = [float(x) for x in bbox]
                        _result = {}
                        _result["image_id"] = image_id
                        _result["category_id"] = id_category[class_id]
                        _result["score"] = float(r["scores"][_idx])
                        _mask = maskUtils.encode(np.asfortranarray(mask))
                        _mask["counts"] = _mask["counts"].decode("UTF-8")
                        _result["segmentation"] = _mask
                        _result["bbox"] = [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]]
                        _final_object.append(_result)
    fp = open(PREDICTION_FILE, "w")
    import json
    print("Writing JSON...")
    fp.write(json.dumps(_final_object))
    fp.close()