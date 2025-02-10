import pandas as pd
import numpy as np
from skimage.transform import PiecewiseAffineTransform, warp
import cv2
import json
import time
import os
import logging
import datetime
import sys
sys.path.append('..')
from logging_utils.utils import remove_outdated_logs

if __name__ == "__main__":
    with open("./plugin.json", "rb") as f:
        args = json.load(f)
    
    start_time = time.time()
    
    input_path = args['info']['inpath']
    output_path = args['info']['outpath']
    inType = args['info']['inType']
    outType = args['info']['outType']
    
    log_folder = args['info']['logpath']
    remove_outdated_logs(log_folder)
    
    current_date = datetime.datetime.now().strftime('%Y%m%d')
    log_path = "{:s}/preprocessing_{:s}.log".format(log_folder, current_date)
    if not os.path.isdir(log_folder):
        os.mkdir(log_folder)
    logging.basicConfig(level=logging.INFO, filename=log_path, format="%(levelname)s:%(asctime)s:%(message)s")
    logging.info("Start Processing")
    
    idea_grids_path = args["params"]["idealGridsPath"]["value"]
    moved_grids_path = args["params"]["wrapedGridsPath"]["value"]
    
    try:
        ret, image_stack = cv2.imreadmulti(input_path, flags=cv2.IMREAD_UNCHANGED)
        image_stack = np.array(image_stack)
        fixed_points = pd.read_csv(idea_grids_path)
        moved_points = pd.read_csv(moved_grids_path)
        logging.info(".tif .csv file read complete")
    except:
        logging.error("read files failed, please check")
        exit(1)
    
    fixed_points = fixed_points[['X', 'Y']]
    moved_points = moved_points[['X', 'Y']]

    fixed_points = np.array(fixed_points)
    moved_points = np.array(moved_points)

    try:
        output = []
        tform = PiecewiseAffineTransform()
        tform.estimate(fixed_points, moved_points)
        for image in image_stack:
            output.append(warp(image, tform, output_shape=image.shape))
        output = np.array(output)
        logging.info("warping successful")
    except:
        logging.error("warping fail, please check if fixed points and moved points match")
        exit(1)
    
    cv2.imwritemulti(output_path, output)
    