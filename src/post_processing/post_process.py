import cv2
import numpy as np
import json
from utils import *
import time, datetime
import os
import logging
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
    
    median_filter_size = args['params']['medianFilterSize']['value']
    num_cols, num_rows = args['params']['horizontalPages']['value'], args['params']['verticalPages']['value']
    z_order = args['params']['zOrder']['value']
    
    
    try:
        if inType == 'tif':
            ret, img_stack = cv2.imreadmulti(input_path, flags=cv2.IMREAD_UNCHANGED)
            img_stack = np.array(img_stack, dtype=np.float32)
        elif inType == 'npy':
            img_stack = np.load(input_path)
        print("Load data finished...")
        logging.info("Load data successful")
    except:
        logging.error("Load data failed")
        exit(1)
    
       
    # try:
    # img = merge_images(images=img_stack, num_rows=num_rows, num_cols=num_cols)
    img = stitch_image_stack(image_stack=img_stack, row=num_rows, col=num_cols, z_order=z_order)
    
    logging.info("Merge images successful")
    # except:    
    #     logging.error("Merge images failed. May due to the mismatch of horizontal and vertical page numbers.")
    #     exit(1)
        
    # median blur and CLAHE
    print("Start Processing...")
    try:
        img = cv2.medianBlur(img, ksize=median_filter_size)
        # add normalization
        # img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 4096
        # normalized image to range 0-4096 using opencv
        img = cv2.normalize(img, None, 0, 4096, cv2.NORM_MINMAX)
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=[8, 8])
        # img = clahe.apply(img)
        # print("Apply CLAHE successful")
        logging.info("Apply Median filter Successful")
    except:
        logging.error("Applt CLAHE fail")
        exit(0)
    
    try: 
        if outType == 'tif':
            img = np.array(img, dtype=np.uint16)
            cv2.imwrite(output_path, img)
        elif outType == 'npy':
            np.save(output_path, img)
        logging.info("Save data successful")
    except:
        logging.error("Save data failed")
        exit(1)
    
    end_time = time.time()
    logging.info("Total time consumption: {:.3f}s".format(end_time - start_time))
    
    print("Save Data Completed.")
