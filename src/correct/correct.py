from utils import *
from naive_estimate import NaiveEstimate
import numpy as np
import cv2
import json
import datetime, time
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
    log_path = "{:s}/correct_{:s}.log".format(log_folder, current_date)
    if not os.path.isdir(log_folder):
        os.mkdir(log_folder)
    logging.basicConfig(level=logging.INFO, filename=log_path, format="%(levelname)s:%(asctime)s:%(message)s")
    
    estimatorName = args['params']['estimatorName']['value']
    window_size = args['params']['windowSize']['value']
    gaussian_sigma = args['params']['gaussianFilterSigma']['value']
    select_scale_factor = args['params']['deviationScaleFactor']['value']
    
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
        exit(0)
    
    print("Estimator: {:s}".format(estimatorName))
    
    if estimatorName == "NaiveEstimate":
        estimator = NaiveEstimate(window_size=window_size, gaussian_sigma=gaussian_sigma,
                                             select_scale_factor=select_scale_factor)
    elif estimatorName == "BaSicEstimate":
        try:
            from basic_estimate import BaSicEstimate
            estimator = BaSicEstimate()
        except:
            estimator = NaiveEstimate(window_size=window_size, gaussian_sigma=gaussian_sigma,
                                             select_scale_factor=select_scale_factor)
            print("Import basicpy error, fall back to Naive Estimate")
            logging.error("Import basicpy error, fall back to Naive Estimate")
    
    logging.info("Estimate method: {:s}".format(estimatorName))
    
    print("Start Estimating...")
    estimate_start_time = time.time()
    if estimatorName == "BaSicEstimate":
        flat_field, dark_field = estimator(img_stack)
    else:
        dark_field = cv2.imread("../temp/darkest_img.tif", cv2.IMREAD_UNCHANGED)
        flat_field, dark_field = estimator(img_stack)
    # save flat_field and dark_field as images
    cv2.imwrite("../temp/flat_field.tif", flat_field)
    cv2.imwrite("../temp/dark_field.tif", dark_field)
    estimate_end_time = time.time()
    print("estimate time: {:.3f}s".format(estimate_end_time - estimate_start_time))
    img_stack = reverse_with_flat_bg(img_stack, flat_field, dark_field)
    logging.info("Estimated flat_field range: {:.3f} - {:.3f}".format(flat_field.min(), flat_field.max()))
    logging.info("Estimated dark_field range: {:.3f} - {:.3f}".format(dark_field.min(), dark_field.max()))
    
    try: 
        if outType == 'tif':
            # save img_stack in 16-bit tif format
            img_stack = img_stack.astype(np.uint16)
            cv2.imwritemulti(output_path, img_stack)
        elif outType == 'npy':
            np.save(output_path, img_stack)
        logging.info("Save data successful")
    except:
        logging.error("Save data failed")
        exit(0)

    end_time = time.time()
    logging.info("Total time consumption: {:.3f}s".format(end_time - start_time))
    
    print("Output Image Saved.")
