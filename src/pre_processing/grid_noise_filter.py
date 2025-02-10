import numpy as np
from numba import jit
import json
import cv2
import datetime, time
import logging
import os
from utils import *
from intensity_compress import intensity_compress
import sys
sys.path.append('..')
from logging_utils.utils import remove_outdated_logs
from scipy.ndimage import median_filter

# 注意adaptive mask的输入是频域值
@jit(nopython=True)
def adaptive_mask(src):
    mask = np.ones(src.shape, dtype=np.uint8)
    window_width = 10
    window_length = 170

    intensity = []
    intensity_sum = 0
    for i in range(0, src.shape[0] + 1 - window_width, window_width):
        box = src[i : i + window_width, 0:window_length]
        intensity.append([i, np.abs(box).sum()])
        intensity_sum += np.abs(box).sum()

    intensity.sort(key=lambda x: x[1], reverse=True)
    threshold = 1.25 * intensity_sum / len(intensity)
    # for index, value in intensity:
    #     print("index:", index, "value:", value)

    for index, value in intensity[:4]:
        # print("value:", value, "threshold:", threshold, "index:", index)
        if value > threshold:
            mask[index : index + window_width, :window_length] = 0
            mask[index : index + window_width, -window_length:] = 0

    return mask


def grid_noise_filter(src):
    out = []
    for img in src:
        imgs_fft = np.fft.fft2(img)
        imgs_fft = np.fft.fftshift(imgs_fft)
        mask = adaptive_mask(imgs_fft)
        imgs_fft = imgs_fft * mask
        imgs_new = np.fft.ifft2(imgs_fft)
        imgs_new = np.abs(imgs_new)
        out.append(imgs_new)

    return np.array(out)

if __name__=='__main__':
    with open('./plugin.json', 'rb') as f:
        args = json.load(f)
    
    start_time = time.time()
    
    print("Start Processing...")
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
    
    min_num = args['params']['minNum']['value']
    max_num = args['params']['maxNum']['value']
    col_num, row_num = args['params']['horizontalPages']['value'], args['params']['verticalPages']['value']
    average_num = args['params']['averageNumber']['value']

    try:
        if inType == 'tif':
            ret, img = cv2.imreadmulti(input_path, flags=cv2.IMREAD_UNCHANGED)
            img = np.array(img)
        elif inType == 'npy':
            img = np.load(input_path)
        print("Load data finished...")
        logging.info("Load data successful")
    except:
        logging.error("Load data failed")
        exit(0)
    
    try:
        # save the pixel-wise darkest image
        # darkest_img = np.min(img, axis=0).astype(np.uint16)
        # select the 0.05 percentile of the image as the darkest image
        darkest_img = np.percentile(img, 0.02, axis=0).astype(np.uint16)
        # apply median filter to the darkest image
        darkest_img = median_filter(darkest_img, size=9)
        # if /temp folder does not exist, create it
        
        if not os.path.exists("../temp"):
            os.makedirs("../temp")
        cv2.imwrite("../temp/darkest_img.tif", darkest_img)
        
        
        img, applied_padding = image_padding(img, average_num=average_num, col=col_num, row=row_num)
        img = average_img(img, average_num)
        # save the average image to /temp folder for debugging
        cv2.imwritemulti("../temp/averaged_img.tif", img)
        logging.info("Image padding successful")
    except:
        logging.info("Image padding failed, please check the image numbers")
        exit(1)
    
    if applied_padding is True:
        logging.info("Not enough images, applied padding")
    else:
        logging.info("Image numbers match, no padding")
    
    img = intensity_compress(img, min_num=min_num, max_num=max_num)
    img = grid_noise_filter(img)
    logging.info("Apply adaptive noise filter successful")
    
    print("Process filter finished, start saving...")
    try: 
        if outType == 'tif':
            # save image in 16-bit
            img = img.astype(np.uint16)
            cv2.imwritemulti(output_path, img)
        elif outType == 'npy':
            np.save(output_path, img)
        logging.info("Save data successful")
    except:
        print("Save data failed")
        logging.error("Save data failed")
        exit(0)
    
    end_time = time.time()
    logging.info("Total time consumption: {:.3f}s".format(end_time - start_time))
    
    print("Save data complete.")
    