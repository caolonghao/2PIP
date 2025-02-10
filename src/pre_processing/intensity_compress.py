import numpy as np
import argparse
import cv2

def intensity_compress(src, min_num=2.5, max_num=97.5):
    out = []
    for img in src:
        max_threshold = np.percentile(img, max_num)
        min_threshold = np.percentile(img, min_num)
        img = np.clip(img, min_threshold, max_threshold)
        out.append(img)

    return np.array(out)

def parse_args():
    parser = argparse.ArgumentParser(description="compress image intensity(darkest and greatest)")
    parser.add_argument("--min_num", type=float, default=2.5)
    parser.add_argument("--max_num", type=float, default=97.5)
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--input_type", type=str, default="npy")
    parser.add_argument("--output_type", type=str, default="npy")
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    min_num = args.min_num
    max_num = args.max_num
    input_path = args.input_path
    output_path = args.output_path
    intput_type = args.input_type
    output_type = args.output_type
    
    if intput_type == "npy":
        img = np.load(input_path)
    elif intput_type == "tif":
        ret, img = cv2.imreadmulti(input_path, flags=cv2.IMREAD_UNCHANGED)
        img = np.array(img)
    
    img = intensity_compress(img, min_num=min_num, max_num=max_num)
    
    if output_type == "npy":
        np.save(output_path, img)
    elif output_type == "tif":
        cv2.imwritemulti(output_path, img)
    