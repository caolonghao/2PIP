import sys
import os
import itk
import numpy as np
from pathlib import Path
import json
from utils import create_shift_from_image_stack
import shutil
import cv2
from SimpleMontage import create_mosaic

def montage_configure(input_path, Montage_folder, num_rows, num_cols, 
                      hor_x_shift, hor_y_shift, ver_y_shift, ver_x_shift, z_order):
    
    # import ipdb; ipdb.set_trace()
    
    base_path = input_path.parent
    ret, image_stack = cv2.imreadmulti(str(input_path), flags=cv2.IMREAD_UNCHANGED)
    image_stack = np.array(image_stack, dtype=np.float32)
    
    if Path.exists(Montage_folder):
        shutil.rmtree(Montage_folder)
    
    Path.mkdir(Montage_folder)
    
    cordinates = []
    for i in range(num_rows):
        x_pivot = 0
        y_pivot = i * ver_y_shift
        for j in range(num_cols):
            index = i * num_cols + j
            image = image_stack[index]
            image = cv2.normalize(image, None, 0, 1024, cv2.NORM_MINMAX, cv2.CV_16U)
            cv2.imwrite(str(Montage_folder / "crop_{:d}.tif".format(index)), image)
    
    configfile_path = Montage_folder / "TileConfiguration.txt"
    
    shift_config = create_shift_from_image_stack(image_stack, num_rows, num_cols, hor_x_shift, hor_y_shift,
                                                 ver_y_shift, ver_x_shift, z_order)
    with open(str(configfile_path), 'w') as f:
        f.write('dim = 2\n')
        for index, x, y in shift_config:
            f.write('crop_{}.tif;;({}, {})\n'.format(index, x, y))

if __name__ == "__main__":
    with open('plugin.json', 'r') as f:
        args = json.load(f)
    
    input_path = Path(args["info"]["inpath"])
    out_file = Path(args["info"]["outpath"])
    output_folder = out_file.parent
    
    if not Path.exists(output_folder):
        Path.mkdir(output_folder)
    
    Montage_folder = output_folder / "MontageTest"
    
    num_rows = args["params"]["num_rows"]["value"]
    num_cols = args["params"]["num_cols"]["value"]
    hor_x_shift = args["params"]["hor_x_shift"]["value"]
    hor_y_shift = args["params"]["hor_y_shift"]["value"]
    ver_y_shift = args["params"]["ver_y_shift"]["value"]
    ver_x_shift = args["params"]["ver_x_shift"]["value"]
    z_order = args["params"]["z_order"]["value"]
    
    montage_configure(input_path, Montage_folder, num_rows, num_cols, 
                      hor_x_shift, hor_y_shift, ver_y_shift, ver_x_shift, z_order)
    
    create_mosaic(Montage_folder, output_folder, out_file)
