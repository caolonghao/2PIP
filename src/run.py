import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import json
import subprocess
import shutil


def read_information(information_file):
    """
    Read the information file
    """
    with open(information_file, 'r') as file:
        data = file.read()

    # 使用正则表达式查找所需的值
    frames_per_slice = re.search(r'Frames_per_slice\s+(\d+)', data)
    horizontal_pages = re.search(r'Horizontal_Pages\s+(\d+)', data)
    vertical_pages = re.search(r'Vertical_Pages\s+(\d+)', data)

    # 提取匹配到的值
    frames_per_slice_value = int(frames_per_slice.group(1)) if frames_per_slice else None
    horizontal_pages_value = int(horizontal_pages.group(1)) if horizontal_pages else None
    vertical_pages_value = int(vertical_pages.group(1)) if vertical_pages else None
    
    if frames_per_slice_value is None:
        print("Error: Frames per slice not found in the information file. Try new version of the information file.")
        frames_per_slice = re.search(r'Frame/Page\s+(\d+)', data)
        frames_per_slice_value = int(frames_per_slice.group(1)) if frames_per_slice else None
    
    if horizontal_pages_value is None:
        print("Error: Horizontal pages not found in the information file. Try new version of the information file.")
        horizontal_pages = re.search(r'Row\s+(\d+)', data)
        horizontal_pages_value = int(horizontal_pages.group(1)) if horizontal_pages else None
        
    if vertical_pages_value is None:
        print("Error: Vertical pages not found in the information file. Try new version of the information file.")
        vertical_pages = re.search(r'Column\s+(\d+)', data)
        vertical_pages_value = int(vertical_pages.group(1)) if vertical_pages else None
    
    return frames_per_slice_value, horizontal_pages_value, vertical_pages_value

# 修改 plugin.json 文件中的值
def get_nested_value(data, keys):
    """递归地获取嵌套字典中的值"""
    for key in keys:
        try:
            data = data[key]
        except KeyError:
            return None
    return data

def set_nested_value(data, keys, value):
    """递归地设置嵌套字典中的值"""
    for key in keys[:-1]:
        data = data.setdefault(key, {})
    data[keys[-1]] = value

def update_json_entries(directory, filename, updates):
    # 构建完整的文件路径
    file_path = os.path.join(directory, filename)
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"Error: The file {file_path} does not exist.")
        return
    
    # 读取JSON文件
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 更新指定的键
    for keys, new_value in updates.items():
        current_value = get_nested_value(data, keys)
        if current_value is not None:
            set_nested_value(data, keys, new_value)
        else:
            print(f"Warning: Path {' -> '.join(keys)} not found in the JSON file.")
    
    # 写回修改后的JSON数据
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    print(f"Updated multiple entries in {file_path}")

# 运行 pre_processing.py 脚本
def run_script(script_path, working_directory):
    result = subprocess.run(['python', script_path], cwd=working_directory, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Error:", result.stderr)



if __name__=='__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run the analysis')
    # Add arguments
    parser.add_argument('--input_dir', type=str, default='E:\Mycode\Light Field Correction\Data\jiezhichang\AF T1',
                        help='Input Directory')
    parser.add_argument('--method', type=str, default='NaiveEstimate', help='Method to use', 
                        choices=['NaiveEstimate', 'BaSicEstimate', 'None'])
    
    # Parse arguments
    args = parser.parse_args()
    
    input_dir = args.input_dir
    method = args.method
    
    information_dir = os.path.join(input_dir, 'Information.txt')
    if not os.path.exists(information_dir):
        print("Old version of the information file not found. Try new version of the information file.")
        information_dir = os.path.join(input_dir, 'Information-CHA.txt')
    
    if not os.path.exists(information_dir):
        raise FileNotFoundError(f"Information file not found in {input_dir}")
    
    frames_per_slice, horizontal_pages, vertical_pages = read_information(information_dir)
    
    data_folder = os.path.join(input_dir, 'CellVideo')
    
    raw_data_path = os.path.join(data_folder, 'CellVideo 0.tif')
    compressed_data_path = os.path.join(data_folder, 'Compressed.tif')
    corrected_data_path = os.path.join(data_folder, method + 'Corrected.tif')
    final_output_path = os.path.join(data_folder, method + 'FinalOutput.tif')
    
    # change the json file in pre_processing folder and go into it to run the pre_processing
    # 目标子目录和文件路径
    subdir = 'pre_processing'
    json_file = 'plugin.json'
    script_file = 'grid_noise_filter.py'
    
    # 修改文件
    updates = {
        ("info", "inpath"): raw_data_path,
        ("info", "outpath"): compressed_data_path,
        ("params", "averageNumber", "value"): frames_per_slice,
        ("params", "horizontalPages", "value"): horizontal_pages,
        ("params", "verticalPages", "value"): vertical_pages
    }
    
    update_json_entries(subdir, json_file, updates)
    # import ipdb
    # ipdb.set_trace()
    run_script(script_file, subdir)
    
    
    
    # print(f'Frames per slice: {frames_per_slice}')
    # print(f'Horizontal pages: {horizontal_pages}')
    # print(f'Vertical pages: {vertical_pages}')
    
    if method == 'None':
        # copy the compressed data to the corrected data
        shutil.copy(compressed_data_path, corrected_data_path)
    else:
        # modify and run the correct.py
        subdir = 'correct'
        json_file = 'plugin.json'
        script_file = 'correct.py'
        
        updates = {
            ("info", "inpath"): compressed_data_path,
            ("info", "outpath"): corrected_data_path,
            ("info", "method"): method,
            ("params", "estimatorName", "value"): method,
        }
        
        update_json_entries(subdir, json_file, updates)
        
        run_script(script_file, subdir)
    
    subdir = 'post_processing'
    json_file = 'plugin.json'
    script_file = 'post_process.py'
    
    updates = {
        ("info", "inpath"): corrected_data_path,
        ("info", "outpath"): final_output_path,
        ("params", "horizontalPages", "value"): horizontal_pages,
        ("params", "verticalPages", "value"): vertical_pages
    }
    
    update_json_entries(subdir, json_file, updates)
    
    run_script(script_file, subdir)
    