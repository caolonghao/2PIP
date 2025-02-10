import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# For denoise evaluation

def calculate_entropy(image, histogram_path = None):
    # Convert the image to grayscale if it is not already
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Compute the histogram of the image
    hist, _ = np.histogram(image, bins=256, range=(0, 256))
    
    # if histogram_path is not None, save the histogram graph
    if histogram_path is not None:
        plt.figure()
        plt.bar(np.arange(256), hist, color='b', edgecolor='k')
        plt.xlim([0, 256])
        plt.title('Histogram')
        plt.savefig(histogram_path)
        plt.close()
    
    # Normalize the histogram to get the probability distribution
    hist = hist.astype(float) / np.sum(hist)
    
    # Calculate the entropy
    entropy = -np.sum([p * np.log2(p) for p in hist if p > 0])
    
    return entropy

# calculate the normalized inverse coefficient variation of the image
def calculate_icv(avg_intensity):
    # Calculate the mean and standard deviation of the average intensity
    mean = np.mean(avg_intensity)
    std = np.std(avg_intensity)
    
    print(f'Mean: {mean}, Std: {std}')
    
    # Calculate the normalized inverse coefficient variation
    icv = mean / std
    
    return icv
    
def eval_multi_icv_plot(images, row_patch_num=None, col_patch_num=None, plot_path=None):
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.figure(figsize=(12, 7))
    
    icv_values = []
    custom_colors = ['#A1A8FF', '#008080', '#3bd181']
    
    num_patches = (len(images[0][0]) - 1024) // 512 + 1
    x = np.arange(num_patches)
    # print(x.shape)
    width = 0.2
    
    for idx, (image, name) in enumerate(images):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        axis = 'row' if row_patch_num is not None else 'column'
        
        icv_curve = []
        if axis == 'row':
            for i in range(0, image.shape[0] - 1024 + 1, 512):
                patch = image[i:i+1024, :] if i + 1024 <= image.shape[0] else image[i:, :]
                print(patch.shape,i,i+1024)
                avg_intensity = np.mean(patch, axis=1)
                icv_curve.append(calculate_icv(avg_intensity))
        elif axis == 'column':
            for i in range(0, image.shape[1] - 1024 + 1, 512):
                patch = image[:, i:i+1024] if i + 1024 <= image.shape[1] else image[:, i:]
                avg_intensity = np.mean(patch, axis=0)
                icv_curve.append(calculate_icv(avg_intensity))
        else:
            raise ValueError('Invalid axis argument. Must be either "row" or "column".')
        
        icv_values.append((icv_curve, name))
        
        # Adjust the bar positions to avoid overlap
        plt.bar(x + idx * width, icv_curve, width, color=custom_colors[idx % len(custom_colors)], label=name)
    
    plt.title('ICV Curve across ' + axis + 's', fontsize=16)
    plt.xlabel('Patch Index', fontsize=14)
    plt.ylabel('ICV', fontsize=14)
    plt.xticks(x + width, [f'Patch {i+1}' for i in range(num_patches)], fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    if plot_path is not None:
        plt.savefig(plot_path, bbox_inches='tight')




def eval_multi_average_intensity(images, row_patch_num=None, col_patch_num=None, plot_path=None):
    plt.rcParams['font.family'] = 'Times New Roman'
    # Create a new plot
    plt.figure()
    
    icv_values = []
    
    # Colors to use for different lines
    # custom_colors = ['#FB9017', '#3bd181', '#9017FB']
    custom_colors=['#A1A8FF','#008080','#3bd181']
    
    for idx, (image, name) in enumerate(images):
        # Convert the image to grayscale if it is not already
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        axis = 'row' if row_patch_num is not None else 'column'
        
        # Calculate the average intensity in every row or column of the image
        if axis == 'row':
            avg_intensity = np.mean(image, axis=1)
            avg_distance = image.shape[0] // row_patch_num
        elif axis == 'column':
            avg_intensity = np.mean(image, axis=0)
            avg_distance = image.shape[1] // col_patch_num
        else:
            raise ValueError('Invalid axis argument. Must be either "row" or "column".')
        
        icv_values.append((calculate_icv(avg_intensity), name))
        
        # Plot the average intensity with a unique color and label
        plt.plot(avg_intensity, color=custom_colors[idx % len(custom_colors)], label=name)
    
        if axis == 'row':
            for i in range(row_patch_num // 2):
                plt.axvspan((2 * i + 1) * avg_distance, (2 * i + 2) * avg_distance, color='gray', alpha=0.05)
        else:
            for i in range(col_patch_num // 2):
                plt.axvspan((2 * i + 1) * avg_distance, (2 * i + 2) * avg_distance, color='gray', alpha=0.05)

    
    plt.title('Average Intensity across ' + axis + 's', fontsize=16)
    plt.xlabel(axis.capitalize(), fontsize=14)
    plt.xlim([0, len(avg_intensity)])
    plt.ylabel('Average Intensity', fontsize=14)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    if plot_path is not None:
        plt.savefig(plot_path)
    return icv_values

def get_vertical_pages(info_file_path):
    vertical_pages = 10  # 默认值
    try:
        with open(info_file_path, 'r') as f:
            for line in f:
                if line.startswith('Vertical_Pages'):
                    vertical_pages = int(line.split()[-1])
                    break
    except Exception as e:
        print(f"Error reading {info_file_path}: {str(e)}")
    return vertical_pages

def process_images_in_directory(base_dir, log_file):
    with open(log_file, 'w') as log:
        for root, dirs, files in os.walk(base_dir):
            for dir_name in dirs:
                if dir_name == 'CellVideo':
                    cellvideo_path = os.path.join(root, dir_name)
                    try:
                        info_file_path = os.path.join(root, 'Information.txt')
                        if not os.path.exists(info_file_path):
                            log.write(f"Missing Information.txt in directory: {root}\n")
                            continue
                        row_patch_num = get_vertical_pages(info_file_path)
                        print(row_patch_num)
                        naive_path = os.path.join(cellvideo_path, 'NaiveEstimateFinalOutput.tif')
                        basic_path = os.path.join(cellvideo_path, 'BaSicEstimateFinalOutput.tif')
                        none_path = os.path.join(cellvideo_path, 'NoneFinalOutput.tif')
                        
                        if not all(os.path.exists(p) for p in [naive_path, basic_path, none_path]):
                            log.write(f"Missing file(s) in directory: {cellvideo_path}\n")
                            continue
                        
                        images = [
                            (cv2.imread(naive_path, flags=cv2.IMREAD_UNCHANGED), 'NaiveEstimate'),
                            (cv2.imread(basic_path, flags=cv2.IMREAD_UNCHANGED), 'BaSicEstimate'),
                            (cv2.imread(none_path, flags=cv2.IMREAD_UNCHANGED), 'NoneEstimate')
                        ]
                        
                        output_plot_path = os.path.join(cellvideo_path, 'average_intensity_plot.png')
                        out_icv_plot_path= os.path.join(cellvideo_path, 'icv_plot.png')                                                                        
                        icv_values = eval_multi_average_intensity(images, row_patch_num=row_patch_num, plot_path=output_plot_path)
                        eval_multi_icv_plot(images, row_patch_num=row_patch_num, plot_path=out_icv_plot_path)

                        for icv, name in icv_values:
                            log.write(f"{cellvideo_path} - {name} ICV: {icv}\n")
                    
                    except Exception as e:
                        log.write(f"Error processing directory {cellvideo_path}: {str(e)}\n")

if __name__ == '__main__':
    image_path = "E:\Mycode\Light Field Correction\Data\jiezhichang\AF T2\CellVideo/NaiveEstimateFinalOutput.tif"
    basic_image_path = "E:\Mycode\Light Field Correction\Data\jiezhichang\AF T2\CellVideo/BaSicEstimateFinalOutput.tif"
    origin_image_path = "E:\Mycode\Light Field Correction\Data\jiezhichang\AF T2\CellVideo/NoneFinalOutput.tif"
    
    images = [
        (cv2.imread(image_path, flags=cv2.IMREAD_UNCHANGED), 'Ours'),
        (cv2.imread(basic_image_path, flags=cv2.IMREAD_UNCHANGED), 'BaSiC'),
        # (cv2.imread(origin_image_path, flags=cv2.IMREAD_UNCHANGED), 'Raw')
    ]
    
    # image = cv2.imread(image_path, flags=cv2.IMREAD_UNCHANGED)
    icv_value = eval_multi_average_intensity(images, row_patch_num=10, plot_path='average_intensity.png')
    
    # calculate icv of the images, higher is better (less fluctuation in the average intensity)
    for icv, name in icv_value:
        print(f'{name} ICV: {icv}')