import os
import subprocess
import logging

# Base directory
base_dir = r"E:\Mycode\Light Field Correction\Data\640353(crosstalk)"

# Method to be used
method = "BaSicEstimate"  # NaiveEstimate, BasicEstimate, None

# Configure logging
logging.basicConfig(filename='process_log.txt', level=logging.ERROR,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Get the second-level directories
second_level_dirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

# Iterate through each second-level directory
for dir in second_level_dirs:
    sub_dirs = [os.path.join(dir, sub_dir) for sub_dir in os.listdir(dir) if os.path.isdir(os.path.join(dir, sub_dir))]
    for sub_dir in sub_dirs:
        # Construct the command
        command = ["python", "run.py", "--input_dir", sub_dir, "--method", method]
        
        try:
            # Execute the command
            print(f"Executing: {' '.join(command)}")
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            # Log the error if the command fails
            logging.error(f"Command {' '.join(command)} failed with error: {e}")
        except Exception as e:
            # Log any other unexpected errors
            logging.error(f"An unexpected error occurred while executing command {' '.join(command)}: {e}")
