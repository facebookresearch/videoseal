# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
#!/bin/bash

# Set the number of jobs
num_jobs=10

# Create the logs directory if it doesn't exist
mkdir -p logs

# Loop through each job and submit it
for ((i=0; i<num_jobs; i++)); do
  # Submit the job using srun and redirect output to a log file
  srun -p learnlab --nodes=1 --ntasks=1 --cpus-per-task=40 --mem 240G --time 4320 python process_sab.py $i > logs/$i.log 2>&1 &
done
"""

import os
import csv
import shutil
from PIL import Image
import tqdm
import numpy as np
from multiprocessing import Pool

# Define the source and destination directories
src_dir = '/path/to/your/data'
dst_dir = '/path/to/your/splits'

# Create the destination directories if they don't exist
os.makedirs(dst_dir, exist_ok=True)
os.makedirs(os.path.join(dst_dir, 'train'), exist_ok=True)
os.makedirs(os.path.join(dst_dir, 'test'), exist_ok=True)
os.makedirs(os.path.join(dst_dir, 'val'), exist_ok=True)

# Get a list of all image files in the source directory
img_files = [f for f in os.listdir(src_dir) if f.endswith('.jpg') or f.endswith('.png')]
print(f'Found {len(img_files)} image files')

# Calculate chunks for 100 jobs
num_jobs=10
num_test = 1000
num_val = 1000
start_idx = num_test + num_val

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('job_id', type=int)
    args = parser.parse_args()

    print('starting job', args.job_id)
    
    # Calculate this job's chunk
    total_images = len(img_files[start_idx:])
    chunk_size = total_images // num_jobs
    job_start = args.job_id * chunk_size + start_idx
    job_end = min((args.job_id + 1) * chunk_size + start_idx, len(img_files))
    print(f'{job_end-job_start} images - {job_start} to {job_end}')

    def process_image(img_idx):
        print(img_idx)
        img_file = img_files[img_idx]
        src_path = os.path.join(src_dir, img_file)
        dst_path = os.path.join(dst_dir, 'train', img_file)
        if os.path.exists(dst_path):
            return
        img = Image.open(src_path)
        img = img.resize((256, 256))
        img.save(dst_path)

    with Pool() as pool:
        pool.map(process_image, range(job_start, job_end))
