from os import listdir
from os.path import isfile, join
import os
import shutil
import random

# source of the framefiles
sourcepath1 = '/media/itri/Transcend/firetruck/firetruck_cls_dataset/train/n03345487'
sourcepath2 = '/media/itri/Transcend/firetruck/firetruck_cls_dataset/val/n03345487'

# fetch frame files
def get_frame_files(sourcepath):
    files = []
    for f in listdir(sourcepath):
        if "frame" in f and isfile(join(sourcepath, f)):
            files.append(join(sourcepath, f))
    return files

files1 = get_frame_files(sourcepath1)
files2 = get_frame_files(sourcepath2)

# merge files
all_files = files1 + files2

# Specify the temporary folder
output_fold = "/media/itri/Transcend/firetruck/firetruck_cls_dataset/temporary"

# Ensure the temporary folder exists, create if not
os.makedirs(output_fold, exist_ok=True)

# Move all files to the temporary folder
for f in all_files:
    shutil.move(f, output_fold)

# 80:20 split from temporary folder
all_temp_files = listdir(output_fold)  # Get all files from the temporary folder
random.shuffle(all_temp_files)

# Calculate 80% and 20% split
train_count = round(len(all_temp_files) * 0.8)
train_files = all_temp_files[:train_count]
val_files = all_temp_files[train_count:]

# Ensure source folders exist
os.makedirs(sourcepath1, exist_ok=True)
os.makedirs(sourcepath2, exist_ok=True)

# Move 80% of files back to sourcepath1 (train folder)
for f in train_files:
    shutil.move(join(output_fold, f), sourcepath1)

# Move 20% of files back to sourcepath2 (val folder)
for f in val_files:
    shutil.move(join(output_fold, f), sourcepath2)

# Print finish
print(f"Finish！Total {len(all_temp_files)} files，already put {len(train_files)} files to {sourcepath1}，{len(val_files)} files to {sourcepath2}.")
