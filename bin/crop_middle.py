#!/usr/bin/env python
"""
Usage::


   cd ~/simoncblyth.github.io/env/presentation/GEOM/J25_7_2_opticks_Debug/yupd_bottompipe_adjust/Portraits

   open 20260116_140918.png

   $IPYTHON  ~/env/bin/crop_middle.py 20260116_140918.png

   open 20260116_140918_crop_middle.png


"""

import os, sys
import numpy as np
from PIL import Image


path = sys.argv[1]
fold = os.path.dirname(path)
name = os.path.basename(path)
assert name.endswith(".png") 
crop_name = name.replace(".png", "_crop_middle.png")
crop_path = os.path.join(fold, crop_name)

img = Image.open(path)

img_array = np.array(img)
height, width = img_array.shape[:2] # Note: NumPy uses (H, W)

target_width = 1280
left = (width - target_width) // 2
top = 0
right = left + target_width
bottom = height

cropped_img = img.crop((left, top, right, bottom))

cropped_img.save(crop_path)
#cropped_img.show()

print(f"Original {path}     size: {img.size}") # (2560, 1440)
print(f"Cropped {crop_path} size: {cropped_img.size}") # (1280, 1440)



