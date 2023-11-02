#!/usr/bin/env python
"""
image_target.py
================

Create an image of [2000,1000] pixels with 
text coordinates within each block of [100,100] pixels

https://code-maven.com/create-images-with-python-pil-pillow

"""

import os, numpy as np
from PIL import Image, ImageDraw, ImageFont

grid = np.array( [20,10] )
scale = 100 
path = os.environ["IMAGE_TARGET_PATH"]
     
img = Image.new('RGB', tuple(grid*scale), color = 'red')
d = ImageDraw.Draw(img)

fnt = ImageFont.truetype('/Library/Fonts/Arial.ttf', 25)

for i in range(grid[0]):
    for j in range(grid[1]):
        pos = i*scale,j*scale
        label = "(%d,%d)" % (i, j)
        fill = (255,255,0)
        d.text(pos, label, fill=fill, font=fnt )
    pass
pass

img.save(path)


