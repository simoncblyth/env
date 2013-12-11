#!/usr/bin/env python
"""
* http://ju.outofmemory.cn/entry/18060
"""
from PIL import Image, ImageDraw

if __name__ == '__main__':
    pass
    size, fill = [100,100], 'blue'
    img = Image.new('RGBA', (size[0], size[1]), (0, 0, 0, 0)) 
    draw = ImageDraw.Draw(img)
    draw.rectangle( [0,0] + size, fill=fill) 
    img.save("rectangle.png")



