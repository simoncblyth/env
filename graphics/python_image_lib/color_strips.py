#!/usr/bin/env python
"""
* http://ju.outofmemory.cn/entry/18060
"""
from PIL import Image, ImageDraw

def rectangle(img, box, innerfill, outerfill, width):
    """
    :param img:
    :param box: outer rectangle dimensions 
    :param innerfill:
    :param outerfill:
    :param width: difference between inner and outer rectangles

    ::

           w          
          +----------+
          | +------+ |
          | |      | |
          | |      | |
          | +------+ |
          +----------+  
        b[0]........b[2]............. X 

    """ 
    draw = ImageDraw.Draw(img)
    draw.rectangle(box, fill=outerfill) 
    draw.rectangle( 
        (box[0] + width, box[1] + width, 
         box[2] - width, box[3] - width),
        fill=innerfill
    )
    cxy = map(float, [(box[0]+box[2])/2.,(box[1]+box[3])/2.])
    draw.text(cxy, innerfill )


def color_strips( cols, size, bkd="black"):
    """
    :param cols:
    :param size:
    :param bkd: 
    """ 
    ncol = len(cols)
    py = float(size[1])/ncol
    img = Image.new('RGBA', (size[0], size[1]), (0, 0, 0, 0)) 
    for icol in range(ncol):
        rectangle(img, (0, py*icol, size[0], py*(icol+1)), cols[icol], bkd, 5)
    return img 


if __name__ == '__main__':
    pass
    cols = "lightblue red green blue cyan magenta yellow grey black white".split() 
    img = color_strips( cols, (400, 800))
    img.save("/tmp/color_strips.png")



