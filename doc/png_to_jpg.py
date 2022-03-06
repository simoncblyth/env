#!/usr/bin/env python
"""
png_to_jpg.py
================

i ~/env/doc/png_to_jpg.py 



"""

import sys, PIL


if __name__ == '__main__':
    pngpath = sys.argv[1]
    assert pngpath.endswith(".png")
    img = PIL.Image.open(pngpath)
    jpgpath = pngpath.replace(".png", ".jpg")
    img.save(jpgpath) 





    


