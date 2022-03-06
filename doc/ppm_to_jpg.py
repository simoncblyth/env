#!/usr/bin/env python
"""
ppm_to_jpg.py
================


"""

import sys, PIL


if __name__ == '__main__':
    ppmpath = sys.argv[1]
    assert ppmpath.endswith(".ppm")
    img = PIL.Image.open(ppmpath)
    jpgpath = ppmpath.replace(".ppm", ".jpg")
    img.save(jpgpath) 


    


