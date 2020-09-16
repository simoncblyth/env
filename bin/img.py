#!/usr/bin/env python
"""
img.py
======

::

    img.py ~/opticks_refs/Earth_Albedo_8192_4096.jpg --saveppm

    (base) epsilon:bin blyth$ du -h /Users/blyth/opticks_refs/Earth_Albedo_*
    4.2M	/Users/blyth/opticks_refs/Earth_Albedo_8192_4096.jpg
     96M	/Users/blyth/opticks_refs/Earth_Albedo_8192_4096.ppm


    img.py ~/opticks_refs/Earth_Albedo_8192_4096.jpg --saveppm --scale 8 


    (base) epsilon:npy blyth$ img.py ~/opticks_refs/Earth_Albedo_8192_4096.jpg --saveppm --scale 8 
    2020-08-21 19:46:20,822 INFO    __main__            :49  loaded /Users/blyth/opticks_refs/Earth_Albedo_8192_4096.jpg size array([8192, 4096], dtype=int32) 
    2020-08-21 19:46:20,822 INFO    __main__            :62  resize scale 8 : reduce from from array([8192, 4096], dtype=int32) to (1024, 512) 
    2020-08-21 19:46:21,505 INFO    __main__            :77  save_as_ppm to /Users/blyth/opticks_refs/Earth_Albedo_8192_4096_scaled_8.ppm 
    (base) epsilon:npy blyth$ open /Users/blyth/opticks_refs/Earth_Albedo_8192_4096_scaled_8.ppm 



* https://auth0.com/blog/image-processing-in-python-with-pillow/



"""
import numpy as np
import os, logging, sys
import argparse

log = logging.getLogger(__name__)
from PIL import Image 

class Img(object):
    @classmethod
    def parse_args(cls, doc):
        d = {}
        d["level"] = "INFO"
        d["format"] = "%(asctime)-15s %(levelname)-7s %(name)-20s:%(lineno)-3d %(message)s"
        d["paths"] = []

        parser = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
        parser.add_argument('paths', nargs="*", default=d["paths"], help='base directory')
        parser.add_argument('--saveppm', action="store_true", default=False )
        parser.add_argument('--scale', type=int, default=1 )
        parser.add_argument('--level', default=d["level"], help='log level')
        
        args = parser.parse_args()
        logging.basicConfig(level=getattr(logging, args.level.upper()), format=d["format"])
        return args

    def __init__(self, path):
        assert os.path.exists(path)
        im = Image.open(path)
        size = np.array(im.size, dtype=np.int32)
        log.info("loaded %s size %r " % (path, size))
        self.path = path 
        self.im = im
        self.size = size    

    def get_scaled(self, scale):
        assert scale in [1,2,4,8,16,32,64,128,256,512,1024] 
        if scale == 1:
            im = self.im
            pfx = ""
        else:
            newsize = tuple(map(int, self.size/scale))
            pfx = "_scaled_%d" % scale 
            log.info("resize scale %d : reduce from from %r to %r " % (scale, self.size, newsize) ) 
            im = self.im.resize(newsize)
        pass
        return im, pfx


    def save_as_ppm(self, scale=1):
        path = self.path
        size = self.size
        im, pfx = self.get_scaled(scale)
        ppm = path.replace(".jpg","%s.ppm" % pfx)

        if os.path.exists(ppm):
            log.info("ppm exists already at %s " % ppm)
        else:
            log.info("save_as_ppm to %s " % ppm )
            im.save(ppm)
        pass


if __name__ == '__main__':
    args = Img.parse_args(__doc__)
    for path in args.paths: 
        img = Img(path)
        if args.saveppm: 
            img.save_as_ppm(args.scale)
        pass
    pass
    



