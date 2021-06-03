#!/usr/bin/env python
"""
img.py
======

::

    https://visibleearth.nasa.gov/collection/1484/blue-marble

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

def array_fromstring(ss):
    """
    :param ss: string such as "1280px_720px"
    :return a: int array of pixel dimensions 
    """
    return np.fromstring(ss.replace("px",""), sep="_", dtype=np.int32) 

class Img(object):
    @classmethod
    def parse_args(cls, doc):
        d = {}
        d["level"] = "INFO"
        d["format"] = "%(asctime)-15s %(levelname)-7s %(name)-20s:%(lineno)-3d %(message)s"
        d["paths"] = []
        d["slidesize"] = "1280px_720px"

        parser = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
        parser.add_argument('paths', nargs="*", default=d["paths"], help='base directory')
        parser.add_argument('--saveppm', action="store_true", default=False )
        parser.add_argument('--s5', action="store_true", default=False, help="Emit s5_background_image spec"  )
        parser.add_argument('--scale', type=int, default=1 )
        parser.add_argument('--level', default=d["level"], help='log level')
        parser.add_argument('--slidesize', default=d["slidesize"] )
        parser.add_argument('--savepng', action="store_true", default=False)
        parser.add_argument('--savepgm', action="store_true", default=False)
 
        args = parser.parse_args()

        args.slidesize = array_fromstring(args.slidesize)
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

    def save_as_pgm(self, scale=1):
        path = self.path
        size = self.size
        im, pfx = self.get_scaled(scale)
        pgm = path.replace(".jpg","%s.pgm" % pfx)

        if os.path.exists(pgm):
            log.info("pgm exists already at %s " % pgm)
        else:
            log.info("save_as_pgm to %s " % pgm )
            im.save(pgm)
        pass

    def save_as_png(self, scale=1):
        path = self.path
        size = self.size
        im, pfx = self.get_scaled(scale)
        png = path.replace(".jpg","%s.png" % pfx)

        if im.mode != "RGBA":
            log.info("converting from mode %s to RGBA" % im.mode)
            im2 = im.convert("RGBA")
        else:
            im2 = im 
        pass
        if os.path.exists(png):
            log.info("png exists already at %s " % png)
        else:
            log.info("save_as_png to %s " % png )
            im2.save(png)
        pass

if __name__ == '__main__':
    args = Img.parse_args(__doc__)

    imgs =  []
    for path in args.paths: 
        img = Img(path)
        imgs.append(img)
        if args.saveppm: 
            img.save_as_ppm(args.scale)
        elif args.savepng: 
            img.save_as_png(args.scale)
        elif args.savepgm: 
            img.save_as_pgm(args.scale)
        pass
    pass
   
    print(args.slidesize)
    for img in imgs:
        print(img.size)

 



