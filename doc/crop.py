#!/usr/bin/env python
"""
Usage example, cropping Safari screen captures of S5 slides
to make a set of png 

"""
import os, logging, sys
import argparse

log = logging.getLogger(__name__)
from PIL import Image 

class Crop(object):
    def __init__(self, args ):
        self.args = args
        self.style = args.style
        self.stylename = args.style.__class__.__name__
        self.description = self.style.description
        self.param = self.style.param

    def __repr__(self):
        return "%s %s %s " % ( self.__class__.__name__ , self.stylename, self.description )

    def product_path(self, path):
        args = self.args
        base, ext = os.path.splitext(path)

        if ext != args.ext:
            log.warning("converting ext from %s to %s " % (ext, args.ext))
            ext = args.ext
        pass

        UNCROPPED = "_uncropped" 
        if base.endswith(UNCROPPED):
            cpath = base[:-len(UNCROPPED)] + ext 
        else:
            cpath = base + "_crop" + ext 
        pass
        return cpath 

    def __call__(self, path):
        """
        Cropping box is specified by 4 values::  

            box = (left, upper, right, lower)

        All four coordinates are measured from the top/left corner, and describe the
        distance from that corner to the:

        #. left edge
        #. upper edge
        #. right edge 
        #. bottom edge

               +----------------------+
               |    param[0]          |
               +----------------------+
               |                      |
               |                      |
               |                      |
               +----------------------+
               |     param[1]         ||
               +----------------------+

        """ 
        args = self.args
        cpath = self.product_path(path)

        log.info( "cropping %s to create %s " % ( path, cpath ))  
        im = Image.open(path)
        width, height = im.size   

        if self.style is None:
            left = args.left
            upper = args.top
            right = left+args.width
            lower = args.top + args.height  
        else:
            left = 0
            right = width
            upper = self.style.param[0]
            lower = height - self.style.param[1]
        pass
        box = (left, upper, right, lower)
        pass

        log.info("width %s height %s cropping to box %s " % (width, height, repr(box)))
        pass
        im = im.crop(box)
        im.save(cpath)




class OLD_SAFARI(object):
    """
    Preview.app cropping tool gives pixel dimensions to use when 
    pulling out manual crops that can be used to set these param 

    Old crop::

          +-----------------------------------------+
          |                                         |   148
          +-----------------------------------------+
          |                                         |
          |                                         |
          |                                         |
          |                                         |
          |                                         |
          +~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~+  
          +-----------------------------------------+    30
           
                                                    -----------
                                                        178 
                                                    -----------

    **Issue : Over cropping at top of slides**

    Pixel dimensions::

        Original : 2560x1562
        Cropped  : 2560x1382    # 180 pixels trimmed, more at top than bottom 

    Examining uncropped using the Preview.app selection tool 
    suggests that top crop should be 120 (not 148) and there 
    is no longer a need for a bottom crop. However for the desired 16/9 
    aspect its better to crop by 122, as shown below::

        In [7]: 2560/(16/9)
        Out[7]: 1440.0

        In [9]: 2560/(1562-120)
        Out[9]: 1.7753120665742026

        In [10]: 2560/(1562-121)
        Out[10]: 1.7765440666204024

        In [11]: 2560/(1562-122)
        Out[11]: 1.7777777777777777

    """
    description = "vertically chop the head by param[0] and tail by param[1]"
    #param = (148, 30 )
    param = (122, 0 )

class SAFARI(object):
    description = "vertically chop the head by param[0] only"
    param = (159, 0 )

class PYVISTA(object):
    description = "vertically chop the window chrome at the top only"
    param = (44,0)

class MATPLOTLIB(object):
    description = "chop the window chrome at top and bottom"
    param = (44,74)
    
class GENERIC(object):
    description = "chop the window chrome at top and bottom"
    param = (44,74)
     


def main():
    parser = argparse.ArgumentParser()
    d = {}
 
    d['level'] = "INFO"
    parser.add_argument("--level", default=d['level'] ) 

    # left top width height are only used when style is NONE
    d['left'] = 0
    d['top'] = 0
    d['width'] = 2560
    d['height'] = 1440
    parser.add_argument("--left", default=d['left'], type=int ) 
    parser.add_argument("--top", default=d['top'], type=int ) 
    parser.add_argument("--width", default=d['width'], type=int) 
    parser.add_argument("--height", default=d['height'], type=int)


    d['style'] = "safari"
    d['ext'] = ".png"
    d['replace'] = False
    parser.add_argument("--style", default=d['style'] )
    parser.add_argument("--ext", default=d['ext'] )
    parser.add_argument("--replace", action="store_true", default=d['replace'] )

    d['path'] = ""
    parser.add_argument("path", nargs='*', default=d['path'] )

    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.level.upper()),format="%(asctime)s %(name)s %(levelname)-8s %(message)s" )

    styles = {"safari":SAFARI, "pyvista":PYVISTA, "matplotlib":MATPLOTLIB, "generic":GENERIC } 
    style_kls = styles.get(args.style, None)
    log.info("args.style  %s  style_kls.description  %s " % (args.style, style_kls.description ))
    args.style = style_kls

    crop = Crop(args)
    log.info(crop)

    for path in args.path:
        if os.path.exists(path):
            cpath = crop.product_path(path)
            if os.path.exists(cpath) and not args.replace:
                log.info("product path exists already %s " % cpath )
            else:
                ext = path[-4:]
                if ext in [".png",".jpg"]:
                    crop(path)
                elif ext == '.pdf':
                    log.info("PIL cannot handle cropping PDF ")
                else:
                    log.info("PIL cannot handle image type %s " % path )
                pass
            pass


if __name__ == '__main__':
    main()


    


