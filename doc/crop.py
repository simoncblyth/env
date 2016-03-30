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
    style = { 
          'safari_headtail_old':{
                                 'description':"vertically chop the head by param[0] and tail by param[1]",
                                 'param':(120, 20 ),  
                            },

          'safari_headtail':{
                                 'description':"vertically chop the head by param[0] and tail by param[1]",
                                 'param':(148, 30 ),  # formerly (190,30) use Preview.app measurement tool to count the pixels
                            },


            }

    def __init__(self, args ):
        self.args = args
        style_ = args.style

        if style_ in self.style:
            self.style_ = style_
            self.description = self.style[style_]['description']
            self.param = self.style[style_]['param']
        else:    
            self.style_ = None
            self.description = args.style
        pass

    def __repr__(self):
        return "%s %s %s " % ( self.__class__.__name__ , self.style_, self.description )


    def product_path(self, path):
        args = self.args
        base, ext = os.path.splitext(path)
        if ext != args.ext:
            log.warning("converting ext from %s to %s " % (ext, args.ext))
            ext = args.ext
        pass
        cpath = base + "_crop" + ext 
        return cpath 

    def __call__(self, path):
        """
        All four coordinates are measured from the top/left corner, and describe the
        distance from that corner to the:
        
        #. left edge
        #. upper edge
        #. right edge 
        #. bottom edge

        """ 
        args = self.args
        cpath = self.product_path(path)

        log.info( "cropping %s to create %s " % ( path, cpath ))  
        im = Image.open(path)
        width, height = im.size   

        # safari_headtail

        if self.style_ is None:
            box = (args.left, args.top, args.left+args.width,  args.top + args.height )
        elif self.style_.startswith("safari_headtail"):
            left = 0
            right = width
            upper = self.param[0]
            lower = height - self.param[1]
            box = (left, upper, right, lower)
        else:
            assert "unexpected style %s " % self.style_
        pass

        log.info("width %s height %s cropping to box %s " % (width, height, repr(box)))
        pass
        im = im.crop(box)
        im.save(cpath)







def main():
    parser = argparse.ArgumentParser()
    d = {}

    d['level'] = "INFO"
    d['style'] = "safari_headtail"
    d['path'] = ""
    d['left'] = 0
    d['top'] = 0
    d['width'] = 2560
    d['height'] = 1440
    d['ext'] = ".png"
 
    parser.add_argument("--level", default=d['level'] ) 

    parser.add_argument("--left", default=d['left'], type=int ) 
    parser.add_argument("--top", default=d['top'], type=int ) 
    parser.add_argument("--width", default=d['width'], type=int) 
    parser.add_argument("--height", default=d['height'], type=int)

    parser.add_argument("--style", default=d['style'] )
    parser.add_argument("--ext", default=d['ext'] )

    parser.add_argument("path", nargs='*', default=d['path'] )

    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.level.upper()),format="%(asctime)s %(name)s %(levelname)-8s %(message)s" )

    crop = Crop(args)
    log.info(crop)

    for path in args.path:
        if os.path.exists(path):
            cpath = crop.product_path(path)
            if os.path.exists(cpath):
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


    


