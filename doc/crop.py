#!/usr/bin/env python
"""
Usage example, cropping Safari screen captures of S5 slides
to make a set of png 

"""
import os, logging, sys
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

    def __init__(self, style_ ):
        if style_ in self.style:
            self.style_ = style_
            self.description = self.style[style_]['description']
            self.param = self.style[style_]['param']
        else:    
            raise Exception("style %s not handled " % style_ )

    def __repr__(self):
        return "%s %s %s " % ( self.__class__.__name__ , self.style_, self.description )

    def __call__(self, path):
        """
        All four coordinates are measured from the top/left corner, and describe the
        distance from that corner to the:
        
        #. left edge
        #. upper edge
        #. right edge 
        #. bottom edge

        """ 
        base, ext = os.path.splitext(path)
        cpath = base + "_crop" + ext 
        log.info( "cropping %s to create %s " % ( path, cpath ))  
        im = Image.open(path)
        width, height = im.size   

        # safari_headtail
        left = 0
        right = width

        upper = self.param[0]
        lower = height - self.param[1]

        box = (left, upper, right, lower)
        pass
        im = im.crop(box)
        im.save(cpath)


def main():
    logging.basicConfig(level=logging.INFO)
    crop = Crop("safari_headtail")
    log.info(crop)
    for path in sys.argv[1:]:
        if os.path.exists(path):
            if path[-4:] == '.png':
                crop(path)
            elif path[-4:] == '.pdf':
                log.info("PIL cannot handle cropping PDF ")
            else:
                pass


if __name__ == '__main__':
    main()


    


