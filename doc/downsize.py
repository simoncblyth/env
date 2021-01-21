#!/usr/bin/env python
"""
Usage example, downsizing retina png screen captures for non-retina usage


pdflatex is complaining about some screenshots::

   libpng warning: iCCP: known incorrect sRGB profile

Testing with non-downsized gives the same error so 
presumably there are different versions of libpng being 
used to make the screenshot PNGs and to read them 
in pdflatex.

* https://wiki.archlinux.org/index.php/Libpng_errors

Some changes in libpng 1.6+ cause it issue warning or even not work correctly with the original HP/MS sRGB profile. 
The old profile uses a D50 whitepoint, where D65 is standard.

Fixed using Preview.app Tools>Assign profile.. and picking "Generic RGB"



https://stackoverflow.com/questions/2498875/how-to-invert-colors-of-image-with-pil-python-imaging

   
image = Image.open('your_image.png')
inverted_image = PIL.ImageOps.invert(image)
inverted_image.save('new_name.png')



"""
import os, logging, sys
log = logging.getLogger(__name__)
from PIL import Image 


import PIL.ImageOps    


def fmt(width, height):
    return "%dpx_%dpx" % (width, height)

class Resize(object):
    def __init__(self, factor=2, suffix="_half"):
        self.factor = int(factor)
        self.suffix = suffix

    def __repr__(self):
        return "%s %s  " % ( self.__class__.__name__ , self.factor  )

    def __call__(self, path, invert=False):
        """
        """ 
        base, ext = os.path.splitext(path)
        dpath = base + self.suffix + ext 
        im = Image.open(path)
        width, height = im.size   
        factor = self.factor

        dwidth = int(int(width)/factor)
        dheight = int(int(height)/factor)


        from_ = fmt(width, height)
        to_ = fmt(dwidth, dheight)

        log.info( "downsize %s to create %s %s -> %s " % ( path, dpath, from_, to_ ))  
        imd = im.resize((dwidth, dheight), Image.ANTIALIAS) 

        if invert:
            imd = PIL.ImageOps.invert(imd)
        pass

        imd.save(dpath)


def main():
    logging.basicConfig(level=logging.INFO)
    downsize = Resize(factor=2, suffix="_half")
    log.info(downsize)

    invert = os.path.basename(sys.argv[0]) == "downsize_invert.py"

    for path in sys.argv[1:]:
        if os.path.exists(path):
            if path[-4:] == '.png':
                downsize(path, invert=invert)
            elif path[-4:] == '.pdf':
                log.info(".pdf downsizing not covered by PIL ?")
            else:
                pass


if __name__ == '__main__':
    main()


    


