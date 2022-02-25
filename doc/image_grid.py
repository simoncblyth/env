#!/usr/bin/env python
"""
image_grid.py
================

Usage example::

    #!/bin/bash -l 

    paths=$(ls -1rt cxr_i0_t0_solidXJfixture:*:-3.jpg)
    outstem=image_grid_cxr_solidXJfixture:xx:-3

    export ANNOTATE=1     
    export OUTSTEM=$outstem

    ${IPYTHON:-ipython} ~/env/doc/image_grid.py $paths 

    ls -l $outstem*
    du -hs $outstem*

    name=$outstem.jpg 

    if [ -f "$name" ]; then 
        open $name
    fi 


"""
import os, logging, sys, math

import PIL
from PIL import ImageDraw
from PIL import ImageFont

log = logging.getLogger(__name__)

class IMG(object):
    @classmethod
    def Downsize(cls, img, factor ):
        width, height = img.size   
        dwidth = int(int(width)/factor)
        dheight = int(int(height)/factor)
        log.info("IMG.downsize factor %s from width %s height %s to dwidth %s dheight %s " % (factor, width, height, dwidth, dheight))
        imgd = img.resize( (dwidth, dheight), PIL.Image.ANTIALIAS)
        return imgd 

    @classmethod
    def Fontpath(cls):
        default_fontpath = os.path.expandvars("$OPTICKS_PREFIX/externals/imgui/imgui/extra_fonts/Cousine-Regular.ttf")
        fontpath = os.environ.get("FONTPATH", default_fontpath )
        return fontpath 

    @classmethod
    def Annotate( cls, img, txt, pos=(0,0), fontsize=16, rgb=(255,255,255) ):
        draw = ImageDraw.Draw(img)
        fontpath = cls.Fontpath()
        font = ImageFont.truetype(fontpath, fontsize)
        draw.text(pos, txt, rgb, font=font)

    @classmethod 
    def GridSpec(cls, all_paths, all_anno, rows, cols):
        """
        :param all_paths: list of paths to images
        :param all_anno: list of annotation strings or empty list
        :param rows: number of rows
        :param cols: number of cols
        :return gridspec: np.object array of shape (rows, cols) containing IMG 
        """
        num = rows*cols 
        gridspec = np.zeros([rows, cols], dtype=np.object )   
        num_gap = num - len(all_paths)
        for i in range(len(all_paths)):
            path = all_paths[i]
            anno = all_anno[i] if len(all_anno) == len(all_paths) else str(i)
            offset = num_gap if "HEADGAP" in os.environ else 0 
            gridspec.ravel()[offset+i] = cls(path, anno)
        pass
        return gridspec

    @classmethod 
    def Grid(cls, grid, annotate=False):
        """

             +---+---+---+---+
             |   |   |   |   |  0
             +---+---+---+---+
             |   |   |   |   |  1    rows
             +---+---+---+---+   
             |   |   |   |   |  2
             +---+---+---+---+
             |   |   |   |   |  3
             +---+---+---+---+
               0   1   2   3

                  columns 

        """
        assert len(grid.shape) == 2
        rows, cols = grid.shape

        x_imgs = grid[np.where(grid != 0)]
        assert len(x_imgs) > 0  

        first_img = x_imgs[0] 
        w, h = first_img.img.size   # size of first non-None img
        full = "SAVE_FULL" in os.environ
        gridpath = cls.GridPath(first_img, full=full)
        comp = PIL.Image.new('RGB', size=(cols*w, rows*h))
        
        for r in range(rows):
            for c in range(cols):
                obj = grid[r,c]
                if obj == 0: continue
                if obj.img is None: continue
                if annotate and not obj.anno is None:
                    cls.Annotate(obj.img, obj.anno, pos=(0,0), fontsize=256 ) 
                pass
                comp.paste(obj.img, box=(c*w, r*h))
            pass
        pass

        img = comp if full else IMG.Downsize( comp,  rows )
        return cls(path=gridpath, anno=None, img=img)

    @classmethod
    def GridPath(cls, first_img, full=False):
        firstpath = first_img.path 
        griddir = os.path.dirname( firstpath )
        stub, ext = os.path.splitext( firstpath )
        outstem = os.environ.get("OUTSTEM", "image_grid")
        gridpath = os.path.join( griddir, "%s%s%s" % (outstem, "_full" if full else "",   ext) ) 
        return gridpath 

    @classmethod 
    def OldGrid(cls, imgs, rows, cols, annotate=False):
        assert len(imgs) == rows*cols
        x_imgs = list(filter(lambda img:not img.img is None, imgs))
        w, h = x_imgs[0].img.size   # size of first non-None img

        comp = PIL.Image.new('RGB', size=(cols*w, rows*h))
        
        for i, img in enumerate(imgs):
            if img.img is None: continue
            if annotate and not img.anno is None:
                cls.Annotate(img.img, img.anno, pos=(0,0), fontsize=256 ) 
            pass
            comp.paste(img.img, box=(i%cols*w, i//cols*h))
        pass
        return comp

    def __init__(self, path=None, anno=None, img=None):

        self.path = path
        self.anno = anno

        if not img is None:
            self.img = img 
        else:
            if not path is None and os.path.exists(path): 
                img = PIL.Image.open(path)
            else:
                img = None
            pass
            self.img = img 
        pass
        
    def __repr__(self):
        return "IMG img %s anno %s path %s " % (self.img, self.anno, self.path)

    def save(self):
        log.info("save to %s " % self.path)
        self.img.save(self.path) 


if __name__ == '__main__':
     logging.basicConfig(level=logging.INFO)

     args = sys.argv[1:]

     if len(args) > 0:
         pathlist = args[0]
         all_paths = open(pathlist).read().splitlines()
     pass
     if len(args) > 1:
         annolist = args[1]
         all_anno = open(annolist).read().splitlines()
         assert len(all_anno) == len(all_paths)
     else:
         all_anno = []
     pass

     # rounds up, and leaves gaps so aim for the number of paths to be close to squares: 1,4,9,25,36,49,64,81,100  
     side = math.ceil(math.sqrt(len(all_paths)))    
     rows, cols = side, side 

     gridspec = IMG.GridSpec( all_paths, all_anno, rows, cols )

     log.info("all_paths %d all_anno %d grid.shape %s " % (len(all_paths), len(all_anno), str(gridspec.shape)))

     grid = IMG.Grid(gridspec, annotate="ANNOTATE" in os.environ )

     grid.save()


