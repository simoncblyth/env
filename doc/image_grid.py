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

def image_grid(imgs, rows, cols, annotate=False):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = PIL.Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        if annotate:
            image_annotate(img, str(i), pos=(0,0), fontsize=256 ) 
        pass
        grid.paste(img, box=(i%cols*w, i//cols*h))
    pass
    return grid

def image_downsize( img, factor ):
    width, height = img.size   
    dwidth = int(int(width)/factor)
    dheight = int(int(height)/factor)
    log.info("image_downsize factor %s from width %s height %s to dwidth %s dheight %s " % (factor, width, height, dwidth, dheight))
    imgd = img.resize( (dwidth, dheight), PIL.Image.ANTIALIAS)
    return imgd 

def get_fontpath():
    default_fontpath = os.path.expandvars("$OPTICKS_PREFIX/externals/imgui/imgui/extra_fonts/Cousine-Regular.ttf")
    fontpath = os.environ.get("FONTPATH", default_fontpath )
    return fontpath 

def image_annotate( img, txt, pos=(0,0), fontsize=16, rgb=(255,255,255) ):
    draw = ImageDraw.Draw(img)
    fontpath = get_fontpath()
    font = ImageFont.truetype(fontpath, fontsize)
    draw.text(pos, txt, rgb, font=font)

if __name__ == '__main__':
     logging.basicConfig(level=logging.INFO)

     all_paths = sys.argv[1:]
     side = int(math.sqrt(len(all_paths)))
     rows, cols = side, side 
     num = rows*cols

     grid_paths = all_paths[:num] 

     log.info("all_paths %d side %d rows %d cols %d grid_paths %d " % (len(all_paths), side, rows, cols, len(grid_paths)))

     log.info("\n".join(grid_paths))

     imgs = [PIL.Image.open(path) for path in grid_paths]

     annotate="ANNOTATE" in os.environ 
     grid = image_grid( imgs, rows, cols, annotate=annotate )


     #image_annotate(grid, "test", fontsize=400 )


     firstpath = grid_paths[0]
     griddir = os.path.dirname( firstpath )
     stub, ext = os.path.splitext( firstpath )

     outstem = os.environ.get("OUTSTEM", "image_grid")

     if "SAVE_FULL" in os.environ:
         gridpath0 = os.path.join( griddir, "%s_full%s" % (outstem, ext) ) 
         log.info("saving to %s " % gridpath0 )
         grid.save(gridpath0)
     pass

     gridpath1 = os.path.join( griddir, "%s%s" % (outstem, ext) ) 
     dgrid = image_downsize( grid,  side )
     log.info("saving to %s " % gridpath1 )
     dgrid.save(gridpath1)














