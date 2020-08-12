#!/usr/bin/env python
"""
comb.py : grouped combination of PNGs vertically or horizontally
===================================================================

Combining PNG 2-by-2 : for example for paired annotated presentation pages

::

   mkdir /tmp/tt 
   cd /tmp/tt
   cp /tmp/simoncblyth.bitbucket.io/env/presentation/opticks_may2020_hsf_TALK/*_crop.png .

   comb.py -g2    ## 2-by-2 combination of all PNGs in current dir, in name sorted order 
   rm *_crop.png

   open .

   # adjust the sort order, select and then using scripting interface to make PDF from the PNG 


"""
import os, logging, sys
import argparse

log = logging.getLogger(__name__)
from PIL import Image 

class Comb(object):

    @classmethod
    def parse_args(cls, doc):
        d = {}
        d["level"] = "INFO"
        d["format"] = "%(asctime)-15s %(levelname)-7s %(name)-20s:%(lineno)-3d %(message)s"
        d["outpath"] = "test.png"
        d["mode"] = "vertical"
        d["group"] = 0
        d["paths"] = []
        d["dir" ] = "."

        parser = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
        parser.add_argument('paths', nargs="*", default=d["paths"], help='base directory')
        parser.add_argument('--level', default=d["level"], help='log level')
        parser.add_argument('-m','--mode', default=d["mode"], choices=["horizontal", "vertical"], help='vertical or horizontal')
        parser.add_argument('-g','--group', type=int, default=d["group"], help='number of items to combine or zero to combine all')
        parser.add_argument('-o','--outpath', default=d["outpath"], help='output file name')
        parser.add_argument('-d','--dir', default=d["dir"], help='directory to find paths in')
        
        args = parser.parse_args()
        logging.basicConfig(level=getattr(logging, args.level.upper()), format=d["format"])
        if not args.dir is None:
            args.paths = sorted(os.listdir(args.dir))
        pass 
        return args

    def __init__(self, paths, outpath, mode="vertical"):
        """
        :param paths: list of paths to PNGs to be combined into one 
        :param outpath:
        :param mode: 
        """
        self.paths = paths
        self.outpath = outpath
        self.mode = mode 

        images = [Image.open(x) for x in self.paths]
        widths, heights = zip(*(i.size for i in images))

        print("widths  : %s " % repr(widths))
        print("heights : %s " % repr(heights))

        self.images = images
        self.widths = widths
        self.heights = heights

    def combine_horizontal(self):
        tot_width = sum(self.widths)
        max_height = max(self.heights)
        new_im = Image.new('RGB', (tot_width, max_height))
        x_offset = 0
        for im in self.images:
            new_im.paste(im, (x_offset,0))
            x_offset += im.size[0]
        pass
        return new_im

    def combine_vertical(self):
        max_width = max(self.widths)
        tot_height = sum(self.heights)
        log.info("combine_vertical max_width %d tot_height %d " % ( max_width, tot_height ))
        new_im = Image.new('RGB', (max_width, tot_height))
        y_offset = 0
        for im in self.images:
            new_im.paste(im, (0, y_offset))
            y_offset += im.size[1]
        pass
        return new_im

    def combine(self):
        mode = self.mode
        log.info("combining with mode : %s " % mode)
        if mode == "horizontal":
            new_im = self.combine_horizontal()
        elif mode == "vertical":
            new_im = self.combine_vertical()
        else:
            new_im = None
        pass
        return new_im

    def save(self): 
        new_im = self.combine()
        path = self.outpath  
        log.info("saving to %s " % path )
        new_im.save(path)


if __name__ == '__main__':

    args = Comb.parse_args(__doc__)
    log.info("combine %d paths %s " % (len(args.paths), repr(args.paths)))
    group = args.group

    if group == 0:
        c = Comb(args.paths, args.outpath, args.mode)
        c.save()
    elif group == 1:
        log.info("nothing to do")
    elif group > 1:
        assert len(args.paths) % group == 0,  ("wrong number of paths for group size", len(args.paths), group )
        pass
        n = len(args.paths) / group 
        log.info(" combine %s paths with group size %d making %d new img " % (len(args.paths), group, n ))
        for i in range(n):
            paths = args.paths[i*group:(i+1)*group] 
            outpath = "%0.3d.png" % i                ## hmm this assumes the input paths are not of this form 000.png 001.png etc..
            print("%3d : %s : %s " % (i, repr(paths), outpath))
            c = Comb(paths, outpath, args.mode)
            c.save()
        pass
    pass




