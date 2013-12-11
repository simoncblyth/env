#!/usr/bin/env python
"""
daetex.py
==========

Generate PNG images for the textures referenced by a G4DAE generated 
collada document.


"""
import os, sys, logging
log = logging.getLogger(__name__)
import lxml.etree as ET
from PIL import Image, ImageDraw

NS = 'http://www.collada.org/2005/11/COLLADASchema'
parse_ = lambda _:ET.parse(os.path.expandvars(_)).getroot()


class Materials(object):
    PPE = 'blue' 
    MixGas = 'blue' 
    Air = 'blue' 
    Bakelite = 'blue' 
    Foam = 'blue' 
    Aluminium = 'blue' 
    Iron = 'blue' 
    GdDopedLS = 'blue' 
    Acrylic = 'red' 
    Teflon = 'blue' 
    LiquidScintillator = 'blue' 
    Bialkali = 'blue' 
    OpaqueVacuum = 'blue' 
    Vacuum = 'blue' 
    Pyrex = 'blue' 
    UnstStainlessSteel = 'blue' 
    PVC = 'blue' 
    StainlessSteel = 'blue' 
    ESR = 'blue' 
    Nylon = 'blue' 
    MineralOil = 'green' 
    BPE = 'blue' 
    Ge_68 = 'blue' 
    Co_60 = 'blue' 
    C_13 = 'blue' 
    Silver = 'blue' 
    Nitrogen = 'blue' 
    Water = 'blue' 
    NitrogenGas = 'blue' 
    IwsWater = 'blue' 
    ADTableStainlessSteel = 'blue' 
    Tyvek = 'blue' 
    OwsWater = 'blue' 
    DeadWater = 'blue' 
    RadRock = 'blue' 
    Rock = 'blue' 

    @classmethod
    def color(cls, name):
        return getattr(cls, name, 'grey')

    @classmethod
    def format(cls, names):
        hdr = "class Materials(object):"
        return "\n".join(["",hdr]+map(lambda _:"    %s = '%s' " % (_,cls.color(_))  , names )+[""])


class DAETex(list):
    def __init__(self, path ):
        list.__init__(self)
        path = os.path.abspath(os.path.expandvars(os.path.expanduser(path))) 
        dae = parse_(path)
        dir = os.path.dirname(path) 
        ilib = dae.find("{%s}library_images" % NS )
        self.dir = dir
        self[:] = map(lambda _:_.text, ilib.findall("{%s}image/{%s}init_from" % (NS,NS) ))

    def abspath(self, relpath):
        return os.path.abspath(os.path.join(self.dir, relpath))
    def name(self, relpath):
        return os.path.splitext(os.path.basename(relpath))[0]
    def paths(self):
        return map(self.abspath, self)
    def names(self):
        return map(self.name, self)

    def generate_texture(self, relpath):
        abspath = self.abspath(relpath)
        if os.path.exists(abspath):
            log.info("texture %s exists already " % abspath )
            return
        pass    
        name = self.name(relpath)
        color = Materials.color(name)
        log.info("creating %s : %s with color %s " % (name, abspath,color) )
        generate_image( abspath, color )

    def generate_textures(self):
        for r in self:
            self.generate_texture(r)

    def dump(self):    
        for r in self:
            n = self.name(r)
            p = self.abspath(r)
            c = Materials.color(n)
            log.info("%-25s : %-10s : %s " % (  n, c, p))

def daetex(path):
    dt = DAETex(path)
    #dt.dump()
    dt.generate_textures()
   
def generate_image( path, color, size=[100,100] ):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        log.info("creating dir %s " % dir )
        os.makedirs(dir)
    pass    
    img = Image.new('RGBA', (size[0], size[1]), (0, 0, 0, 0)) 
    draw = ImageDraw.Draw(img)
    draw.rectangle( [0,0] + size, fill=color) 
    img.save(path)

class Defaults(object):
    logformat = "%(asctime)s %(name)s %(levelname)-8s %(message)s"
    loglevel = "INFO"
    logpath = None

def parse_args(doc):
    from optparse import OptionParser
    defopts = Defaults()
    op = OptionParser(usage=doc)
    op.add_option("-o", "--logpath", default=defopts.logpath , help="logging path" )
    op.add_option("-l", "--loglevel",   default=defopts.loglevel, help="logging level : INFO, WARN, DEBUG. Default %default"  )
    op.add_option("-f", "--logformat", default=defopts.logformat , help="logging format" )

    opts, args = op.parse_args()
    level = getattr( logging, opts.loglevel.upper() )

    if opts.logpath:  # logs to file as well as console, needs py2.4 + (?)
        logging.basicConfig(format=opts.logformat,level=level,filename=opts.logpath)
        console = logging.StreamHandler()
        console.setLevel(level)
        formatter = logging.Formatter(opts.logformat)
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)  # add the handler to the root logger
    else:
        logging.basicConfig(format=opts.logformat,level=level)
    pass
    log.info(" ".join(sys.argv))
    return opts, args



def main():
    opts, args = parse_args(__doc__) 
    return daetex(args[0])

if __name__ == '__main__':
    main()
    

