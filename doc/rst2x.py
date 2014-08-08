#!/usr/bin/env python
"""
Usage::

    python rst2x.py ~/e/muon_simulation/presentation/gpu_optical_photon_simulation.txt

#. uses docutils to parse restructured text document 
#. traverse the doctree, finding image nodes 

Giving unknown directive warnings for my s5 additions

#. s5_video
#. s5_background_image

"""
import sys, logging
try:
    import IPython as IP
except ImportError:
    IP = None

log = logging.getLogger(__name__)
from docutils.core import publish_doctree
import docutils.nodes

def parse( source ):
    return doctree

def main():
    logging.basicConfig(level=logging.INFO)

    path = sys.argv[1]
    log.info("reading %s " % path )
    with open(path,"r") as fp:
        source = fp.read()

    log.info("parsing %s chars " % len(source)) 
    doctree = publish_doctree(source)

    imgs = doctree.traverse(docutils.nodes.image)
    log.info("found %s images " % len(imgs))
    for img in imgs:
        print img['uri']

    #IP.embed()

if __name__ == '__main__':
    main()

