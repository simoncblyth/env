#!/usr/bin/env python
"""

"""
from __future__ import with_statement
import os, logging
log = logging.getLogger(__name__)
from env.sphinxext.bashrst import bashrst 


class BashRst(object): 
    def __init__(self, srcdir=None, gendir=None, suffix=None, outdir=None ):
        if not srcdir:srcdir = os.getcwd()
	if not outdir:outdir = "_docs"
        if not gendir:gendir = os.path.join(srcdir, outdir )
        if not suffix:suffix = ".bash"	
        if not os.path.isdir(gendir):
   	    os.makedirs(gendir)   
	pass    
        self.srcdir = srcdir
        self.gendir = gendir
	self.suffix = suffix

    def allwalk(self):
	"""
	Walking all .bash (not currently used)
	"""
        for dirpath, dirs, names in os.walk(self.srcdir):
            rdir = dirpath[len(self.srcdir)+1:]
            for name in names:
                root, ext = os.path.splitext(name)
                if not ext == self.suffix: 
                    continue
                path = os.path.join(dirpath, name)
                conv = bashrst(path, self.srcdir, delim="EOU", gbase=self.gendir, kids=False )
                print conv

    def walk(self, path):
	"""
	:param path: to root bash file eg ``env.bash``

	Only files linked via toctree are walked
	"""
        conv, paths = bashrst(path, self.srcdir, delim="EOU", gbase=self.gendir, kids=True )
        for p in paths:
            self.walk( p + ".bash" )		
        

def main(root):
    br = BashRst()
    br.walk(root)


if __name__ == '__main__':
    main("env.bash")

