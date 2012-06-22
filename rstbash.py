#!/usr/bin/env python
import os, logging
from env.sphinxext.bashrst import bashrst 


class BashRst(object): 
    def __init__(self, srcdir=None, gendir=None, suffix=".bash"):
        if not srcdir:
            srcdir = os.getcwd()
        if not gendir: 
            gendir = os.path.join(srcdir, "_bashrst" )
        if not os.path.isdir(gendir):
   	    os.makedirs(gendir)   
	pass    
        self.srcdir = srcdir
        self.gendir = gendir
	self.suffix = suffix

    def walk(self):
        for dirpath, dirs, names in os.walk(self.srcdir):
            rdir = dirpath[len(self.srcdir)+1:]
            for name in names:
                root, ext = os.path.splitext(name)
                if not ext == self.suffix: 
                    continue
                path = os.path.join(dirpath, name)
                self.handle(path)

    def handle(self, path):
	print path    
        conv = bashrst(path, self.srcdir, delim="EOU", gbase=self.gendir )
        print conv


if __name__ == '__main__':
 
    br = BashRst()
    br.walk()

