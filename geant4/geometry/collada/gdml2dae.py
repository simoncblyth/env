#!/usr/bin/env python

import os
import xml.etree.cElementTree as ET
import collada as co

class GDML2DAE(object):
    def __init__(self, path, opts):
        path = os.path.expandvars(path)
        self.opts = opts  
        self.gdml = ET.parse(path).getroot()
        self.dae = co.Collada()
        self.count = 0  
        self.gtags = {}

    def _print(self, gdmlnode):
        print gdmlnode.tag, gdmlnode.attrib

    def _translate_material(self, gnode):
        tag = gnode.tag
        att = gnode.attrib
        print self.count, tag, att

    def _translate(self, gnode):
        self.count += 1 
        tag = gnode.tag
        att = gnode.attrib

        if self.count % self.opts.mod == 0:
            print self.count, tag, att

        if tag == 'material': 
            self._translate_material(gnode)

        if not tag in self.gtags:
            self.gtags[tag] = 0
        self.gtags[tag] += 1     

    def recurse(self, base, fn ):
        """
        Over the source GDML document structure
        """  
        fn(base)
        for child in base:
            self.recurse(child, fn)

    def report(self):
        for tag in self.gtags.keys():
            print "%15s %s " % ( tag, self.gtags[tag] )

    def __call__(self):
        self.recurse(self.gdml, self._translate)

    
class Opts(object):
    mod = 100000


if __name__ == '__main__':
    path = "$LOCAL_BASE/env/geant4/geometry/gdml/g4_01.gdml"



    gd = GDML2DAE(path, Opts())
    gd()
    gd.report()


