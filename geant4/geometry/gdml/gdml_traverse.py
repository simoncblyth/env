#!/usr/bin/env python
import os, logging, re
log = logging.getLogger(__name__)

ptr = re.compile("0x[0-9a-fA-F]{7}$")

#from xml.etree import ElementTree as ET
import xml.etree.cElementTree as ET

class Parse(object):
    def __init__(self, path):
        path = os.path.expandvars(path)
        log.info("parsing %s " % path ) 
        root = ET.parse(path).getroot()
        log.info("completed parse")
        self.root = root 
        self.path = path
        #self.recurse(root)

        define, materials, solids, structure, setup = tuple([_ for _ in root])

        assert define.tag == 'define'
        assert materials.tag == 'materials'
        assert solids.tag == 'solids'
        assert structure.tag == 'structure'
        assert setup.tag == 'setup'

        self.define = define
        self.materials = materials
        self.solids = solids
        self.structure = structure
        self.setup = setup
        self.physvol_names = [] 
        self.recurse(structure, self.physvol_)

    def print_(self, node):
        print node.tag, node.attrib

    def physvol_(self, node):
        if node.tag != 'physvol':return
        uname = node.attrib['name']
        assert ptr.search(uname), "expecting last 9 chars of name to look like a pointer eg ..CrossRib10xc4f2a08  0xc4f2a08 "
        name = uname[:-9]
        print name

    def recurse(self, base, fn ):
        fn( base )
        for node in base:
            self.recurse(node, fn)

    def traverse(self):
        for line in file(self.path).readlines():
            print line,


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    #path =  '$LOCAL_BASE/env/geant4/geometry/gdml/g4_00.gdml'  # names truncated to 99 chars
    path =  '$LOCAL_BASE/env/geant4/geometry/gdml/g4_01.gdml'   # truncation lifted
    pr = Parse(path)
    print pr


