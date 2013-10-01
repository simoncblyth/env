#!/usr/bin/env python
"""

Colons are mentioned as not being allowed but I supect the problem is the slash "/"

* http://www.schemacentral.com/sc/xsd/t-xsd_NCName.html

::

    PYTHONPATH=$(g4py-libdir):$PYTHONPATH python g4gdml.py


Validation problem with names::

    G4GDML: VALIDATION ERROR! Datatype error: Type:InvalidDatatypeValueException, Message:Value '/dd/Geometry/Sites/lvNearSiteRock0xb82e578' is not valid NCName . at line: 30935
    G4GDML: VALIDATION ERROR! Datatype error: Type:InvalidDatatypeValueException, Message:Value '/dd/Structure/Sites/db-rock0xc633af8_pos' is not valid NCName . at line: 30936
    G4GDML: VALIDATION ERROR! Datatype error: Type:InvalidDatatypeValueException, Message:Value '/dd/Structure/Sites/db-rock0xc633af8_rot' is not valid NCName . at line: 30937
    G4GDML: Reading definitions...
    G4GDML: Reading materials...
    G4GDML: Reading solids...
    G4GDML: Reading structure...
    G4GDML: Reading setup...
    G4GDML: Reading '/data1/env/local/env/geant4/geometry/gdml/g4_01.gdml' done!
    Stripping off GDML names of materials, solids and volumes ...

    In [10]: 

    In [13]: vpv = prs.GetWorldVolume()

    In [14]: vpv
    Out[14]: <Geant4.G4geometry.G4PVPlacement object at 0x9ad4a3c>

"""
import os, sys
sys.path.insert(1,os.path.expandvars('$DYB/external/build/LCG/geant4.9.2.p01/environments/g4py/lib'))
import Geant4

class Traverse(object):
    def __init__(self, world):
        self.world = world
        self.count = 0

    def print_(self, pv, lv, nd):
        pvn = pv.GetName()
        lvn = lv.GetName()
        print "%6s %3s %-50s %s " % ( self.count, nd, lvn, pvn )

    def recurse(self, pv, fn):
        lv = pv.GetLogicalVolume()
        nd = lv.GetNoDaughters()
        fn(pv, lv, nd) 
        self.count += 1 
        for i in range(nd): 
            dpv = lv.GetDaughter(i)
            self.recurse(dpv, fn)

    def __call__(self):
        self.recurse(self.world, self.print_ )


if __name__ == '__main__':
    path = "$LOCAL_BASE/env/geant4/geometry/gdml/g4_01.gdml"
    prs = Geant4.G4GDMLParser()
    validate = False
    prs.Read(os.path.expandvars(path),validate)
    vpv = prs.GetWorldVolume()
    trv = Traverse(vpv)
    trv()




