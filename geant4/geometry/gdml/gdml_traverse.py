#!/usr/bin/env python
import os

def traverse(path):
    path = os.path.expandvars(path)
    for line in file(path).readlines():
        print line,


if __name__ == '__main__':
    path =  '$LOCAL_BASE/env/geant4/geometry/gdml/g4_00.gdml' 
    traverse(path)


