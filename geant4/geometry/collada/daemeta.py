#!/usr/bin/env python

import sys
import collada
from collada.xmlutil import etree as ET
from collada.xmlutil import COLLADA_NS as NS


def geom_meta(dae):
    for g in dae.geometries:
        m =  g.xmlnode.find("{%(NS)s}mesh/{%(NS)s}extra/{%(NS)s}meta" % dict(NS=NS))
        print g 
        print ET.tostring(m)

def bound_meta(dae):
    boundgeom = list(dae.scene.objects('geometry'))
    for g in boundgeom:
        print g 
         

def main():
    path = sys.argv[1]
    dae = collada.Collada(path)
    bound_meta(dae)





