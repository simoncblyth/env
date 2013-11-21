#!/usr/bin/env python

import sys
import collada
from collada.xmlutil import etree as ET
from collada.xmlutil import COLLADA_NS as NS


def meta(path):
    dae = collada.Collada(path)
    for g in dae.geometries:
        m =  g.xmlnode.find("{%(NS)s}mesh/{%(NS)s}extra/{%(NS)s}meta" % dict(NS=NS))
        print g 
        print ET.tostring(m)


def main():
    meta(sys.argv[1])





