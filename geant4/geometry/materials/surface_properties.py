#!/usr/bin/env python
"""

::

    g4pb:materials blyth$ ./surface_properties.py g4_00.dae.6 __dd__Geometry__AdDetails__AdSurfacesAll__ESRAirSurfaceTop REFLECTIVITY

    ##########  optical surface 

    2    __dd__Geometry__AdDetails__AdSurfacesAll__ESRAirSurfaceTop               1 0 0 0 nskin:0 nborder:1 
        REFLECTIVITY         : (31)  
    [[  1.55000000e-06   9.85050000e-01]
     [  1.63000000e-06   9.84060000e-01]
     [  1.68000000e-06   9.67230000e-01]
     ..., 
     [  6.20000000e-06   9.90000000e-03]
     [  1.03300000e-05   9.90000000e-03]
     [  1.55000000e-05   9.90000000e-03]]

    ##########  skin surface 


    ##########  border  surface 

    0  {'surfaceproperty': '__dd__Geometry__AdDetails__AdSurfacesAll__ESRAirSurfaceTop', 'name': '__dd__Geometry__AdDetails__AdSurfacesAll__ESRAirSurfaceTop'}
    2    __dd__Geometry__AdDetails__AdSurfacesAll__ESRAirSurfaceTop               1 0 0 0 nskin:0 nborder:1 
        REFLECTIVITY         : (31)  
    [[  1.55000000e-06   9.85050000e-01]
     [  1.63000000e-06   9.84060000e-01]
     [  1.68000000e-06   9.67230000e-01]
     ..., 
     [  6.20000000e-06   9.90000000e-03]
     [  1.03300000e-05   9.90000000e-03]
     [  1.55000000e-05   9.90000000e-03]]

    ##########  counts 

    {'bordersurface': 8, 'skinsurface': 34, 'opticalsurface': 42}
    g4pb:materials blyth$ 


"""
import os
import sys
import lxml.etree as ET
from common import as_optical_property_vector

parse_ = lambda _:ET.parse(os.path.expandvars(_)).getroot()
COLLADA_NS = "http://www.collada.org/2005/11/COLLADASchema"

class OpticalSurface(object):
    def __init__(self, elem, index=-1):
        self.elem = elem
        self.index = index
        self.data = self._get_data()
        self.skin = []
        self.border = []

    def __repr__(self):
        return "%-2s " % self.index + "  %(name)-70s   %(model)s %(finish)s %(type)s %(value)s " % self.elem.attrib + "nskin:%s nborder:%s " % (len(self.skin),len(self.border)) 

    def dump(self, qpr):
        print self
        self.dump_props(qpr)

    def _get_data(self):
        data = {} 
        for matrix in self.elem.findall(".//{%s}matrix" % COLLADA_NS ):
            name = matrix.attrib['name']
            assert matrix.attrib['coldim'] == '2'
            data[name] = as_optical_property_vector( matrix.text )
        return data

    def dump_props(self,qpr):
        for prop in self.elem.findall(".//{%s}property" % COLLADA_NS ):
            name, ref = prop.attrib['name'], prop.attrib['ref']
            if len(qpr)==0 or name.startswith(qpr):
                opv = self.data[ref] 
                print "    %-20s : (%s)  " % (name, len(opv)) 
                print opv


class DAE(object):
    def __init__(self, path):
        self.xml = parse_(path)
        self.optical = {}
        self.skin = {}
        self.border = {}
        self.counts = {}
        self.make_index()


    def make_index(self):
        opticalsurface = self.xml.findall(".//{%s}opticalsurface" % COLLADA_NS )
        skinsurface = self.xml.findall(".//{%s}skinsurface" % COLLADA_NS )
        bordersurface = self.xml.findall(".//{%s}bordersurface" % COLLADA_NS )
        self.counts = dict(opticalsurface=len(opticalsurface),skinsurface=len(skinsurface),bordersurface=len(bordersurface))
        assert len(skinsurface) + len(bordersurface) == len(opticalsurface) , self.counts # every opticalsurface occurs referred from one skin or border surface

        for i, elem in  enumerate(opticalsurface):
            self.optical[elem.attrib['name']] = OpticalSurface(elem, i )

        for i, elem in  enumerate(skinsurface):
            name = elem.attrib['surfaceproperty']
            osurf = self.optical[name]
            osurf.skin.append( elem )

        for i, elem in  enumerate(bordersurface):
            name = elem.attrib['surfaceproperty']
            osurf = self.optical[name]
            osurf.border.append( elem )


    def dump_opticalsurface(self, qid="", qpr=""):
        for ksurf in sorted(self.optical, key=lambda _:self.optical[_].index):
            if len(qid) == 0 or ksurf.startswith(qid):
                osurf = self.optical[ksurf]
                osurf.dump(qpr)

    def dump_surface(self, skin_or_border, qid, qpr, check=True):
        assert skin_or_border in "skin border".split()  
        for i, surf in  enumerate(self.xml.findall(".//{%s}%ssurface" % (COLLADA_NS,skin_or_border) )):

            name = surf.attrib['surfaceproperty']
            if len(qid) == 0 or name.startswith(qid):
                print "%-2s" % i , surf.attrib
                osurf = self.optical[name]
                if check:
                    esurf_0 = self.xml.find(".//{%s}opticalsurface[@name='%s']" % (COLLADA_NS, name))
                    esurf_1 = osurf.elem
                    assert esurf_0 == esurf_1, (esurf_0, esurf_1)
                osurf.dump(qpr)

    def dump(self, qid="", qpr=""):
        print "\n" + "#" * 10, " optical surface \n"
        self.dump_opticalsurface(qid,qpr)
        print "\n" + "#" * 10, " skin surface \n"
        self.dump_surface('skin',qid,qpr)
        print "\n" + "#" * 10, " border  surface \n"
        self.dump_surface('border',qid,qpr)
        print "\n" + "#" * 10, " counts \n"
        print self.counts


if __name__ == '__main__':

    args = sys.argv[1:]
    narg = len(args)

    pth, qid, qpr = "g4_00.dae","", ""
    if narg > 0:pth = args[0]
    if narg > 1:qid = args[1]
    if narg > 2:qpr = args[2]

    dae = DAE(pth)
    dae.dump(qid,qpr)


