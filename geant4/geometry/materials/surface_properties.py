#!/usr/bin/env python
"""

"""
import os
import lxml.etree as ET
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

    def dump(self):
        print self
        self.dump_props()

    def _get_data(self):
        data = {} 
        for matrix in self.elem.findall(".//{%s}matrix" % COLLADA_NS ):
            name, coldim, vals = matrix.attrib['name'],matrix.attrib['coldim'],matrix.attrib['values'] 
            assert coldim == '2'
            data[name] = vals
        return data

    def dump_props(self):
        for prop in self.elem.findall(".//{%s}property" % COLLADA_NS ):
            name, ref = prop.attrib['name'], prop.attrib['ref']
            vals = self.data[ref] 
            print "    %-20s : (%s)[%s] " % (name, len(vals),vals) 


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


    def dump_opticalsurface(self):
        for ksurf in sorted(self.optical, key=lambda _:self.optical[_].index):
            osurf = self.optical[ksurf]
            osurf.dump()

    def dump_surface(self, skin_or_border, check=True):
        assert skin_or_border in "skin border".split()  
        for i, surf in  enumerate(self.xml.findall(".//{%s}%ssurface" % (COLLADA_NS,skin_or_border) )):
            print "%-2s" % i , surf.attrib
            name = surf.attrib['surfaceproperty']
            osurf = self.optical[name]
            if check:
                esurf_0 = self.xml.find(".//{%s}opticalsurface[@name='%s']" % (COLLADA_NS, name))
                esurf_1 = osurf.elem
                assert esurf_0 == esurf_1, (esurf_0, esurf_1)
            osurf.dump()

    def dump(self):
        print "\n" + "#" * 10, " optical surface \n"
        self.dump_opticalsurface()
        print "\n" + "#" * 10, " skin surface \n"
        self.dump_surface('skin')
        print "\n" + "#" * 10, " border  surface \n"
        self.dump_surface('border')
        print "\n" + "#" * 10, " counts \n"
        print self.counts


if __name__ == '__main__':
    dae = DAE("g4_00.dae.4")
    #dae.dump_opticalsurface()
    dae.dump()


