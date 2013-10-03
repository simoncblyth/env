#!/usr/bin/env python
"""
TODO:

#. CLI 

::

    simon:~ blyth$ gdml.py 
    INFO:env.geant4.geometry.gdml.gdml:examine_materials found 36 
    INFO:env.geant4.geometry.gdml.gdml:examine_solids found 707
    INFO:env.geant4.geometry.gdml.gdml:examined 249 volume 
    INFO:env.geant4.geometry.gdml.gdml:examined 5642 physvol 
    INFO:env.geant4.geometry.gdml.gdml:examine_setup found  World0xc6337a8  
    0 World0xc6337a8 /dd/Materials/Vacuum0xbaff828 WorldBox0xc6328f0
         0 /dd/Geometry/Sites/lvNearSiteRock0xb82e578 <Element 'position' at 0x1944aa0> <Element 'rotation' at 0x1944ab8>
    1 /dd/Geometry/Sites/lvNearSiteRock0xb82e578 /dd/Materials/Rock0xb849090 near_rock0xb8499c8
         0 /dd/Geometry/Sites/lvNearHallTop0xb745f10 <Element 'position' at 0x19449c8> None
         1 /dd/Geometry/Sites/lvNearHallBot0xb7dd4a8 <Element 'position' at 0x1944a10> None
    2 /dd/Geometry/Sites/lvNearHallTop0xb745f10 /dd/Materials/Air0xb830740 near_hall_top_dwarf0xb82f8e0
         0 /dd/Geometry/PoolDetails/lvNearTopCover0xbad46a0 <Element 'position' at 0x6a9c38> None
         1 /dd/Geometry/RPC/lvRPCMod0xb739980 <Element 'position' at 0x6a9c80> <Element 'rotation' at 0x6a9c98>
         2 /dd/Geometry/RPC/lvRPCMod0xb739980 <Element 'position' at 0x6a9ce0> <Element 'rotation' at 0x6a9cf8>
         3 /dd/Geometry/RPC/lvNearRPCRoof0xbab2040 <Element 'position' at 0x6a9d40> <Element 'rotation' at 0x6a9d58>


"""
import os, sys, logging
log = logging.getLogger(__name__)
import xml.etree.cElementTree as ET
#import xml.etree.ElementTree as ET

parse_ = lambda _:ET.parse(os.path.expandvars(_)).getroot()


def findone(elem, name, att=None):
    all = elem.findall(name)
    assert len(all) == 1, ( all, elem, name)
    if att:
        return all[0].attrib[att]
    return all[0]

class GDML(object):
    def __init__(self, path):
        self.gdml = parse_(path)
        self.volcount = 0 
        self.world = None
        self.material = {}
        self.solid = {}
        self.volume = {}
        self.physvol = {}
        self.examine()

    def namednode(self, node, assert_=None):
        tag, att = node.tag, node.attrib
        if assert_:
            assert tag == assert_, ("expecting element tag %s not %s " % (assert_, tag ))
        name = att['name']
        if name[-9:-7] == '0x':
            nam, id = name[0:-9], name[-9:]
        else:
            nam, id = name, self.count
        return tag, att, name, nam, id  

    def examine_physvol(self, physvol):
        tag, att, name, nam, id = self.namednode(physvol, assert_='physvol')
        volumeref = findone(physvol, 'volumeref', att='ref')
        position = physvol.find('position')
        rotation = physvol.find('rotation')
        self.physvol[name] = (volumeref,position,rotation)
        return name

    def examine_volume(self, volume):
        tag, att, name, nam, id = self.namednode(volume, assert_='volume')
        materialref = findone( volume,'materialref', att='ref')
        solidref = findone( volume, 'solidref', att='ref')
        pvs = []
        for physvol in volume.findall('physvol'): 
            pv = self.examine_physvol(physvol) 
            pvs.append(pv)
        self.volume[name] = (name, materialref, solidref, tuple(pvs))

    def examine_structure(self, structure):
        vols = list(structure)
        vols.reverse()   
        worldname = vols[0].attrib['name']
        assert worldname[0:5] == 'World', "World first not %s " % vols[0].attrib 
        for volume in vols:
            self.examine_volume(volume)
        pass    
        log.info("examined %s volume " % len(self.volume))    
        log.info("examined %s physvol " % len(self.physvol))    

    def examine_materials(self, materials):
        for material in materials.findall('material'):
            tag, att, name, nam, id = self.namednode(material, assert_='material')
            self.material[name] = (name)
        log.info("examine_materials found %s " % len(self.material))    

    def examine_solids(self, solids):
        for solid in solids:
            tag, att, name, nam, id = self.namednode(solid)
            self.solid[name] = (name)
        log.info("examine_solids found %s" % len(self.solid))    

    def examine_setup(self, setup):
        world = setup.find("world") 
        self.world = world.attrib['ref']
        log.info("examine_setup found  %s  " % self.world )    

    def examine(self):
        materials = self.gdml.find("materials")
        solids = self.gdml.find("solids")
        structure = self.gdml.find("structure")
        setup = self.gdml.find("setup")
        pass
        self.examine_materials(materials)
        self.examine_solids(solids)
        self.examine_structure(structure)
        self.examine_setup(setup)

    def check_volume(self, volume):
        lvname, materialref, solidref, pvs = volume
        assert lvname in self.volume
        assert materialref in self.material
        assert solidref in self.solid
        for pv in pvs:
            assert pv in self.physvol

    def walk(self, lv=None):
        if lv is None:
            lv = self.world
        volume = self.volume[lv]
        self.check_volume(volume)
        yield volume
        lvname, materialref, solidref, pvs = volume
        for pv in pvs:
            lv, position, rotation = self.physvol[pv]
            for _ in self.walk(lv): # recursive yield trick
                yield _


def main():
    logging.basicConfig(level=logging.INFO)
    args = sys.argv[1:]
    if len(args) == 0:
        path = "$LOCAL_BASE/env/geant4/geometry/gdml/g4_01.gdml"
    else:
        path = args[0]
    log.info("reading gdml from %s " % path )

    gdml = GDML(path)

    #for i,volume in enumerate(gdml.walk()):
    #    print i, volume 

    count = -1  
    for volume in gdml.walk():
        count += 1
        if count>10:break
        lvname, materialref, solidref, pvs = volume
        material = gdml.material[materialref]
        solid = gdml.solid[solidref]
        print count, lvname, material, solid 
        for ipv,pv in enumerate(pvs):
            lv, pos, rot = gdml.physvol[pv]
            print "    ", ipv, lv, pos, rot 
            #plv,pmat,psol,ppvs = gdml.volume[lv]


if __name__ == '__main__':
    main()



