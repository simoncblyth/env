#!/usr/bin/env python
"""
XMLDAE
=======

XML level view of a collada .dae, for debugging 

Objective is not to recreate pycollada, but merely to 
be a convenient debugging tool to ask XML based questions 
of the DAE.


Usage
------

Specify topnode id string or index on commandline, or no args to look at all::

    simon:collada blyth$ ./xmldae.py _dd_Geometry_PoolDetails_lvOutInWaterPipeNearTub0xb91c4b0  230 231 
      230  _dd_Geometry_PoolDetails_lvOutInWaterPipeNearTub0xb91c4b0                                             0    #_dd_Materials_PVC0x981a5e8 
      230  _dd_Geometry_PoolDetails_lvOutInWaterPipeNearTub0xb91c4b0                                             0    #_dd_Materials_PVC0x981a5e8 
      231  _dd_Geometry_PoolDetails_lvOutOutWaterPipeNearTub0xb91c558                                            0    #_dd_Materials_PVC0x981a5e8 

"""
import os, sys, logging
log = logging.getLogger(__name__)
import xml.etree.cElementTree as ET
#import xml.etree.ElementTree as ET



COLLADA_NS='http://www.collada.org/2005/11/COLLADASchema'
tag = lambda _:str(ET.QName(COLLADA_NS,_))

parse_ = lambda _:ET.parse(os.path.expandvars(_)).getroot()
tostring_ = lambda _:ET.tostring(_)


def qname(name):
    if '/' in name:
        qname = '/'.join(map(tag,name.split('/'))) 
    else:
        qname = tag(name) 
    return qname 

def findone(elem, name, att=None):
    all = elem.findall(qname(name))
    assert len(all) == 1, ( all, elem, name)
    if att:
        return all[0].attrib[att]
    return all[0]

def findall(elem, name, att=None, fn=None):
    all = elem.findall(qname(name))
    if att:
        return map(lambda _:_.attrib[att], all)
    if fn:
        return map(fn, all)
    return all    

def find(elem, name):
   return elem.find(qname(name))


class TopNode(dict):
    fmt = "  %(index)-4s %(topnode_id)-100s  %(nsub)-4s %(material_target)s "  
    def __str__(self):
        return self.fmt % self



class XMLDAE(object):
    def __init__(self, path):
        dae = parse_(path)

        self.effect = {}
        self.material = {}
        self.geometry = {}
        self.topnode = {}    # top level immediately under library_nodes
        self.subnode = {}
        self.scene = {}

        self.examine(dae)

    def examine_geometries(self, geometries):
        count = 0 
        for geometry in findall(geometries, 'geometry'):
            count += 1
            id = geometry.attrib['id']
            self.geometry[id] = geometry
        pass    
        assert len(self.geometry) == count , "geometry count mismatch"    
        log.debug("examine_geometries found %s " % len(self.geometry))    

    def examine_effects(self, effects):
        self.effect = findall(effects,'effect', att="id")  # list of ids 
        log.debug("examine_effects found %s " % len(self.effect))    

    def examine_materials(self, materials):
        self.material = findall(materials,'material', fn=lambda _:findone(_,"instance_effect", att="url"))  # list of instance_effect url 
        log.debug("examine_materials found %s" % len(self.material))    

    def examine_scenes(self, scenes):
        for s in findall(scenes,"visual_scene"):
            id = s.attrib['id']
            self.scene[id] = findone( s, "node/instance_node", att="url")
        log.debug("scenes %s " % self.scene)

    def examine_scene(self, scene):
        self.scene_url = findone(scene,"instance_visual_scene", att="url")

    def check_instance_geometry(self, instance_geometry):
        instance_material = findone(instance_geometry, "bind_material/technique_common/instance_material")
        ima = instance_material.attrib
        assert 'symbol' in ima and 'target' in ima
        return ima

    def check_topnode(self, topnode, index):
        id = topnode.attrib['id']
        #print tostring_(topnode)
        instance_geometry = findone(topnode, 'instance_geometry')
        ima = self.check_instance_geometry(instance_geometry)
        subnodes = findall( topnode, "node")
        return TopNode(index=index, subnodes=subnodes, topnode_id=id, nsub=len(subnodes), geometry_url=instance_geometry.attrib['url'], material_symbol=ima['symbol'], material_target=ima['target'])

    def examine_nodes(self, nodes):
        count = 0 
        for node in findall(nodes, 'node'):
            count += 1 
            id = node.attrib['id']
            index = len(self.topnode)
            ckd = self.check_topnode(node, index)
            #print ckd 
            self.topnode[id] = ckd
        assert len(self.topnode) == count , "top level node count mismatch"    
        log.debug("examine_nodes found %s" % len(self.topnode))    

    def examine(self, dae):
        effects = find(dae,"library_effects")
        materials = find(dae,"library_materials")
        geometries = find(dae,"library_geometries")
        nodes = find(dae,"library_nodes")
        scenes = find(dae,"library_visual_scenes")
        scene = find(dae,"scene")
        pass
        self.examine_effects(effects)
        self.examine_materials(materials)
        self.examine_geometries(geometries)
        self.examine_nodes(nodes)
        self.examine_scenes(scenes)
        self.examine_scene(scene)


def main():
    logging.basicConfig(level=logging.INFO)
    args = sys.argv[1:]
    path = "$LOCAL_BASE/env/geant4/geometry/xdae/g4_01.dae"
    xmldae = XMLDAE(path)

    if len(args) == 0:
        for k,v in sorted(xmldae.topnode.items(), key=lambda kv:xmldae.topnode[kv[0]]['index']):
            print v
    else:
        for arg in args:
            try:
                int(arg)
                kvs = filter(lambda kv:kv[1]['index']==int(arg),xmldae.topnode.items())
                assert len(kvs) == 1
                id, tn = kvs[0]
            except ValueError:
                id = str(arg)
                tn = xmldae.topnode[id]
            print tn
            for sn in tn['subnodes']:
                print tostring_(sn)



if __name__ == '__main__':
    main()



