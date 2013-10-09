#!/usr/bin/env python
"""
XMLDAE
=======

Grab latest .dae from N with *dae-;dae-get*

XML level view of a collada .dae, for debugging 

Objective is not to recreate pycollada, but merely to 
be a convenient debugging tool to ask XML based questions 
of the DAE.


Usage
------

List all topnodes with more that 0 children::

     xmldae.py -c 0 



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
isorted_ = lambda d,idx:sorted(d.items(),key=lambda kv:d[kv[0]].meta[idx]) 


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

                
                
def kvfindidx(nodes, index):               
   kvs = filter(lambda kv:kv[1].meta['index']==int(index),nodes.items())
   assert len(kvs) == 1, (len(kvs), "\n", kvs)
   return kvs[0][1]

def kvfindone(nodes, id):
   kvs = filter(lambda kv:kv[0].startswith(id),isorted_(nodes,'index'))
   assert len(kvs) == 1, kvs
   return kvs[0][1]

def kvfindall(nodes, id):
   kvs = filter(lambda kv:kv[0].startswith(id),isorted_(nodes,'index'))
   return map(lambda _:_[1], kvs)

def kvselect( nodes, args ):
    if len(args) == 0:args = range(0,len(nodes))
    tns = []
    for arg in args:
        try:
            int(arg)
            tn = kvfindidx(nodes, int(arg))
            tns.append(tn)
        except ValueError:
            all = kvfindall(nodes, str(arg))
            tns.extend(all)
    pass   
    log.debug("args %s yielded %s topnode " % (args,len(tns)))    
    return tns





class Node(list):
    fmt = "  %(index)-4s %(id)-100s  %(nsub)-4s tgt:%(target)s  ref:%(ref)s matrix:%(matrix)s "  

    id = property(lambda self:self.meta['id'])
    def __init__(self, xmlnode, index=None, opts=None):
        list.__init__(self)
        self.opts = opts
        self.meta = {}
        self.xmlnode = xmlnode
        id = xmlnode.attrib['id']
        subnodes = findall( xmlnode, "node")
        self.meta = dict(id=id, index=index, nsub=len(subnodes), target=None, ref=None, geourl=None, matrix=None)

        for elem in self.xmlnode:
            if elem.tag == qname('instance_geometry'):
                self.collect_geometry(elem)
            elif elem.tag == qname('matrix'):
                self.meta['matrix']=elem.text.lstrip().rstrip()
            elif elem.tag == qname('instance_node'):
                url = elem.attrib['url'] 
                assert url[0] == '#', url
                self.meta['ref'] = url[1:] 
            
        for subindex,xmlsubnode in enumerate(subnodes):
            log.debug("creating subnode from \n%s " % tostring_(xmlsubnode))
            subnode = Node(xmlsubnode, subindex, opts)
            log.debug("submode: %s " % subnode)
            self.append(subnode)
        assert len(self) == len(subnodes) 

    def collect_geometry(self, instance_geometry):
        geourl = instance_geometry.attrib['url']
        instance_material = findone(instance_geometry, "bind_material/technique_common/instance_material")
        symbol = instance_material.attrib['symbol']
        target = instance_material.attrib['target']
        assert target[0] == '#'
        target = target[1:]
        self.meta.update(geourl=geourl, symbol=symbol, target=target)     

    def __str__(self):
        lines = [self.fmt % self.meta]
        if self.opts.xmldump:
            lines.append(tostring_(self.xmlnode))
        return "\n".join(lines)    

class XMLDAE(object):
    def __init__(self, dae, opts):
        self.opts = opts

        self.effect = {}
        self.material = {}
        self.geometry = {}
        self.topnode = {}    # top level immediately under library_nodes
        self.subnode = {}
        self.node = {}
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

    def examine_library_nodes(self, library_nodes):
        count = 0 
        for index, xmlnode in enumerate(findall(library_nodes, 'node')):
            count += 1 
            node = Node(xmlnode, index, self.opts)
            self.topnode[node.id] = node
        assert len(self.topnode) == count , "top level node count mismatch"    
        log.debug("examine_nodes found %s" % len(self.topnode))    

    def examine(self, dae):
        effects = find(dae,"library_effects")
        materials = find(dae,"library_materials")
        geometries = find(dae,"library_geometries")
        library_nodes = find(dae,"library_nodes")
        scenes = find(dae,"library_visual_scenes")
        scene = find(dae,"scene")
        pass
        self.examine_effects(effects)
        self.examine_materials(materials)
        self.examine_geometries(geometries)
        self.examine_library_nodes(library_nodes)
        self.examine_scenes(scenes)
        self.examine_scene(scene)

    def walk(self, argrootid):
        ns = kvselect(self.topnode, argrootid)         
        assert len(ns) == 1, ns
        rootnode = ns[0]
        log.info("walk starting from argrootid %s rootid %s " % ( argrootid, rootnode['id']))
        print rootnode
        #print tostring_(rootnode['xmlnode'])
       
    def recurse(self, node):
        for subnode in node:
            self.recurse(subnode)
         





def parse_args(doc):
    from optparse import OptionParser
    defopts = Defaults()
    op = OptionParser(usage=doc)
    op.add_option("-o", "--logpath", default=defopts.logpath )
    op.add_option("-l", "--loglevel",   default=defopts.loglevel, help="logging level : INFO, WARN, DEBUG ... Default %default"  )
    op.add_option("-f", "--logformat", default=defopts.logformat )
    op.add_option("-c", "--childgt",  type=int, default=defopts.childgt)
    op.add_option("-s", "--subnode",  action="store_true" ,  default=defopts.subnode, help="dump subnodes of the targetted level")
    op.add_option("-w", "--walk",  action="store_true" ,  default=defopts.walk, help="recursive walk ")
    op.add_option("-t", "--traverse",  action="store_true" ,  default=defopts.traverse, help="non-recursive node traversal")
    op.add_option("-p", "--daepath", default=defopts.daepath )
    op.add_option("-d", "--debug", action="store_true", default=defopts.debug )
    op.add_option("-x", "--xmldump", action="store_true", default=defopts.xmldump )

    opts, args = op.parse_args()
    level = getattr( logging, opts.loglevel.upper() )

    if opts.logpath:  # logs to file as well as console, needs py2.4 + (?)
        logging.basicConfig(format=opts.logformat,level=level,filename=opts.logpath)
        console = logging.StreamHandler()
        console.setLevel(level)
        formatter = logging.Formatter(opts.logformat)
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)  # add the handler to the root logger
    else:
        try: 
            logging.basicConfig(format=opts.logformat,level=level)
        except TypeError:
            hdlr = logging.StreamHandler()              # py2.3 has unusable basicConfig that takes no arguments
            formatter = logging.Formatter(opts.logformat)
            hdlr.setFormatter(formatter)
            log.addHandler(hdlr)
            log.setLevel(level)
        pass
    pass
    log.info(" ".join(sys.argv))
    daepath = os.path.expandvars(os.path.expanduser(opts.daepath))
    if not daepath[0] == '/':
        opts.daepath = os.path.join(os.path.dirname(__file__),daepath)
    else:
        opts.daepath = daepath 

    base, ext = os.path.splitext(os.path.abspath(daepath))
    dbpath = base + ".dae.db"
    opts.dbpath = dbpath
    assert os.path.exists(daepath), (daepath,"DAE file not at the new expected location, please create the directory and move the .dae  there, please")
    pass    
    return opts, args

class Defaults(object):
    logformat = "%(asctime)s %(name)s %(levelname)-8s %(message)s"
    loglevel = "INFO"
    logpath = None
    childgt = -1 
    subnode = False 
    walk = False 
    traverse = False 
    debug = False 
    xmldump = False 
    daepath = "$LOCAL_BASE/env/geant4/geometry/xdae/g4_01.dae"

def main():
    opts, args = parse_args(__doc__) 
    log.info("reading %s " % opts.daepath )
    dae = parse_(opts.daepath)

    if opts.debug:
        xml_world = findall(dae, 'library_nodes/node')[-1]
        #print 'world\n',tostring_(xml_world)
        xml_node = findall(xml_world, 'node')[0]
        #print 'xml_node\n',tostring_(xml_node)
        #xml_instance_node = findall(xml_node, 'instance_node')[0]
        #print 'instance_node\n',tostring_(xml_instance_node)

        node = Node(xml_node, index=0, opts=opts)    # this is failing to see the instance_node
        print 'node\n', node.meta
        print tostring_(node.xmlnode)
      

    if opts.traverse or opts.walk:
        xmldae = XMLDAE(dae, opts)
    else:
        return

    if opts.traverse:
        nodes = kvselect( xmldae.topnode, args )
        for node in nodes:
            if len(node) > opts.childgt:
                print node
                if opts.subnode:
                    for subnode in node:
                        print subnode

    if opts.walk:          
         if len(args) > 0:
             rootid = args[0]
         else:
             rootid = 'World'
         xmldae.walk(rootid)


if __name__ == '__main__':
    main()



