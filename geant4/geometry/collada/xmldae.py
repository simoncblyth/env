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

#import xml.etree.cElementTree as ET
#import xml.etree.ElementTree as ET
import lxml.etree as ET


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
    registry = {}
    xmlcache = {}

    @classmethod
    def make(cls, dae, xmlnode):
        """
        NB distinction between xml id `xid` and refs `xref` which correspond 
        to XML document ids and refs
        and the unique `uid` corresponding to the tree that the recursion creates 

        """
        assert xmlnode is not None
        xid = Node.id_(xmlnode)

        count = 0 
        uid_ = lambda _,c:"%s.%s" % (_, c)  
        uid = None
        while uid is None or uid in cls.registry: 
            uid = uid_(xid,count)
            count += 1

        #log.info("uid %s " % uid )    
        node = cls(dae, xmlnode, uid, len(cls.registry))
        key = node.id
        assert key not in cls.registry 
        cls.registry[key] = node
        return node

    @classmethod
    def resolve_xmlnode(cls, xml, xref):
        """
        First look in the registry for pre-existing Node objects, if not there 
        check the xmlcache and pull the Node object into existance.
        """
        if xref[0] == '#':xref = xref[1:]
       
        xmlnode = cls.xmlnode_from_cache( xml, xref )
        if xmlnode is None:
            xmlnode = cls.xmlnode_from_findall( xml, xref )
        if xmlnode is None:
            for xid, xnode in cls.xmlcache.items():
                if xid == xref:
                    xmlnode = xnode
                    break
        # try again from xmlcache  : why is it necessary to try again ?
        assert xmlnode is not None
        if xmlnode is None:
            log.info("Still FAILED to resolve %s cache lenth %s " % (xref,len(cls.xmlcache)) )
            cref = len(xref) - 10 
            for xid, xnode in cls.xmlcache.items():
                if xid == xref:
                    print "[%s]MATCH : HOW DID IT MANAGE TO FAIL" % xid
                elif xid[0:cref] == xref[0:cref]:
                    print "[%s]NEAR" % xid
                else:
                    pass
        else:
            log.debug("resolved %s to %s " % ( xref, xmlnode) )
        return xmlnode

    @classmethod
    def id_(cls, xmlnode):
        return xmlnode.attrib['id']


    @classmethod
    def xmlnode_from_findall(cls, xml, xref ):
        if xref[0] == '#':
            xref = xref[1:]
        for node in xml.findall('.//{%s}node' % COLLADA_NS ):
            xid = node.attrib['id']
            if xid == xref:
                return node
        return None      

    @classmethod
    def build_xmlcache(cls, xml):
        uid = set()
        for xmlnode in xml.findall('.//{%s}node' % COLLADA_NS ):
            xid = xmlnode.attrib['id']
            uid.add(xid)
            cls.xmlcache[xid] = xmlnode
        assert len(cls.xmlcache) == len(uid), ("missing or duplicated node id ", len(cls.xmlcache), len(uid))
        log.info("collect_xmlcache found %s nodes " % len(cls.xmlcache))

    @classmethod
    def xmlnode_from_cache(cls, xml, xref ):
        if xref[0] == '#':
            xref = xref[1:]
        if len(cls.xmlcache) == 0:
            cls.build_xmlcache(xml)
        return cls.xmlcache.get(xref, None)  # somthing dodgy about getting xml elems of of cache




    def __init__(self, dae, xmlnode, uid, index):
        list.__init__(self)
        self.meta = {}
        self.dae = dae
        self.opts = dae.opts
        self.xmlnode = xmlnode
        self.xid = Node.id_(xmlnode)
        self.id = uid         
        self.index = index
        self.meta = dict(xid=self.xid, id=self.id, index=index, target=None, ref=None, geourl=None, matrix=None)

        for elem in self.xmlnode:
            if elem.tag == qname('instance_geometry'):
                self.collect_geometry(elem)
            elif elem.tag == qname('matrix'):
                self.meta['matrix']=elem.text.lstrip().rstrip().replace("\t","").replace("\n",", ")
            elif elem.tag == qname('instance_node'):
                url = elem.attrib['url'] 
                self.meta['ref'] = url
                rxnode = Node.resolve_xmlnode(dae.xml, url)
                assert rxnode is not None, "failed to resolve instance_node url %s " % url 
                refnode = Node.make(dae, rxnode)
                self.append(refnode)

        xmlsubnodes = findall( xmlnode, "node")
        for xmlsubnode in xmlsubnodes:
            subnode = Node.make(dae, xmlsubnode)
            self.append(subnode)

        self.meta['nsub'] = len(self)    

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
    def __init__(self, xml, opts):
        self.xml = xml
        self.opts = opts

        self.effect = {}
        self.material = {}
        self.geometry = {}
        self.topnode = {}    # top level immediately under library_nodes
        self.subnode = {}
        self.node = {}
        self.scene = {}

        self.examine(xml)

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
        log.info("scenes %s " % self.scene)

    def examine_scene(self, scene):
        url = findone(scene,"instance_visual_scene", att="url")
        assert url[0] == '#', url
        url = url[1:]
        self.rooturl = self.scene[url]
        log.info("scene url: %s rooturl:%s " % (url, self.rooturl) )

    def examine_library_nodes(self, library_nodes):
        """
        Hmm, I do not like document order index. The recursive traverse index has more meaning.
        """
        pass

    def examine(self, xml):
        effects = find(xml,"library_effects")
        materials = find(xml,"library_materials")
        geometries = find(xml,"library_geometries")
        library_nodes = find(xml,"library_nodes")
        scenes = find(xml,"library_visual_scenes")
        scene = find(xml,"scene")
        pass
        self.examine_effects(effects)
        self.examine_materials(materials)
        self.examine_geometries(geometries)
        #self.examine_library_nodes(library_nodes)
        self.examine_scenes(scenes)
        self.examine_scene(scene)

    def walk(self):
        xnode = Node.resolve_xmlnode(self.xml, self.rooturl) 
        rootnode = Node.make( self, xnode)
        log.info("walk starting from rooturl %s rootid %s " % ( self.rooturl, rootnode.id))
        self.recurse(rootnode)
    def recurse(self, node):
        self.visit(node)
        for subnode in node:
            self.recurse(subnode)
    def visit(self, node):
        pass
        print node.index, node.id





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
    xml = parse_(opts.daepath)

    if opts.debug:
        allnode=xml.findall('.//{%s}node' % COLLADA_NS )
        uid = set()
        for i,node in enumerate(allnode):
            id = node.attrib['id']
            uid.add(id)
        assert len(allnode) == len(uid)
        print "allnode", len(allnode)
        #for id in list(uid):
        #    node = Node.xmlfind(xml, id)
        #    print id, node

        allrefnode=xml.findall('.//{%s}instance_node' % COLLADA_NS )
        print "refnode", len(allrefnode)
        for i,refnode in enumerate(allrefnode):
            ref = refnode.attrib['url'][1:]
            print i, ref
            assert ref in uid
            node = Node.xmlfind(xml, ref)
            print node

    if opts.traverse or opts.walk:
        xmldae = XMLDAE(xml, opts)
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
        xmldae.walk()
        print "registry %s " % len(Node.registry)
        print "xmlcache %s " % len(Node.xmlcache)



if __name__ == '__main__':
    main()



