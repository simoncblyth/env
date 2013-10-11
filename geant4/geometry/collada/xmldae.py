#!/usr/bin/env python
"""
XMLDAE
=======

Grab latest .dae from N with *dae-;dae-get*

XML level view of a collada .dae, for debugging 

Objective is not to recreate pycollada, but merely to 
be a convenient debugging tool to ask XML based questions 
of the DAE.

Questions
----------

#. Would the xml documument ids be unique without the pointers 0x.... ?

   * checking the .gdml there are dupes in the subtraction/union solids and between element/material names
     see ~/e/tools/checkxml.py 
   * checking the GDML writing on which the DAE writing is based,  there is currently only a global addPointerToName 
     switch so cannot easily turn it off for volumes and not for solids as would break references to solids

#. Can I reproduce VRML2 output from the DAE ? As a validation of all those transformations and everything else.

   * PV count now matches
   * PV name matching, the NCName IDref XML restriction forced replacing 3 chars ":/#" with  "_"
   
     * that is difficult to reverse, need some more unused acceptable chars (single chars would be best)
     * iterating on dae-edit;dae-validate find that "." and "-" are acceptable on other than the first char 

     * http://www.schemacentral.com/sc/xsd/t-xsd_NCName.html
     * http://stackoverflow.com/questions/1631396/what-is-an-xsncname-type-and-when-should-it-be-used
     * http://docs.marklogic.com/xdmp:encode-for-NCName
     * :google:`NCName encoding decoding`
     * https://nees.org/tools/vees/browser/xerces/src/xercesc/util/XMLString.cpp
     * http://msdn.microsoft.com/en-us/library/system.xml.xmlconvert.aspx 

  * TODO:

    * add checkxml.py collection of all id characters to see if "." is used 



Reversible Char Swaps
~~~~~~~~~~~~~~~~~~~~~~~

::

    /  ->   _
    :  ->   -      (colon always precedes digits eg :1 )  
    #  ->   .


The only '-' containg names that beings with '/'::

    /dd/Structure/Sites/db-rock0xc633af8
    /dd/Structure/Sites/db-rock0xc633af8_pos
    /dd/Structure/Sites/db-rock0xc633af8_rot





Usage
------

`xmldae.py -w -y PV`   
     recursive walk dumping PV 

`xmldae.py -w -y PV -r`   
     recursive walk dumping PV, dump parent node (the LV) also 

`xmldae.py -w -y PV -z 6`
     truncate recursion depth to level 6, for speed


"""
import os, sys, logging
log = logging.getlogger(__name__)

#import xml.etree.celementtree as et
#import xml.etree.elementtree as et
import lxml.etree as et


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

                
class Node(list):
    xmlcache = {}

    @classmethod
    def find_uid(cls, xid):
        """
        :param xid: XML document id attribute value 

        Find a unique id for the emerging tree Node, distinct from the source xml node id
        Note that the LV and PV registries are separate
        """
        count = 0 
        uid = None
        while uid is None or uid in cls.registry: 
            uid = "%s.%s" % (xid,count)
            count += 1
        return uid

    @classmethod
    def make(cls, dae, xmlnode, depth, parent=None ):
        """
        NB distinction between xml id `xid` and refs `xref` which correspond 
        to XML document ids and refs
        and the unique `uid` corresponding to the tree that the recursion creates 

        NB put placeholder in the registry prior to node instanction  
        as ctor recursion means that all descendants of a node will be created 
        before that nodes ctor completes
        """
        if depth > dae.opts.depthmax:
            log.debug("recursion truncated at depth %s depthmax %s  " % (depth, dae.opts.depthmax) )
            return None

        assert xmlnode is not None
        xid = Node.id_(xmlnode)
        uid = cls.find_uid(xid)

        if parent is None:
            log.warn("not registering uid %s " % uid ) 
            index = 0                   # only the World is not registered, that lives at index 0
        else:    
            cls.registry[uid] = True    # place holder
            index = len(cls.registry)   # start from 1

        node = cls(dae, xmlnode, uid, index, depth, parent )

        if parent is not None:
            cls.registry[uid] = node 
            assert node.id == uid 

        cls.created += 1
        if cls.created % 1000 == 0:
            print node
        return node

    @classmethod
    def resolve_xmlnode(cls, xml, xref):
        """
        First look in the cache for pre-existing xmlnode elements, 
        if not there try a document findall, if still not found
        take a spin over the cache again.

        Bizarrely it seems that reliably access the element at this
        third attempt.
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

    def __init__(self, dae, xmlnode, uid, index, depth, parent ):
        list.__init__(self)
        self.meta = {}
        self.dae = dae
        self.opts = dae.opts
        self.xmlnode = xmlnode
        self.xid = Node.id_(xmlnode)
        self.id = uid         
        self.index = index
        self.depth = depth
        self.parent = parent
        self.meta = dict(xid=self.xid, depth=depth, id=self.id, index=index, target=None, geourl=None, matrix=None)

        # over immediate sub-elements, not recursively 
        for elem in self.xmlnode:
            if elem.tag == qname('instance_geometry'):
                self.collect_geometry(elem)
            elif elem.tag == qname('matrix'):
                self.meta['matrix']=elem.text.lstrip().rstrip().replace("\t","").replace("\n",", ")
            elif elem.tag == qname('instance_node'):
                url = elem.attrib['url'] 
                rxnode = Node.resolve_xmlnode(dae.xml, url)
                assert rxnode is not None, "failed to resolve instance_node url %s " % url 
                refnode = LV.make(dae, rxnode, depth + 1, parent=self)  # can also be recursive here too 
                if refnode is not None:
                    self.append(refnode)

        xmlsubnodes = findall( xmlnode, "node")
        for xmlsubnode in xmlsubnodes:
            subnode = PV.make(dae, xmlsubnode, depth + 1, parent=self)   # NB recursive tree creation here
            if subnode is not None:
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


class PV(Node):
    registry = {}
    created = 0
    fmt = "PV  %(index)-4s %(depth)-3s %(nsub)-2s %(id)-100s  mtx:%(matrix)s "  
    pass
class LV(Node):
    registry = {}
    created = 0
    fmt = "LV  %(index)-4s %(depth)-3s %(nsub)-2s %(id)-100s  %(target)-30s  %(geourl)-20s "  
    pass



class XMLDAE(object):
    def __init__(self, xml, opts):
        self.xml = xml
        self.opts = opts

        self.effect = {}
        self.material = {}
        self.geometry = {}
        self.scene = {}

        self.examine(xml)

    def __str__(self):
        lines = []
        lines.append("effect: %s " % len(self.effect))
        lines.append("material: %s " % len(self.material))
        lines.append("geometry: %s " % len(self.geometry))
        lines.append("scene: %s " % len(self.scene))
        lines.append("rooturl: %s " % self.rooturl)
        return "\n".join(lines)

    def examine(self, xml):
        effects = find(xml,"library_effects")
        materials = find(xml,"library_materials")
        geometries = find(xml,"library_geometries")
        scenes = find(xml,"library_visual_scenes")
        scene = find(xml,"scene")
        pass
        self.examine_effects(effects)
        self.examine_materials(materials)
        self.examine_geometries(geometries)
        self.examine_scenes(scenes)
        self.examine_scene(scene)

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
        url = findone(scene,"instance_visual_scene", att="url")
        assert url[0] == '#', url
        url = url[1:]
        self.rooturl = self.scene[url]
        log.debug("scene url: %s rooturl:%s " % (url, self.rooturl) )

    def create_tree(self):
        log.info("create_tree starting from root %s " % ( self.rooturl))
        xnode = Node.resolve_xmlnode(self.xml, self.rooturl) 
        depth = 0
        root = LV.make( self, xnode, depth, parent=None)
        self.root = root
        log.info("create_tree completed from root")
        self.summary()

    def walk(self):    
        log.info("walk starting " )
        self.recurse(self.root, 0)
        log.info("walk done " )
    def recurse(self, node, rdepth):
        self.visit(node, rdepth)
        for subnode in node:
            self.recurse(subnode, rdepth+1)
    def visit(self, node, rdepth):
        assert rdepth == node.depth 
        if self.opts.voltype is not None and self.opts.voltype != node.__class__.__name__:
            return
        if node.depth < self.opts.depthmax:
            if node.index >= self.opts.indexmin and node.index <= self.opts.indexmax: 
                if self.opts.parent:
                    print "p ", node.parent
                print "  ", node



    def summary(self):
        print "PV registry %s created %s " % (len(PV.registry), PV.created)
        print "LV registry %s created %s " % (len(LV.registry), LV.created)
        print "xmlcache %s " % len(Node.xmlcache)

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
    parent = False 
    depthmax = 100 
    indexminmax = "0,100000"
    daepath = "$LOCAL_BASE/env/geant4/geometry/xdae/g4_01.dae"
    voltype = None

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
    op.add_option("-r", "--parent", action="store_true", default=defopts.parent )
    op.add_option("-z", "--depthmax", type=int, default=defopts.depthmax )
    op.add_option("-i", "--indexminmax", default=defopts.indexminmax, help="comma delimited min,max index integers" )
    op.add_option("-d", "--debug", action="store_true", default=defopts.debug )
    op.add_option("-x", "--xmldump", action="store_true", default=defopts.xmldump )
    op.add_option("-y", "--voltype", default=defopts.voltype, help="PV or LV or None for both" )

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


    minmax = map(int,opts.indexminmax.split(","))
    assert len(minmax) == 2
    opts.indexmin = minmax[0]
    opts.indexmax = minmax[1]

    assert os.path.exists(daepath), (daepath,"DAE file not at the new expected location, please create the directory and move the .dae  there, please")
    return opts, args



def main():
    opts, args = parse_args(__doc__) 
    log.info("reading %s " % opts.daepath )
    xml = parse_(opts.daepath)

    if opts.debug:
        checkid(xml)

    if opts.traverse or opts.walk:
        xmldae = XMLDAE(xml, opts)
        xmldae.create_tree()

    if opts.traverse:
        pass

    if opts.walk:          
        xmldae.walk()


if __name__ == '__main__':
    main()



