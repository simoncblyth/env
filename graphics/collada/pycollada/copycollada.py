#!/usr/bin/env python
"""

"""
import os, logging
log = logging.getLogger(__name__)
import collada
from collada.xmlutil import etree as ElementTree

collada.scene.Node.__str__ = lambda self:'<Node %s transforms=%d, children=%d>' % (self.id, len(self.transforms), len(self.children))
collada.scene.NodeNode.__str__ = lambda self:'<NodeNode node=%s nodechildren=%d>' % (self.node.id,len(self.node.children))


def unique_id( id, container ):
    uid = None
    count = 0
    while uid is None or uid in container:
        uid = "%s.%s" % (id,count)
        count += 1
    return uid

class DAECopy(object):
    """

    Succeeds to create an identical copy, as determined with::

        ./diff.sh 

    Which does::

        xmllint --format orig.dae > origf.dae
        xmllint --format copy.dae > copyf.dae
        diff origf.dae copyf.dae > dif.txt

    Without the format there are 6 hunk differences related to newlines::

        diff orig.dae copy.dae

    """
    def __init__(self, dae ):
        self.orig = dae
        self.copy = collada.Collada()
        self.index_tree(dae)

    def fullcopy(self):
        self.copy_asset( self.orig.assetInfo )
        for mat in self.orig.materials:
            self.copy_material(mat)
        for geo in self.orig.geometries:
            self.copy_geometry(geo)
        for node in self.orig.nodes:
            self.copy_node(node)
        for scene in dae.scenes:
            self.copy_scene(scene)
        #print ElementTree.tostring(self.copy.xmlnode)
        self.save()    
        return self.copy

    def save(self):
        self.copy.scene = self.copy.scenes[0]
        self.copy.save()

    def copy_asset( self, assetInfo ):        
        cass = collada.asset.Asset.load( self.copy, {}, assetInfo.xmlnode)
        self.copy.assetInfo = cass 
  
    def copy_material( self, mat ):    
        ceff = collada.material.Effect.load( self.copy, {},  mat.effect.xmlnode ) 
        self.copy.effects.append(ceff)   # must append the fx before can load the material that refers to it 
        cmat = collada.material.Material.load( self.copy, {} , mat.xmlnode )
        self.copy.materials.append(cmat)
 
    def copy_geometry( self, geo ):
        cgeo = collada.geometry.Geometry.load( self.copy, {}, geo.xmlnode)
        self.copy.geometries.append(cgeo)
 
    def copy_node( self, node):
        cnode = collada.scene.Node.load( self.copy, node.xmlnode , {} )
        self.copy.nodes.append(cnode)
 
    def copy_scene( self, scene ):
        cscene = collada.scene.Scene.load( self.copy, scene.xmlnode )
        self.copy.scenes.append(cscene)

    def index_tree( self,  dae):
        """
        Recurse over the tree collecting nodes into the `self.nodes` uid keyed dict  
        """
        log.info("index tree")
        self.nodes = {}
        self.leafcount = 0 
        self.nodecount = 0 
        top = dae.scene.nodes[0]
        self.recurse(top)
        log.info("leafcount %s nodecount %s nodes %s " % (self.leafcount, self.nodecount, len(self.nodes)))

    def recurse( self, node):
        if not hasattr(node,'children') or len(node.children) == 0:# leaf
            self.leafcount += 1 
        else:
            self.nodecount += 1 
            uid = unique_id( node.id, self.nodes )
            self.nodes[uid] = node                  
            for child in node.children:
                self.recurse(child)

    def rdump_leaf(self, node):print "leaf", node
    def rdump_nonleaf(self, node):print "nonleaf", node
    def rdump(self, node):
        self.traverse( node, self.rdump_leaf, self.rdump_nonleaf )

    def traverse( self, node, visit_leaf=lambda _:_ , visit_nonleaf=lambda _:_):
        if not hasattr(node,'children') or len(node.children) == 0:# leaf
            visit_leaf(node)
        else:
            visit_nonleaf(node)
            for child in node.children:
                self.traverse(child, visit_leaf, visit_nonleaf )

    def handle_geometry(self, node):
        self.copy_geometry( node.geometry )
        for matnode in node.materials:
            material = matnode.target
            self.copy_material(material)

    def handle_node( self, node):
        print "-" * 100
        print node
        print ElementTree.tostring(node.xmlnode)

    def collect_nodes( self, node):
        if node.__class__.__name__ == 'NodeNode':
            self.refnodes.append(node.node)
        elif node.__class__.__name__ == 'Node':
            self.subnodes.append(node)
        else:
            pass 

    def dumpobjects(self, node ):
        for bg in node.objects('geometry'):
            print bg 

    def subcopy(self, uid ):
        """
        :param uid: of root node to be copied

        From recursive traverses starting from the identified root node
        extract the parts of the original model that need to be copied
        to construct a sub-collada document.

        """
        subroot = self.nodes[uid]
        log.info("subcopy uid %s from subroot %s " % (uid, subroot))  
        #self.dumpobjects(subroot)
        #self.rdump(root)
        log.info("rprint ")
        rprint(subroot)

        self.copy_asset( self.orig.assetInfo )
        self.traverse( subroot, self.handle_geometry ) 

        self.refnodes = []
        self.subnodes = []
        self.traverse( subroot, lambda _:_ , self.collect_nodes ) 
        log.info("collected refnodes: %s subnodes: %s  " % (len(self.refnodes),len(self.subnodes)))
        for node in reversed(self.refnodes):
            self.copy_node(node)
        for node in reversed(self.subnodes):
            self.copy_node(node)
        pass

        refroot = collada.scene.NodeNode( subroot )
        top = collada.scene.Node("top", [refroot])
        subscene = collada.scene.Scene("DefaultScene", [top])

        self.copy.scenes.append(subscene)
        self.copy.scene = subscene
        self.copy.save() 


def rprint(node, depth=0, index=0):
    print "    " * depth, "[%d.%d] %s " % (depth, index, node)
    if not hasattr(node,'children') or len(node.children) == 0:# leaf
        pass
    else:
        cut = 5
        shorten = len(node.children) > cut*2    
        for index, child in enumerate(node.children):
            if shorten:
                if index < cut or index > len(node.children) - cut:
                    pass
                elif index == cut:    
                    child = "..."
                else:
                    continue
            rprint(child, depth + 1, index)


def checksub( scpath ):
    log.info("checksub %s " % scpath )
    sub = collada.Collada(scpath)
    for _ in sub.scene.objects('geometry'):
        print _
    assert len(sub.scene.nodes) == 1
    subtop = sub.scene.nodes[0]
    rprint(subtop)     
 

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    path = os.path.expandvars("$LOCAL_BASE/env/geant4/geometry/xdae/g4_01.dae")
    log.info("reading %s " % path )
    dae = collada.Collada(path)
    dc = DAECopy(dae)

    check_fullcopy = False
    if check_fullcopy:
        dae.write("orig.dae")
        cdae = dc.fullcopy()
        cdae.write("copy.dae")
    pass


    #uid = "World0xaa8afb8.0"   TODO test fullcopy starting from World, its slow
    #uid = "__dd__Geometry__AD__lvLSO0xa8d48e8.0"
    uid = "__dd__Geometry__AD__lvOAV--pvLSO0xa8d68e0.0"

    dc.subcopy(uid)

    scpath = "subcopy.dae"
    dc.copy.write(scpath)
    checksub(scpath)
   






