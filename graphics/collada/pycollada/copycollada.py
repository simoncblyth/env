#!/usr/bin/env python
"""

When targetting a particular PV like __dd__Geometry__AD__lvOAV--pvLSO0xa8d68e0 probably 
need to back up to the LV __dd__Geometry__AD__lvOAV0xa8d6838 as the point at which 
to copy the tree. Going up is well defined as always one parent, but going down is not as 
possibly multiple children of the LV.

::

     67612     <node id="__dd__Geometry__AD__lvOAV0xa8d6838">
     67613       <instance_geometry url="#oav0x88402a8">
     67614         <bind_material>
     67615           <technique_common>
     67616             <instance_material symbol="WHITE" target="#__dd__Materials__Acrylic0x8880fd8"/>
     67617           </technique_common>
     67618         </bind_material>
     67619       </instance_geometry>
     67620       <node id="__dd__Geometry__AD__lvOAV--pvLSO0xa8d68e0">
     67621         <matrix>
     67622                 1 0 0 0
     67623 0 1 0 0
     67624 0 0 1 31.5
     67625 0.0 0.0 0.0 1.0
     67626 </matrix>
     67627         <instance_node url="#__dd__Geometry__AD__lvLSO0xa8d48e8"/>
     67628       </node>


For real world coordinates of sub-geometry would need to back up all the way to the "top" !!
In order to apply all the transformation matrices.
Maybe thats the way to do it:

#. grab ancestors list of the target node
#. traverse from the top, but restricting recursion to nodes on the ancestors list until
   hit the target node at which point traverse all nodes


"""
import os, logging
from copy import copy
log = logging.getLogger(__name__)
import collada
from collada.util import IndexedList
from collada.xmlutil import etree as ElementTree

collada.scene.Node.__str__ = lambda self:'<Node %s %s transforms=%d, children=%d>' % (" " * 8, self.id, len(self.transforms), len(self.children))
collada.scene.NodeNode.__str__ = lambda self:'<NodeNode node=%s nodechildren=%d>' % (self.node.id,len(self.node.children))

def shorten_id( id ):
    d = [
           ("__dd__Geometry__Sites__lv", "GS"),
           ("__dd__Geometry__AdDetails__lv","ADD"),
           ("__dd__Geometry__AD__lv","AD"),
           ("__dd__Geometry__Pool__","P"),
           ("__dd__Structure__","S"),
       ]    
    for long,short in d:
        if id.startswith(long):
            return short + id[len(long):]
    return id   

def unique_id( id, container ):
    uid = None
    count = 0
    while uid is None or uid in container:
        uid = "%s.%s" % (id,count)
        count += 1
    return uid


class DAENode(object):
   """
   Used for random access into the collada tree 
   """
   registry = {}
   def __init__(self, node, ancestors ):
       self.node = node
       self.ancestors = ancestors
       self.children = []

       if hasattr(node,'id'):
           bid = node.id
       elif hasattr(node,'geometry'):
           bid = "geonode." + node.geometry.id
       else:
           assert 0 
       self.id = unique_id( bid, self.registry )
       self.registry[self.id] = self       

   def summary(self):
       return "\n".join( map(str, self.ancestors + [self.node] ))
   def __str__(self):
       return "DNode %s : %s " % ( self.id, self.node )


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
        self.topnode = dae.scene.nodes[0]
        self.copy = collada.Collada()
        self.index_tree()

    def fullcopy(self):
        self.copy_asset()
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

    def copy_asset( self ):        
        assetInfo = self.orig.assetInfo
        cass = collada.asset.Asset.load( self.copy, {}, assetInfo.xmlnode)
        self.copy.assetInfo = cass 
  
    def copy_matnode( self, matnode ): 
        cmat = self.copy_material( matnode.target )
        return copy(matnode) 

    def copy_material( self, mat ):    
        ceff = collada.material.Effect.load( self.copy, {},  mat.effect.xmlnode ) 
        self.copy.effects.append(ceff)   # must append the fx before can load the material that refers to it 
        cmat = collada.material.Material.load( self.copy, {} , mat.xmlnode )
        self.copy.materials.append(cmat)
        return cmat
 
    def copy_geometry( self, geo ):
        cgeo = collada.geometry.Geometry.load( self.copy, {}, geo.xmlnode)
        self.copy.geometries.append(cgeo)
 
    def copy_node( self, node):
        #cnode = collada.scene.Node.load( self.copy, node.xmlnode , {} )
        cnode = collada.scene.Node( node.id, children=node.children, transforms=node.transforms )
        log.info("copy_node add lib %s " % cnode )
        self.copy.nodes.append(cnode)

    def copy_nodenode( self, nn ):
        cnn = collada.scene.NodeNode( node=nn.node )
 
    def copy_scene( self, scene ):
        cscene = collada.scene.Scene.load( self.copy, scene.xmlnode )
        self.copy.scenes.append(cscene)



    def index_tree( self ):
        """
        Recurse over the tree collecting nodes into the `DAENode` registry, a uid keyed dict 
        containing DAENode instances which represent all potential target nodes and their
        ancestors. 
        """
        log.info("index_tree assigning unique id to nodes")
        self.leafcount = 0 
        self.nodecount = 0 
        self.index_recurse()
        log.info("index_tree : leafcount %s nodecount %s DAENode.registry %s refnodes %s  " % (self.leafcount, self.nodecount, len(DAENode.registry), len(self.refnodes) ))

    def index_recurse( self, node=None, ancestors=[] ):
        """
        Recurse collecting DAENode instances into DAENode.registry
        """
        if node is None:
            node = self.topnode
            self.topdnode = None
            self.refnodes = IndexedList([],('id',))

        if node.__class__.__name__ == 'NodeNode':
            if not node.node.id in self.refnodes:
                self.refnodes.append(node.node)
 
        dnode = DAENode(node, ancestors) 
        if self.topdnode is None:
            self.topdnode = dnode

        if not hasattr(node,'children') or len(node.children) == 0:# leaf
            self.leafcount += 1 
        else:
            self.nodecount += 1 
            for child in node.children:
                self.index_recurse(child, ancestors=ancestors + [node] )   


    def define_target(self, uid , copy_ancestors ):
        target = DAENode.registry[uid]
        self.ctop = None
        self.copy_ancestors = copy_ancestors
        self.target = target
        self.target_node = target.node
        self.target_ancestors = target.ancestors
        log.info("define_target uid %s  " % (uid))
        log.info(target)  


    def traverse_volume_tree( self, lvnode , depth=0 ):
        pass 

    def targetted_recurse( self, node=None, depth=0, after_target=False, parent=None, lastchild=False ):
        """
        Constrain recursive traverse to go directly to a target node
        and only after hitting it to recurse normally.

        Comparing with the initial DAE creation in `G4DAEWriteStructure::PhysvolWrite` 
        which creates and appends to parent the below.  
        This operates with id/url only, avoiding the problem of `NodeNode` 
        needing a preexisting `Node` instance (maybe could introduce a `NodeRef` 
        placeholder to allow doing the same ?)::

              <node id="pvname">
                  <matrix ... />
                  <instance_node url="#lvname" />
              </node> 



        """
        if node is None:
            node = self.topnode

        # classify where we are in the tree by comparison of this node with targetted nodes
        # hmm node non-uniquess may mess this up, would need to traverse the DAENode
        if after_target:
            proceed, copy, mkr = True, True, "--" 
        elif node in self.target_ancestors:
            proceed, copy, mkr = True, self.copy_ancestors, ">>" 
        elif node == self.target_node:                        
            proceed, copy, mkr = True, True, "**" 
            after_target = True
        else:
            proceed, copy, mkr = False,False,"##" 
        pass
        cp = "*" if copy else " "

        fmt = "tgtrec %s %s [%s] %s " 
        if proceed:    
            log.info(fmt  % ( cp, mkr * depth, depth, node )) 
            if copy:
                # copynode needs to be before recurse below in order to have somewhere to hang the children, 
                # unless returned children from the recurse
                cnode = self.copynode(node)  
                if self.ctop is None:
                    self.ctop = cnode
                if parent is not None:
                    parent.children.append(cnode)
                    if lastchild:
                        if parent in self.refnodes:
                            log.info("parent in refnodes %s " % parent )
                            self.addlibnode(parent) 
                        else:
                            log.info("parent NOT in refnodes %s " % parent )
            else:
                cnode = None
            pass    
            if hasattr(node,'children') and len(node.children) > 0:# non-leaf
                for child in node.children:
                    lastchild = child == node.children[-1]
                    self.targetted_recurse(child, depth=depth+1, after_target=after_target, parent=cnode, lastchild=lastchild )


    def addlibnode(self, node):
        """
        reverse node order, for blender import benefit
        """
        if node not in self.copy.nodes:
            log.info("addlibnode %s " % node )
            if len(self.copy.nodes) == 0: 
                self.copy.nodes.append(node)   
            else:    
                self.copy.nodes.insert(0, node)
        else:        
            log.info("addlibnode skip %s " % node )

    def copynode(self, node):
        """
        """
        if node.__class__.__name__ == 'Node':
            #cid = shorten_id(node.id)
            cid = node.id
            cnode = collada.scene.Node( cid , children=[], transforms=node.transforms )   # hmm should be copying transforms too 
        elif node.__class__.__name__ == 'NodeNode':
            #
            # CAUTION NodeNode `id/children/matrix` are properties that pass thru to the referred `node`  
            #
            # the LV nodes referred to by NodeNode (aka instance_node) must be added to library_nodes for sure
            # but the nodes will usually  be incomplete (lacking children) at the time of the traverse visits instance_node
            # so just keep a note of the nodes and add them to library_nodes after the traverse
            #
            #  i need to here refer to a node that cannot yet exist ?
            # 
            crefnode = self.copynode( node.node )  
            cnode = collada.scene.NodeNode( node=crefnode )
        elif node.__class__.__name__ == 'GeometryNode':
            cmatnodes = []
            for matnode in node.materials:
                cmat = self.copy_matnode(matnode)
                cmatnodes.append(cmat)  
            pass
            cnode = collada.scene.GeometryNode( node.geometry, materials=cmatnodes )
            self.copy_geometry( cnode.geometry )
        else:
            assert 0, "unxpected node %s " % node 
        return cnode 

    def subcopy(self, uid, copy_ancestors=True ):

        log.info("refnodes_traverse finds %s %s " % ( len(self.refnodes), len(set(self.refnodes))))
        for _ in self.refnodes:
            log.info(_)

        self.copy_asset()

        self.define_target( uid , copy_ancestors=copy_ancestors )
        self.targetted_recurse()
        top = self.ctop
        cscene = collada.scene.Scene("DefaultScene", [top])
        self.copy.scenes.append(cscene)
        self.copy.scene = cscene

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
    uid = "__dd__Geometry__AD__lvOAV--pvLSO0xa8d68e0.0"    # maybe not so sensical doing this with an LV, PV uniqified makes more sense
    dc.subcopy(uid, copy_ancestors=True )

    scpath = "subcopy.dae"
    dc.copy.write(scpath)
    checksub(scpath)
   






