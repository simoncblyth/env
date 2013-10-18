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
from collada.xmlutil import etree as ElementTree

collada.scene.Node.__str__ = lambda self:'<Node %s %s transforms=%d, children=%d>' % (" " * 8, self.id, len(self.transforms), len(self.children))
collada.scene.NodeNode.__str__ = lambda self:'<NodeNode node=%s nodechildren=%d>' % (self.node.id,len(self.node.children))

def shorten_id( id ):
    d = [
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
   Used for random access into the collada tree and for all unique node 
   wrapping to allow tree pruning. The wrapping allows pruning portions
   of the tree without reference breaking concerns as there is no node 
   reuse for DAENode.
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
        self.index_tree(dae)

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

    def index_tree( self,  dae):
        """
        Recurse over the tree collecting nodes into the `DAENode` registry, a uid keyed dict 
        containing DAENode instances which represent all potential target nodes and their
        ancestors. 
        """
        log.info("index_tree assigning unique id to nodes")
        self.leafcount = 0 
        self.nodecount = 0 
        self.recurse()
        log.info("index_tree : leafcount %s nodecount %s DAENode.registry %s " % (self.leafcount, self.nodecount, len(DAENode.registry)))


    def recurse( self, node=None, ancestors=[] , parent_dnode=None):
        """
        Recurse collecting DAENode instances into DAENode.registry
        and wrap the original collada tree in DAENode 
        """
        top = False
        if node is None:
            node = self.topnode
            top = True

        dnode = DAENode(node, ancestors) 
        if top:
            self.topdnode = dnode

        if parent_dnode is not None: 
            parent_dnode.children.append(dnode)

        if not hasattr(node,'children') or len(node.children) == 0:# leaf
            self.leafcount += 1 
        else:
            self.nodecount += 1 
            for child in node.children:
                self.recurse(child, ancestors=ancestors + [node], parent_dnode=dnode)

    def define_target(self, uid ):
        target = DAENode.registry[uid]
        self.target = target
        self.target_node = target.node
        self.target_ancestors = target.ancestors
        log.info("define_target uid %s  " % (uid))
        log.info(target)  

    def targetted_recurse( self, node=None, depth=0, after_target=False, parent=None ):
        """
        Constrain recursive traverse to go directly to a target node
        and only after hitting it to recurse normally.

        Regarding subcopy : pruning the existing tree is tempting, 
        but node re-use probably make that difficult. Would need to 
        encase the tree in DAENode 
        """
        root = False       
        if node is None:
            node = self.topnode
            root = True

        fmt = "tgtrec %s [%s] %s " 
        if hasattr(node,'children') and len(node.children) > 0:# non-leaf
            if after_target:
                proceed, mkr = True, "--" 
            elif node in self.target_ancestors:
                proceed, mkr = True, ">>" 
            elif node == self.target_node:   # hmm node non-uniquess may mess this up, would need to traverse the DAENode
                proceed, mkr = True, "**" 
                after_target = True
            else:
                proceed, mkr = False,"##" 
            pass
            if proceed:    
                log.info(fmt  % ( mkr * depth, depth, node )) 
                cnode = self.copynode(node)  # needs to be before recurse below in order to have somewhere to hang the children
                if root:
                    self.ctop = cnode
                if parent is not None:
                    parent.children.append(cnode)
                for child in node.children:
                    self.targetted_recurse(child, depth=depth+1, after_target=after_target, parent=cnode )
                
        else:#leaf
            mkr = ".."
            log.info(fmt % ( mkr * depth, depth, node )) 
            cnode = self.copynode(node)
            if parent is not None:
                parent.children.append(cnode)
              
    def traverse( self, node, visit_leaf=lambda _:_ , visit_nonleaf=lambda _:_ ):
        """
        """ 
        if not hasattr(node,'children') or len(node.children) == 0:# leaf
            visit_leaf(node)
        else:
            visit_nonleaf(node)
            for child in node.children:
                self.traverse(child, visit_leaf, visit_nonleaf )

    def copynode(self, node):
        if node.__class__.__name__ == 'Node':
            cnode = collada.scene.Node( shorten_id(node.id) , children=[], transforms=node.transforms )   # hmm should be copying transforms too 
        elif node.__class__.__name__ == 'NodeNode':
            crefnode = self.copynode( node.node )  # these LV nodes referred to by NodeNode (aka instance_node) must be added to library_nodes for sure
            if crefnode not in self.copy.nodes:
                # reverse node order, for blender import benefit
                if len(self.copy.nodes) == 0: 
                    self.copy.nodes.append(crefnode)   
                else:    
                    self.copy.nodes.insert(0, crefnode)   
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


    def handle_geometry(self, node):
        self.copy_geometry( node.geometry )
        for matnode in node.materials:
            material = matnode.target
            self.copy_material(material)

    def subcopy(self, uid ):
        self.copy_asset()

        self.define_target( uid )
        self.targetted_recurse()
        top = self.ctop
        cscene = collada.scene.Scene("DefaultScene", [top])
        self.copy.scenes.append(cscene)
        self.copy.scene = cscene

        self.copy.save() 


    def subcopy_old(self, uid ):
        """
        :param uid: of root node to be copied

        From recursive traverses starting from the identified root node
        extract the parts of the original model that need to be copied
        to construct a sub-collada document.

        """

        self.copy_asset( self.orig.assetInfo )
        self.traverse( subroot, self.handle_geometry ) 

        self.refnodes = []
        self.subnodes = []
        self.traverse( subroot, lambda _:_ , self.collect_nodes ) 
        log.info("collected refnodes: %s subnodes: %s  " % (len(self.refnodes),len(self.subnodes)))

        for node in reversed(self.refnodes):
            self.copy_node(node)


        refroot = collada.scene.NodeNode( subroot )
        top = collada.scene.Node("top", [refroot])
        self.rcopy( subroot, top, present=True )

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
    dc.subcopy(uid)

    scpath = "subcopy.dae"
    dc.copy.write(scpath)
    checksub(scpath)
   






