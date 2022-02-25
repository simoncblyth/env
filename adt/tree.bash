# === func-gen- : adt/tree fgp adt/tree.bash fgn tree fgh adt src base/func.bash
tree-source(){   echo ${BASH_SOURCE} ; }
tree-edir(){ echo $(dirname $(tree-source)) ; }
tree-ecd(){  cd $(tree-edir); }
tree-dir(){  echo $LOCAL_BASE/env/adt/tree ; }
tree-cd(){   cd $(tree-dir); }
tree-vi(){   vi $(tree-source) ; }
tree-env(){  elocal- ; }
tree-usage(){ cat << EOU

Tree Data Structure
=======================

Thinking about putting trees into arrays again.


* https://hbfs.wordpress.com/2009/04/07/compact-tree-storage/


tree traversal preorder/inorder/postorder/levelorder recursive and iteractive algorithms, tick trick
-------------------------------------------------------------------------------------------------------

* http://faculty.cs.niu.edu/~mcmahon/CS241/Notes/Data_Structures/binary_tree_traversals.html
* ~/opticks_refs/Binary_Tree_Traversals.webarchive


env-;pmt-;pmt-ecd  Tree.convert
----------------------------------


::

    189     @classmethod
    190     def convert(cls, parts, explode=0.):
    191         """
    192         :param parts: array of parts
    193         :return: np.array buffer of parts
    194 
    195         #. collect Part instances from each of the nodes into list
    196         #. serialize parts into array, converting relationships into indices
    197         #. this cannot live at lower level as serialization demands to 
    198            allocate all at once and fill in the content, also conversion
    199            of relationships to indices demands an all at once conversion
    200 
    201         """
    202         data = np.zeros([len(parts),4,4],dtype=np.float32)
    203         for i,part in enumerate(parts):
    204             nodeindex = part.node.index
    205             index = i + 1   # 1-based index, where parent 0 means None
    206 
    207             #if part.parent is not None:
    208             #    parent = parts.index(part.parent) + 1   # lookup index of parent in parts list  
    209             #else:
    210             #    parent = 0 
    211             #pass

    ///             
    ///             commented conversion of part.parent pointers into index within the parts    

    212 
    213             data[i] = part.as_quads()
    214 
    215             if explode>0:
    216                 dx = i*explode
    217                 data[i][0,0] += dx
    218                 data[i][2,0] += dx
    219                 data[i][3,0] += dx
    220 
    221             data[i].view(np.int32)[1,1] = index
    222             data[i].view(np.int32)[1,2] = 0      # set to boundary index in C++ ggeo-/GPmt
    223             data[i].view(np.int32)[1,3] = part.flags    # used in intersect_ztubs
    224             # use the w slot of bb min, max for typecode and solid index
    225             data[i].view(np.int32)[2,3] = part.typecode
    226             data[i].view(np.int32)[3,3] = nodeindex
    227         pass
    228         buf = data.view(Buf)
    229         buf.boundaries = map(lambda _:_.boundary, parts)








EOU
}
tree-get(){
   local dir=$(dirname $(tree-dir)) &&  mkdir -p $dir && cd $dir

}
