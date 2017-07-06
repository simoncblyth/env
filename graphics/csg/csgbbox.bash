# === func-gen- : graphics/csg/csgbbox fgp graphics/csg/csgbbox.bash fgn csgbbox fgh graphics/csg
csgbbox-src(){      echo graphics/csg/csgbbox.bash ; }
csgbbox-source(){   echo ${BASH_SOURCE:-$(env-home)/$(csgbbox-src)} ; }
csgbbox-vi(){       vi $(csgbbox-source) ; }
csgbbox-env(){      elocal- ; }
csgbbox-usage(){ cat << EOU

CSG BBox 
===========

How to handle complements in CSG tree bbox combinations ?
-----------------------------------------------------------

Perhaps just implement an inverse flag within the 
bbox, so if the expression would be unbounded just return its
inverse and set the inverse flag ?

Intermediate unbounded are fine its only problematic
when the final bbox is unbounded ? 



Computing CSG tree boundaries as algebraic expressions
---------------------------------------------------------

Marco Mazzetti  
Luigi Ciminiera 

* http://dl.acm.org/citation.cfm?id=164360.164416
* ~/opticks_refs/csg_tree_boundaries_as_expressions_p155-mazzetti.pdf

Summary of the paper:

* bbox obtained from a CSG tree depends on evaluation order !!, 
  as the bbox operation is not associative, 
  you rearrange the boolean expression into UOI : union of intersections form

* upshot of this is that generally the bbox obtained will be overlarge

* handling CSG difference requires defining an InnerBB 
  corresponding to the maximum aabb that is completely inside the shape, 
  then::

      BB(A - B) = BB(A) - InnerBB(B)


Other relevant papers 
-------------------------------------


* https://pdfs.semanticscholar.org/97fe/7ced2dc6c4080c8a28907c75f6c90cb6cfe8.pdf
* ~/opticks_refs/rossignac_active_zones_csg.pdf

Goldfeather

* http://www.cc.gatech.edu/~turk/my_papers/pxpl_csg.pdf
* ~/opticks_refs/csg_tree_normalization_and_pruning_pxpl_csg.pdf 



searches
-----------

* :google:`bounding box tree canonical form`

* :google:`minimum size bbox of csg tree`

* :google:`minimum bounding box`


* https://www.cs.princeton.edu/~chazelle/pubs/BoxtreeHierarchRepSurf3D.pdf


minimum bounding box : by projection onto 3 axis planes
---------------------------------------------------------

* https://sffsymposium.engr.utexas.edu/Manuscripts/1999/1999-019-Chan.pdf


Trees Everywhere
--------------------

* https://www.clear.rice.edu/comp280/10spring/class/09/TreesT.pdf


CGAL
------

* http://doc.cgal.org/4.9/AABB_tree/index.html

The AABB tree data structure takes as input an iterator range of geometric
data, which is then converted into primitives. From these primitives a
hierarchy of axis-aligned bounding boxes (AABBs) is constructed and used to
speed up intersection and distance queries.

* this is somewhat different to my intersest : i want to find the minimum size bbox 
  for a csg tree



openscad
----------

Huh, no handling of complements.

openscad-/src/csgnode.cc::

    128 void CSGOperation::initBoundingBox()
    129 {
    130     const auto &leftbox = this->left()->getBoundingBox();
    131     const auto &rightbox = this->right()->getBoundingBox();
    132     Vector3d newmin, newmax;
    133     switch (this->type) {
    134     case OpenSCADOperator::UNION:
    135         newmin = leftbox.min().array().cwiseMin( rightbox.min().array() );
    136         newmax = leftbox.max().array().cwiseMax( rightbox.max().array() );
    137         this->bbox = BoundingBox( newmin, newmax );
    138         break;
    139     case OpenSCADOperator::INTERSECTION:
    140         newmin = leftbox.min().array().cwiseMax( rightbox.min().array() );
    141         newmax = leftbox.max().array().cwiseMin( rightbox.max().array() );
    142         this->bbox = BoundingBox( newmin, newmax );
    143         break;
    144     case OpenSCADOperator::DIFFERENCE:
    145         this->bbox = leftbox;
    146         break;
    147     default:
    148         assert(false);
    149     }
    150 }


gts
-----

* http://gts.sourceforge.net/reference/gts-bounding-boxes-trees.html

Bounding boxes trees -- implementation of axis-aligned bounding-box trees.

See: gts-;gts-bbtree


aabb tree
--------------

* https://spin.atomicobject.com/2012/01/30/2d-axis-aligned-bounding-box-trees/


povray
---------

povray-/source/core/shape/csg.cpp just skips the inverted in intersects::

     863     if(dynamic_cast<CSGIntersection *>(this) != NULL) // FIXME
     864     {
     865         /*
     866          * Calculate the bounding box of a CSG intersection
     867          * by intersecting the bounding boxes of all children.
     868          */
     869 
     870         NewMin = Vector3d(-BOUND_HUGE);
     871         NewMax = Vector3d(BOUND_HUGE);
     872 
     873         vector<Quadric *> Quadrics;
     874 
     875         /* Process all children. */
     876 
     877         for(vector<ObjectPtr>::iterator Current_Sib = children.begin(); Current_Sib != children.end(); Current_Sib++)
     878         {
     879             /* Inverted objects and height fields mustn't be considered */
     880 
     881             if(!Test_Flag((*Current_Sib), INVERTED_FLAG) && (dynamic_cast<HField *>(*Current_Sib) == NULL)) // FIXME
     882             {
     883                 /* We store quadrics since they'll be processed last, to benefit from confining them to a certain range */
     884                 if(dynamic_cast<Quadric *>(*Current_Sib) == NULL) // FIXME
     885                 {
     886                     if(dynamic_cast<Plane *>(*Current_Sib) != NULL) // FIXME
     887                         Quadric::Compute_Plane_Min_Max(dynamic_cast<Plane *>(*Current_Sib), TmpMin, TmpMax);
     888                     else
     889                         Make_min_max_from_BBox(TmpMin, TmpMax, (*Current_Sib)->BBox);
     890 
     891                     NewMin = max(NewMin, TmpMin);
     892                     NewMax = min(NewMax, TmpMax);
     893                 }
     894                 else
     895                     Quadrics.push_back(dynamic_cast<Quadric *>(*Current_Sib));
     896             }
     897         }
       





EOU
}
csgbbox-dir(){ echo $(local-base)/env/graphics/csg/graphics/csg-csgbbox ; }
csgbbox-cd(){  cd $(csgbbox-dir); }
csgbbox-mate(){ mate $(csgbbox-dir) ; }
csgbbox-get(){
   local dir=$(dirname $(csgbbox-dir)) &&  mkdir -p $dir && cd $dir

}
