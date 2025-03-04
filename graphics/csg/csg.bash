# === func-gen- : graphics/csg/csg fgp graphics/csg/csg.bash fgn csg fgh graphics/csg
csg-src(){      echo graphics/csg/csg.bash ; }
csg-source(){   echo ${BASH_SOURCE:-$(env-home)/$(csg-src)} ; }
csg-vi(){       vi $(csg-source) ; }
csg-env(){      elocal- ; }
csg-usage(){ cat << EOU

CSG : Constructive Solid Geometry
==================================









Searching for CSG developments
---------------------------------


:google:`CSG coincident faces ray tracing`


Active Zones in CSG for accelerating Boundary Evaluation, Redundancy Elimination, Interference Detection and Shading Algorithms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


* ~/opticks_refs/csg_active_zones_rossignac_10.1.1.449.428.pdf
* https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.449.428&rep=rep1&type=pdf


:google:`Parallel GPU Boolean Evaluation for CSG Ray-Tracing`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ~/opticks_refs/Domingues_IST_Parallel_GPU_Boolean_Evaluation_for_CSG_Ray_Tracing_article-1.pdf
* HMM: some intersesting ideas, but fundamentally using interval approach

Traversing and evaluating a CSG tree on the GPU can be a challenging task,
since recursion is not allowed and the available memory is limited. To
overcome this problem, we use a linearized tree representation and a stack to
hold temporary values of the tree during evaluation.  We first attempted to
represent the CSG tree in postfix notation, where each element of the array
contained either the id of a primitive, or the Boolean operator associated.
This method had the limitation of requiring all the elements of the tree to
be processed to evaluate the compound object. When evaluating a CSG tree, the
structure of the tree should be taken into consideration to skip subtrees
that will not influence the outcome of the Boolean evaluation. For example,
when intersecting two objects, if the left subtree has value false, then there
is no need to process the right subtree, because the result will be f alse,
following the Boolean algebra rules. Using a postfix tree representation does
not allows us to easily implement this sort of optimizations.

We solve the early out problem by converting the CSG tree to an array form
which is traversed in depth-first order. With this representation, we fit
each element of the tree in 32 bits length, having a structure that requires
less memory and that is able to skip unnecessary subtrees. To store each tree
node in 32 bits, the following convention was adopted: the 3 most significant
bits representing the operator, and the remaining 29 bits to represent either
the position of the right child, or for leaf nodes, the id of the primitive.
The operator zero indicates a leaf node. With this representation, a CSG tree
can have up to 2^29 nodes and can be used in scenes with 2^29 primitives. These
limits can be increased by representing each node with more bits and adopting a
similar convention.





Rendering Constructive Solid Geometry With Python
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://www.fotonixx.com/posts/efficient-csg/


QuickCSG n-ary CSG using kd-tree
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://hal.inria.fr/hal-01587902/document

QuickCSG: Fast Arbitrary Boolean Combinations of N Solids
by M Douze, 2017



CSG Ray Tracing Revisited
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ~/opticks_refs/csg_ray_tracing_revisited_61364.pdf
* https://www.scitepress.org/papers/2017/61364/61364.pdf

Organize constituents into layers, each with sums of positive and negative shapes. 


In order to avoid having to compute and store all intersection intervals along
a ray, we take advantage of the structure of our scenes. Similar to depth peel-
ing, from the camera perspective the CSG model is organized into a number of
layers Li (i = 1,...,l). Each layer is composed of a number of positive solids
Pi, j (j = 1,...,pi) and negative solids Ni,k (k = 1,...,ni). We will refer to
the negative solids as cutouts in the following. Thus, the CSG operations for a
scene S can be described as::

    S = Sum Li
        
    Li = Sum Pi,j - Sum Ni,k


Where + and - represent union and difference set operations.

Basically, a layer can be seen as the difference of two compound (positive and
neg- ative) objects. Because of this, CSG ray tracing a sin- gle layer can be
done by tracking when a ray runs within a positive or negative medium. To this
end, we employ two counters (posDepth and negDepth) that are attached as custom
parameters to each ray. When- ever the ray tracer finds the closest
intersection of a ray with the primitives of a layer, a hit program is called
which is illustrated in algorithm 1.


::

    if entering primitive then 
        delta := +1;
    else
        delta := −1; 
    endif
    if positive primitive hit then 
        ray.posDepth += delta;
    else
        ray.negDepth += delta; 
    endif
    if (ray.posDepth > 0) && (ray.negDepth <= 0) then
        ReportHit();
    else
        // final hit in layer found
        ContinueRay(ray);   // still inside a negative medium
    endif

Each time a ray enters or leaves a positive or negative primitive, the
counters are increased or decreased, respectively. In case we are inside a
positive medium, but not inside a negative medium, we found the cor- rect
hitpoint corresponding to a layer. Otherwise ray traversal continues using the
updated counters.  To find the final global hitpoint in the scene, a pri- mary
ray is sequentially tested against all layers, and from the set of layer
hitpoints the nearest one is ac- cepted as final position for shading. In the
scene graph used by the ray tracing engines, the layers of a scene can be
stored as independent sub-trees, which can be intersected separately.


::

       enum {ENTER=+1, EXIT=-1 } ; 

       int delta = dot(ray_direction, surface_normal) < 0.f  ? ENTER : EXIT ; 

       if node.complement == false:
            ray.pos_count += delta ;       
       else:
            ray.neg_count += delta ;   
       pass

       if(  ray.pos_count > 0 && ray.neg_count <= 0 ) reportHit
       else ContinueRay
        
::
   
     delta    +1     -1         +1    -1          +1    -1          +1    -1         +1    -1           +1    -1
                            
               +-----+           +-----+           +-----+           +-----+          +-----+            +-----+    
               |     |           |     |           |     |           |     |          |     |            |     |    
           - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
               |     |           |     |           |     |           |     |          |     |            |     |    
               +-----+           +-----+           +-----+           +-----+          +-----+            +-----+    
           
     pos_     [1]    0           1     0           1     0           1     0          1     0            1     0      
     neg_      0     0           0     0           0     0           0     0          0     0            0     0
               |
              reportHit




   ENTER +1 : dot(ray_direction, surface_normal) < 0    ray direction against the normal
   EXIT  -1 : dot(ray_direction, surface_normal) > 0    ray direction with the normal



MULTIUNION OF FOUR CONSTITUENTS 

                                             +-----------------+
                                             |                 |
                                             |                 |
                                             |                 |
                                   +---------|-----------------|---------+
                                   |         |                 |         |
                                   |         |                 |         |
                                   |         |                 |         |
                         +---------|---------|-----------------|---------|---------+
                         |         |         |                 |         |         |
                         |         |         |                 |         |         |
                         |         |         |                 |         |         |
               +---------|---------|---------|-----------------|---------|---------|---------+
               |         |         |         |                 |         |         |         |
               |         |         |         |                 |         |         |         |
         0 - -[1]- - - - 2 - - - - 3 - - - - 4 - - - - - - - - 5 - - - - 6 - - - - 8 - - - - 9      ray starting from outside
              +1        +2        +3        +4                +3        +2        +1        +0      ray depth into geom 
               |         |         |         |                 |         |         |         |
               |         |         |         |                 |         |         |         |
               |    THESE DEPTH TOTALS ARE NOT THE APPROACH TAKEN        |         |         |  
               |    AS ARE OPERATING ON ENTER/EXIT CLASSIFIED FIRST INTERSECTS     |         | 
               |    SO WHAT WE HAVE IS FOUR ENTERS (E) AND NO EXITS (X)  -> PICK THE CLOSEST ENTER  
               |         |         |         |                 |         |         |         |
              [E]        E         E         E                 |         |         |         |        4 ENTERS, 0 EXIT
               |         |         |         |                 |         |         |         |
               |         |         |         |                 |         |         |         |
               |         |         |         |                 |         |         |         |
               |         |         |         |                 |         |         |         |
               |         |         |         |                 |         |         |         |
               |   0 - - 1 - - - - 2 - - - - 3 - - - - - - - - 4 - - - - 5 - - - - 6 - - - -[7]     ray starts inside (but does not known that)
               |        +1        +2        +3                +2        +1        +0        -1 
               |         |         |         |                 |         |         |         |
               |         E         E         E                 |         |         |        [X]       3 ENTERS, 1 EXIT   
               |         |         |         |                 |         |         |         |
               |         |         |         |                 |         |         |         |
               |         |         |         |                 |         |         |         |
               |         |         |         |                 |         |         |         |
               |         |   0 - - 1 - - - - 2 - - - - - - - - 3 - - - - 4 - - - - 5 - - - -[6]
               |         |        +1        +2                +1        +0        -1        -2
               |         |         |         |                 |         |         |         |
               |         |         E         E                 |         |         X        [X]       2 ENTERS, 2 EXIT
               |         |         |         |                 |         |         |         |
               |         |         |         |                 |         |         |         |
               |         |         |         |                 |         |         |         |
               |         |         |         |                 |         |         |         |
               |         |         |         |                 |         |         |         |
               |         |         |   0 - - 1 - - - - - - - - 2 - - - - 3 - - - - 4 - - - -[5]     1 ENTER,  4 EXIT   (wrong each constituent counts once only as nearest intersect) 
               |         |         |        +1                +0        -1        -2        -3      HMM: fear need to check that the ENTER has correponding EXIT 
               |         |         |         |                 |         |         |         |      ACTUALLY the ENTER will shield the EXIT of that constituent (as basing on first intersects) 
               |         |         |         |                 |         |         |         |      SO WILL GET 1 ENTER, 3 EXIT  
               |         |         |         |                 |         |         |         |      THENCE HAVE TO LOOP THE ENTER TO GET ITS EXIT 
               |         |         |         |                 |         |         |         |      THEN HAVE EXITS FOR ALL AND CAN PICK THE FURTHEST
               |         |         |         |                 |         |         |         |
               |         |         |         E                 |         X         X        [X]     1 ENTER, 3 EXIT 
               |         |         |         |                 |         |         |         |
               |         |         |         |                 |         |         |         |
               |         |         |         |                 |         |         |         |
               |         |         |         |     0 - - - - - 1 - - - - 2 - - - - 3 - - - -[4]      All first intersects are EXIT
               |         |         |         |                -1        -2        -3        -4
               |         |         |         |                 |         |         |         |
               |         |         |         |                 X         X         X        [X]     0 ENTER, 4 EXIT 
               |         |         |         |                 |         |         |         |      PURELY EXITS ARE EASY : NO ENTER DEBT : JUST PICK THE FURTHEST 
               |         |         |         |                 |         |         |         |
               |         |         |         |                 |         |         |         |
               |         |         |         |                 |   0 - - 1 - - - - 2 - - - -[3]      Again all first intersects are EXIT
               |         |         |         |                 |        -1        -2        -3
               |         |         |         |                 |         |         |         |
               |         |         |         |                 |         X         X        [X]     0 ENTER, 3 EXIT
               |         |         |         |                 |         |         |         |
               |         |         |         |                 |         |         |         |
               |         |         |         |                 |         |         |         |
               |         |         |         |                 |         |   0 - - 1 - - - -[2]     Yet again all first intersects are EXIT 
               |         |         |         |                 |         |        -1        -2
               |         |         |         |                 |         |         |         |
               |         |         |         |                 |         |         X        [X]     0 ENTER, 2 EXIT 
               |         |         |         |                 |         |         |         |
               |         |         |         |                 |         |         |         |
               |         |         |         |                 |         |         |   0 - -[1]    Only first intersect is EXIT 
               |         |         |         |                 |         |         |        -1
               |         |         |         |                 |         |         |         |
               |         |         |         |                 |         |         |        [X]     0 ENTER, 1 EXIT 
               |         |         |         |                 |         |         |         |
               |         |         |         |                 |         |         |         |     ======================================================
               |         |         |         |                 |         |         |         |     ====== ADD A DISJOINT 5th CONSTITUENT ================  
               |         |         |         |                 |         |         |         |     ======================================================
               |         |         |         |                 |         |         |         |                                           +----------+
               |         |         |         |                 |         |         |         |                                           |          |
               |         |         |         |                 |         |         |    0 - [X]                                          E          |  
               |         |         |         |                 |         |         |         |       1 EXIT, 1 ENTER                     |          |  
               |         |         |         |                 |         |         |         |                                           |          |  
               |         |         |         |                 |         |         |         |                                           |          |
               |         |         |         |                 |         |         |         |                                           +----------+
               |         |         |         |                 |         |         |         |    WHEN THE FURTHEST EXIT IS CLOSER THAN    
               |         |         |         |                 |         |         |         |    THE NEAREST ENTER THE INTERSECT IS JUST THE FURTHEST EXIT 
               |         |         |         |                 |         |         |         |
               |         |         |         |                 |         |         |         |
               |         |         |         |                 |         |         |         |
               |         |         |         |                 |         |         |         |    IS IT CORRECT TO JUST PICK THE FURTHEST EXIT ?
               |         |         |         |                 |         |         |         |
               |         |         |         |                 |         |         |         |
               +---------|---------|---------|-----------------|---------|---------|---------+
                         |         |         |                 |         |         |
                         |         |         |                 |         |         |
                         |         |         |                 |         |         |
                         +---------|---------|-----------------|---------|---------+
                                   |         |                 |         |
                                   |         |                 |         |
                                   |         |                 |         |
                                   +---------|-----------------|---------+
                                             |                 |
                                             |                 |
                                             |                 |
                                             +-----------------+
  

When exiting the compound multiunion the depth is  0,-1,-2,...  (0 or less)

* Each ENTER increases the debt(depth) of the ray 
* Every EXIT pays back a unit of debt(depth) 
* Only when are back to zero or -ve debt is is possible to form a hit 

  * NB the debt(depth) is relative to the ray origin (it is not absolute) 


BUT ARE TRYING TO IMPLEMENT BASED ON ENTER/EXIT CLASSIFIED FIRST INTERSECTS TO FIT WITH RAY TRACING APPROACH 


THINKING ABOUT HOW TO IMPLEMENT MULTIUNION 


* for each ray, collect first intersects with all constituents, order them by distance, classify ENTER EXIT MISS

1. when start outside, the intersect is nearest ENTER   
2. when start inside, the intersect is the farthest EXIT 

  * what about disjoint unions ? 
  * still works as will not get first intersect EXITS as they would be shielded by corresponding ENTERs 




But better to not need to know inside/outside? 
Instead base entirely on single intersects onto each constituent (for ray tracing convenience)

1. all ENTER (no EXITs) [are outside all] -> pick nearest ENTER (no need for obtain any EXITs)
2. all EXIT (no ENTERs) [are at deepest insideness] -> pick farthest EXIT (no need to back-trace to get any ENTERs)
3. mixture of ENTER and EXIT [middling insideness depth] 

   * need to find the EXIT for every ENTER as they could be anywhere, then just pick furthest EXIT ? 
   * NOT QUITE:

     * only find EXITs for ENTERs that are closer than the furthest EXIT found so far (to correctly handle disjointed)
     * cannot optimize, eg only getting EXITs for the furthest ENTERs, because the EXITs can be anywhere unrelated to the ENTERs
     * every EXIT may push out the envelope, (which may mean more EXITs to find)
     * any ENTER within the envelope of furthest EXITs may push the envelope out further : so has to be checked   
     * ENTERs beyond the envelope of EXITs may be disjoint 

     * avoid storing all isects, just keep updating : nearest_enter_isect, farthest_exit_isect   
       yes but need the distances 

   * disjoint constituent will show an ENTER that is beyond the furthest EXIT of the others so it would be wrong to just give the furthest EXIT
 
   * ENTERs closer than the farthest EXIT found so far, places you in debt of finding their EXITs 
   * ENTERs beyond the farthest EXIT found so far, can be ignored for now, if the envelope subsequently 
     engulfs them then then their EXITs must be found otherwise they are dijoints


SUPPOSE THIS HAS TO BE MULTIPLE PASS : AS CANNOT KNOW THE FURTHEST EXIT UNTIL PRIOR PASS COMPLETE 



What storage needed ?

* float4 isect for each constituent, use same storage for ENTER and EXIT as dont need both at once
* sign of isect.w (t) to encode ENTER/EXIT and 0. OR t_min to signify MISS
* can holding all isect at once be avoided ? End result is just one isect.


1. loop over constituents updating : nearest_enter_isect, farthest_exit_isect, ENTER/EXIT/MISS typemask, enter_distances for each(?)
2. when typemask shows all ENTER/EXIT can return nearest_enter_isect/farthest_exit_isect
3. when typemask mixed need to again with t_min advanced 
 


WHAT ABOUT THE BINARY CASE : A MULTIUNION OF TWO CONSTITUENTS 


     UX1 : UNION FROM INSIDE ONE                           UX2 : UNION FROM OUTSIDE 0,1,2                            
                                                           UX3 : UNION FROM INSIDE BOTH 3,4,5                                            
                                                                                                       
                  +-----------------------+                           +-----------------------+        
                  |                     B |                           |                     B |        
       +------------------+               |                +------------------+               |        
       | A        |       |               |                | A        |       |               |        
       |          |       |               |                |          |       |               |        
       |          |       |               |               [E]         E       |               |        
       |   0- - - 1 - - - 2 - - - - - - -[3]           0 -[1]- - - -  2       |               |        
       |          E       X               |                |          |   3 - 4 - - - - - - -[5]       
       |          |       |              [X]               |          |       X              [X]        
       |          +-------|---------------+                |          +-------|---------------+        
       |                  |                                |                  |                        
       |                  |                                |                  |                        
       +------------------+                                +------------------+                        
                           

    CANNOT PICK FURTHEST EXIT (2:X) STRAIGHT AWAY 
    BECAUSE THE ENTER (1:E) PLACES YOU IN DEBT 
    OF HAVING TO RESOLVE IT TO GIVE ITS EXIT (3:X) 
    ONLY THEN CAN YOU PICK BETWEEN (2:X) AND (3:X) 
    TO GET FURTHEST EXIT

    IN THE MULTIUNION DO I NEED TO RESOLVE ALL ENTERS 
    OR PERHAPS JUST RESOLVE THE FURTHEST ENTER TO GET ITS EXIT ?
                                                                             

                                                                                                       
     0: origin                                           0: origin                                     
     1: B Enter                                          1: A Enter                 
     2: A Exit                                           2: B Enter                 
     1,2: B Closer        ==> LOOP_B                     1,2: A Closer        ==> RETURN_A             
     3: B Exit                  
     2,3: A closer        ==> RETURN_B                   3: origin
                                                         4: A Exit
                                                         5: B Exit
                                                         4,5 A Closer         ==> RETURN_B


    * loop the enter to find its other side 




* is it necessary to order constituent intersects or would just updating closest and furthest work ?





g4-cls G4MultiUnion 


G4MultiUnion::DistanceToInNoVoxels
    just minimum distance 

G4MultiUnion::DistanceToOutNoVoxels
::    

     249 G4double G4MultiUnion::DistanceToOutNoVoxels(const G4ThreeVector& aPoint,
     250                                              const G4ThreeVector& aDirection,
     251                                              G4ThreeVector* aNormal) const
     252 {
     253   // Computes distance from a point presumably outside the solid to the solid
     254   // surface. Ignores first surface if the point is actually inside.
     255   // Early return infinity in case the safety to any surface is found greater
     256   // than the proposed step aPstep.
     257   // The normal vector to the crossed surface is filled only in case the box
     258   // is crossed, otherwise aNormal->IsNull() is true.
     259 
     260   // algorithm:
     261   G4ThreeVector direction = aDirection.unit();
     262   G4ThreeVector localPoint, localDirection;
     263   G4int ignoredSolid = -1;
     264   G4double resultDistToOut = 0;


     265   G4ThreeVector currentPoint = aPoint;

     /// currentPoint starts at ray_origin  

 
     267   G4int numNodes = fSolids.size();
     268   for (G4int i = 0; i < numNodes; ++i)
     269   {
     270     if (i != ignoredSolid)
     271     {
                
     272       G4VSolid& solid = *fSolids[i];
     273       const G4Transform3D& transform = fTransformObjs[i];
     274       localPoint = GetLocalPoint(transform, currentPoint);
     275       localDirection = GetLocalVector(transform, direction);

     276       EInside location = solid.Inside(localPoint);

     277       if (location != EInside::kOutside)
     278       {


     /// loop over solids that currentPoint is inside (with prior solid ignored)

     279         G4double distance = solid.DistanceToOut(localPoint, localDirection,
     280                                                 aNormal);
     281         if (distance < kInfinity)   // thats a bit perplexing (unless the solid is unbounded)
     282         {
     283           if (resultDistToOut == kInfinity) resultDistToOut = 0;
     284           if (distance > 0)
     285           {
     286             currentPoint = GetGlobalPoint(transform, localPoint
     287                                           + distance*localDirection);

                     /// advance currentPoint "ray_origin" to exit of this solid 

     288             resultDistToOut += distance;
     289             ignoredSolid = i; // skip the solid which we have just left

     290             i = -1; // force the loop to continue from 0
                     
                     /// looping again with this solid ignored and the advanced origin 

     291           }

     292         }



     293       }     /// are inside
     294     }       /// not ignored
     295   }         /// over nodes



     296   return resultDistToOut;
     297 }




        +------------------------------------------------+
        |                                                |
        |                                                |
        |       +--------------------------------+       |
        |       |                                |       |
        |       |                                |       |
        |       |                                |       |
        |       |        +-------------+         |       |
        |       |        |             |         |       |
        |       |        |             |         |       |
        |       |        |     0 - - - 1 - - - - 2 - - -[3]
        |       |        |       d01   |  d12    |  d23  |       d03 = d01 + d12 + d23
        |       |        |             |         |       |
        |       |        |             |         |       |
        |       |        |             |         |       |
        |       |        |     0 - - - - - - - - 1 - - -[2]
        |       |        |            e01        |  e12  |       e02 = e01 + e12
        |       |        |             |         |       |
        |       |        |             |         |       |
        |       |        |             |         |       |
        |       |        |     0 - - - - - - - - - - - -[1]      f01   lucky escape in one loop
        |       |        |             |  f01    |       |
        |       |        |             |         |       |
        |       |        |             |         |       |
        |       |        +-------------+         |       |
        |       |                                |       |
        |       |                                |       |
        |       |                                |       |
        |       +--------------------------------+       |
        |                                                |
        |                                                |
        |                                                |
        +------------------------------------------------+

The order of the solid loop can lead to multiple steps or one. 



  


Nice Slides
~~~~~~~~~~~~~~~

* http://web.cse.ohio-state.edu/~parent.1/classes/681/Lectures/19.RayTracingCSG.pdf


Near Real-time CSG Rendering using Tree Normalization and Geometric Pruning, Goldfeather (1988)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://apps.dtic.mil/sti/pdfs/ADA201085.pdf
* ~/opticks_refs/CSG_Normalization_Goldfeather_1988_ADA201085.pdf

* paper presents pseudo-code for normalization algorithm 

::

    ((A-B) union C) intersect D 

    (AB'+C)D.


A CSG tree is in normal form if its boolean representation is in disjunctive normal form, 
i.e. if it is a sum of products of literals.

In addition to allowing rendering with a constant number of bits per pixel,
normalizing a CSG tree allows the rendering algorithm to be simpler than it
would be otherwise. Each product in the normalized expression can be rendered
using primitive/primitive interaction rather than subtree/subtree interaction.
In Section 3 we will see that normalization also allows unnecessary portions of
the CSG tree to be recognized and pruned very easily.


p10::

    BBox( A+B )  =  BBox(A) + BBox(B)        UNION
    BBox( A.B )  =  BBox(A).BBox(B)          INTERSECT
    BBox( A-B)   = BBox( A.B' ) = BBox(A)    DIFFERENCE 



* subtree pruning 



Boolean algebra : deMorgan
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://www.electronics-tutorials.ws/boolean/demorgan.html

::

   !(A.B)     = !A + !B
   !(A.B.C)   = !A + !B + !C
   !(A.B.C.D) = !A + !B + !C + !D

   !(A+B)     = !A.!B
   !(A+B+C)   = !A.!B.!C
   !(A+B+C+D) = !A.!B.!C.!D




Boolean algebra : sum-of-products form : AND then OR : UNION then INTERSECT
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://www.electronics-tutorials.ws/boolean/sum-of-product.html

::
 
    Q = !A.(!B.C + B.C + B.!C) + A.B.C          (product . AND/intersect)   (sum + OR/union)

    Q = !A.!B.C + !A.B.C + A.B.!C + A.B.C




Boolean algebra : product-of-sum form 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://www.electronics-tutorials.ws/boolean/product-of-sum.html

::

    A + 0 = A
    A + 1 = 1 
    A + A = A
    A + !A = 1

    A + B = B + A


Examples::

    Q0 = (A+B).(!B + C).(A + 1 )
    
    Q1 = (A+B+C).(A+C).(!B + !C)





 




Normalized CSG Form
~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://developer.openinventor.com/UserGuides/Oiv9/Inventor_Mentor/Rendering_Features/CSG_Rendering/CSG_Rendering.html

* https://www.usenix.org/legacy/event/usenix05/tech/freenix/full_papers/kirsch/kirsch.pdf



CSG State Table
~~~~~~~~~~~~~~~~~~~

* https://www.cl.cam.ac.uk/teaching/0809/AdvGraph/Advanced%20Graphics%2008x02%20-%20Geometric%20methods%20for%20ray%20tracing.ppt
* ~/opticks_refs/CSG_State_Table.png

CSG Regularization
~~~~~~~~~~~~~~~~~~~~


* ~/opticks_refs/Rossignac_CSG_regularization_SPM.pdf

QuickCSG: Fast Arbitrary Boolean Combinations of N Solids
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Matthijs Douze, Jean-Sebastien Franco, Bruno Raffin August 29, 2018

* https://arxiv.org/pdf/1706.01558.pdf
* ~/opticks_refs/QuickCSG_1706_01558.pdf


CSG Ray Tracing Revisited: Interactive Rendering of Massive Models Made of Non-planar Higher Order Primitives
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **interesting approach : layered CSG ray tracing** : HMM DOES IT HANDLE CSG_INTERSECT ?

Seyedmorteza Mostajabodaveh1,2, Andreas Dietrich1,2, Thomas Gierlinger1,2, Frank Michel1,2 and Andre Strok1,2
1Fraunhofer IGD, Darmstadt, Germany

* https://www.scitepress.org/Papers/2017/61364/61364.pdf
* http://publica.fraunhofer.de/documents/N-441713.html

* ~/opticks_refs/CSG_Ray_Tracing_Revisited_Darmstadt_61364.pdf


* https://paperexplained.cn/articles/paper/detail/7c36d0550f5051121a954e580d475af0fd7625ca/



* https://mmostajab.com
* https://mmostajab.com/ray-tracing-a-better-approach/
* https://github.com/mmostajab

* https://de.linkedin.com/in/andreas-dietrich-92349274

* https://github.com/search?l=C%2B%2B&q=constructive+solid+geometry&type=Repositories


In order to avoid having to compute and store all intersection intervals along
a ray, we take advantage of the structure of our scenes. Similar to depth peeling, 
from the camera perspective the CSG model is organized into a number of
layers Li (i = 1,...,l). 

Each layer is composed of a number of positive solids Pi, j (j = 1,...,pi) 
and negative solids Ni,k (k = 1,...,ni). 
We will refer to the negative solids as cutouts in the following. 
Thus, the CSG operations for a scene S can be described as 

   S = Sum Li          Li = Sum Pi,j  -   Sum Ni,k 
       i=1->l                j=1..pi       k=1..ni     

where + and - denote union and difference.

Basically, a layer can be seen as the difference of two compound (positive and
negative) objects. Because of this, CSG ray tracing a single layer can be
done by tracking when a ray runs within a positive or negative medium. To this
end, we employ two counters (posDepth and negDepth) that are attached as custom
parameters to each ray. Whenever the ray tracer finds the closest
intersection of a ray with the primitives of a layer, a hit program is called
which is illustrated in algorithm 1.


    delta = +1 if entering primitive else -1 

    if positive primitive hit:
        ray.posDepth += delta;
    else
        ray.negDepth += delta; 
    pass

    if (ray.posDepth > 0) && (ray.negDepth <= 0):
        ReportHit()   // final hit in layer found
    else
        ContinueRay(ray); // still inside -ve medium 

Each time a ray enters or leaves a positive or negative primitive, the
counters are increased or decreased, respectively. In case we are inside a
positive medium, but not inside a negative medium, we found the correct
hitpoint corresponding to a layer. Otherwise ray traversal continues using the
updated counters.

To find the final global hitpoint in the scene, a primary ray is sequentially
tested against all layers, and from the set of layer hitpoints the nearest one
is accepted as final position for shading. In the scene graph used by the ray
tracing engines, the layers of a scene can be stored as independent subtrees,
which can be intersected separately.


Questions :

* how to partition general CSG tree into layers ?
* how to split tree into +ve and -ve subtrees ?




CSG Operations of Arbitrary Primitives with Interval Arithmetic and Real-Time Ray Casting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Younis Hijazi1, Aaron Knoll2,4, Mathias Schott3, Andrew Kensler3, Charles Hansen3,4, and Hans Hagen2,4

* https://drops.dagstuhl.de/opus/volltexte/2010/2698/pdf/7.pdf
* ~/opticks_refs/CSG_Arbitrary_Implicit_Ray_Casting.pdf


* :google:`factorize CSG expression into positive and negative tree`


A Flexible Pipeline for the Optimization of CSG Trees
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Markus Friedrich, Christoph Roch, Sebastian Feld, Carsten Hahn
Pierre-Alain Fayolle

* ~/opticks_refs/CSG_Tree_Optimization_2008_03674.pdf
* https://arxiv.org/pdf/2008.03674.pdf




* :google:`AVEVA RVM CAD format`
* https://docs.fileformat.com/3d/rvm/
* https://github.com/cdyk/rvmparser





:google:`layered CSG ray trace`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* DNF : looks to correspond to Opticks positivized trees
* https://en.wikipedia.org/wiki/Disjunctive_normal_form


IceSL
~~~~~~

* http://shapeforge.loria.fr/icesl/icesl-whitepaper.pdf


When modeling with CSG it is common for two surfaces to be perfectly aligned.
This implies that some events in the A–buffer will be given the same depth
values. Which surface comes first in this case is undeter- mined. Worse, due to
numerical instabilities the ordering of overlapping surfaces at an angle may
randomly alternate (an effect often referred to as depth–fighting). Such issues
can have dire consequences, since large portions of space could be wrongly
categorized as inside or outside.  Fortunately, we can detect these cases. When
traversing the list of events in the A–buffer we merge all events which are
closer to each other than a given threshold. This filters out most numerical
instabili- ties. Of course, this also prevents modeling of features thinner
than the threshold. However, depth is encoded on 24 bits over the entire height
of the model, and we use a conservative threshold of 64. Even for a model 1000
mm high the threshold represents only 4μm.

Blister
~~~~~~~~~

* https://www.cc.gatech.edu/~jarek/papers/Blister.pdf
* ~/opticks_refs/jarek_Blister.pdf



Parallel GPU Boolean Evaluation for CSG Ray-Tracing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Marco Domingues
Instituto Superior Técnico, Universidade de Lisboa, Portugal

* https://fenix.tecnico.ulisboa.pt/downloadFile/1126295043835447/article.pdf
* ~/opticks_refs/Parallel_GPU_Boolean_Evaluation_for_GPU_Ray_Tracing.pdf




CSG Regularization pros and cons : keeping track of problem surfaces
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* http://www.cs.otago.ac.nz/homepages/andrew/papers/g4.pdf
* ~/opticks_refs/Wyvill_and_Trotman_Preserving_boundary_CSG_g4.pdf


Ulyanov : GPU-optimized Ray-tracing for Constructive Solid Geometry Scenes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* http://ceur-ws.org/Vol-1576/090.pdf
* https://www.graphicon.ru/html/2016/papers/Pages_490-493.pdf




Overview : CSG to B-REP aka CSG Polygonization aka Meshing 
--------------------------------------------------------------

Have implemented within Opticks:

* MC : marching cubes
* DCS : dual contouring sample, using Octree
* IM : implicit mesher

These three all form the isosurface mesh using only the implicit 
signed distance function for the composite CSG solid. 
Details of the research that went into that in isosurface-.

**None of these approaches cope well with very thin solids, such as the cathode.**


Need to get more information into the algorithm ... 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* parametric descriptions of all primitives are not difficult to come up with, 
  perhaps these can be used to guide the meshing algorithms ? 

* perhaps some kind of hybrid parametric/implicit approach is called for ?

Investigating this in csgparametric-



CSG Regularization : getting rid of dangling face/edges
---------------------------------------------------------

* :google:`how to implement CSG regularization`

Cheng : cs slides 
~~~~~~~~~~~~~~~~~~

* http://www.cs.uky.edu/~cheng/cs535/Notes/PS-Ray-1.pdf
* ~/opticks_refs/csg_regularization_slides_cheng.pdf

Regular set is closure of its interior.

My Q:

* how to implement "closure of interior" ?
* how to use *regular neighborhood method*  (?)


Rossignac : Correct shading of CSG solids using depth interval buffer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ~/opticks_refs/csg_rossignac_regularization_DIBs.pdf
* http://www.cc.gatech.edu/~jarek/papers/DIBs.pdf

cmh CSG concepts
~~~~~~~~~~~~~~~~~~

* ~/opticks_refs/csg_cmh_chap2_concepts_regularization.pdf
* https://www.cs.purdue.edu/homes/cmh/distribution/books/chap2.pdf

Epsilon ball neighbourhood of a point

* full ball : inside
* empty ball : outside
* hemi-ball : interior face point
* wedge-ball : edge between two faces 




CSG Normalization and Pruning, Goldfeather
---------------------------------------------

* http://www.dtic.mil/dtic/tr/fulltext/u2/a201085.pdf
* ~/opticks_refs/CSG_Normalization_and_Pruning_Goldfeather_a201085.pdf
* ~/opticks_refs/Goldfeather_CSG_Normalization_And_Pruning.pdf 


CSG Thesis with set theory intro
----------------------------------

* http://www.en.pms.ifi.lmu.de/publications/diplomarbeiten/Sebastian.Steuer/DA_Sebastian.Steuer.pdf
* ~/opticks_refs/CSG_Thesis_DA_Sebastian.Steuer.pdf

* Union and intersection are commutative:
* Union and intersection are distributive over each other
* The empty set E and the reference set R are identity elements for union and intersection

* A UNION !A = ALL 
* A INTERSECT !A = NULL



Hybrid Implicit/Parametric Polygonization
--------------------------------------------


Cutting and stitching meshes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* http://mesh.brown.edu/taubin/pdfs/Gueziec-etal-tvcg01.pdf
* ~/opticks_refs/cutting_and_stitching_meshes_Gueziec-etal-tvcg01.pdf 

Ref [4] from "Boolean Operations... " Fougerolle..

* http://mrl.nyu.edu/~dzorin/papers/biermann2001abo.pdf
* ~/opticks_refs/Approximate_Boolean_Operations_biermann2001abo.pdf


* http://igg.unistra.fr/People/allegre/publications/SMI04/allegre-smi04.pdf
* ~/opticks_refs/Hybrid_Shape_Representation_allegre-smi04.pdf



CSG Book
-----------

* https://books.google.com.tw/books?id=ntnnCAAAQBAJ&pg=PA380&lpg=PA380&dq=simplify+csg+expression+with+many+differences&source=bl&ots=vlkAUhT_tW&sig=aAUJrZlaCohJKmP64-ThU5wOMlE&hl=en&sa=X&ved=0ahUKEwi_ivbQhdTTAhXLJZQKHX_3DHgQ6AEIKjAF#v=onepage&q=simplify%20csg%20expression%20with%20many%20differences&f=false

Theory and Practice of Geometric Modeling
edited by Wolfgang Strasser, Hans-Peter Seidel

p376

CSG Trees that do not have any difference operator are called positive trees.
Each CSG tree can be rewritten as a positive tree by applying De Morgan's laws
(results in complemented primitives).
Studying active zones and S-bounds is much simpler with positive trees.



CSG Thesis with extensive linked Bibliography
------------------------------------------------

* http://www.nigels.com/research/#Thesis


CSG Tree Rotation
-------------------

My unbalanced trees are mostly mono-operator... most all diffs, some all unions

I would guess that means can straightforwardly restructure the tree without changing
its meaning...


* https://en.wikipedia.org/wiki/Tree_rotation


The AVL Tree Rotations Tutorial
By John Hargrove
Version 1.0.1, Updated Mar-22-2007

* https://www.cise.ufl.edu/~nemo/cop3530/AVL-Tree-Rotations.pdf


* http://gfx.uvic.ca/pubs/2016/blob_traversal/paper.pdf


Blist
~~~~~~~

* http://www.cc.gatech.edu/~jarek/papers/Blist.pdf
* ~/opticks_refs/csg_Blist.pdf

The CSG-to-Blist conversion process takes as input the root-node of the binary
tree, T, and produces the corresponding BL table. Both structures have been
described above. The conversion performs the following steps:

1. Convert T into a positive form by applying deMorgan’s laws and propagating complements to the leaves
2. Rotate the tree by switching the left and right children at each node to make the tree left heavy
3. Visit the leaves from left to right and for each leaf, p, fill in the corresponding fields of BL[p]


Ulyanov
----------------------------------

* http://ceur-ws.org/Vol-1576/090.pdf

CSG Converting To Positive Form
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A CSG tree T is represented in the positive form using only U and I operations
and negation of leaf nodes.  This conversion can be easily done using the
following transformations:

    !( x U y ) = !x I !y
    !( x I y ) = !x U !y
       x - y   =  x I !y

The above transformations are applied to the tree in a pre-order traversal, and
thus all complements are propagated to the leaf nodes. The reverse conversion
to general form can be performed using a post- order traversal (in this case
all negations are first removed from the children of each node).

CSG Minimizing Tree Height
~~~~~~~~~~~~~~~~~~~~~~~~~~~

To reduce the traversal state size we desire a well-balanced CSG tree. Our next
optimization stage is aimed to address this problem by minimizing the height of
CSG tree using local transformations. At this stage, two types of treelets are
considered. For brevity, let us call the child node with a greater height (in
the whole tree T) the heavy child. The first type is formed of treelets which
have the same Boolean operation (U or I) in root node N1 and its heavy child N2
(see Figure 5a). 

::
       
          (N1)     
          U    (N2) 
       T1     U  
            T2 T3*     <-- T3* is heavy child of the node N2
                         
   
Swap treelets T1 and T3::

          (N1)     
          U    (N2)          
       T3     U  
            T2 T1* 

If h(T3) > h(T1) + 1 it is beneficial to transpose these subtrees
As with the rotations for binary search trees these result in elevating subtree T3 and demoting subtree T1. 
Thus, the height of the treelet, rooted at N1, is decreased by one.

We use the multi-pass scheme, where at each pass a CSG tree is traversed in
post-order, and appropriate restructuring patterns are applied.

         



* TODO: implement ldepth, rdepth        

Boolean sum of products
--------------------------

* https://www.dyclassroom.com/boolean-algebra/sum-of-products-and-product-of-sums

There are two forms of canonical expression.

Sum of Products (SOP)
Product of Sums (POS)

* https://en.wikipedia.org/wiki/Canonical_normal_form

Minterms 
      For a boolean function of n variables a product term in which each of the n variables appears once 
      (in either its complemented or uncomplemented form) is called a minterm. 
      Thus, a minterm is a logical expression of n variables that employs only the complement operator 
      and the conjunction operator (AND, INTERSECT)

      Minterms are called products because they are the logical AND of a set of variables, 
      3 of the 8 possible minterms for a boolean function of three variables:
 
      * abc
      * a'bc 
      * ab'c   (a AND b AND NOT-c)

maxterms 
      are called sums because they are the logical OR of a set of variables. 

      For a boolean function of n variables a *sum* term in which each of the n variables appears once 
      (in either its complemented or uncomplemented form) is called a maxterm. 
      Thus, a maxterm is a logical expression of n variables that employs only the complement operator and the 
      disjunction operator (OR, UNION).  

      Maxterms are a dual of the minterm idea (i.e., exhibiting a complementary symmetry in all respects). 
      Instead of using ANDs and complements, we use ORs and complements and proceed similarly.

      For example, the following are two of the eight maxterms of three variables:

      * a + b' + c
      * a' + b + c




These concepts are dual because of their complementary-symmetry relationship as expressed by De Morgan's laws.

The term "Sum of Products" or "SoP" is widely used for the canonical form that
is a disjunction (OR, UNION) of minterms (AND, INTERSECT). 

Its De Morgan dual is a "Product of Sums" or "PoS" for the canonical 
form that is a conjunction (AND, INTERSECT) of maxterms (OR, UNION). 

These forms can be useful for the simplification of these functions, which is of
great importance in the optimization of Boolean formulas in general and digital
circuits in particular.



Balancing CSG trees, to make them less deep
---------------------------------------------

* http://www.cs.jhu.edu/~goodrich/cgc/pubs/csg.ps
* ~/opticks_refs/CSG_Tree_Contraction.pdf
* ~/opticks_refs/CSG_Tree_Contraction_With_Figs.pdf


Optimized BList
~~~~~~~~~~~~~~~~~~~

* http://www.cc.gatech.edu/~jarek/papers/OBF.pdf
* ~/opticks_refs/Optimized_BList_Form_CSG.pdf


How to handle deep unbalanced trees ?
-----------------------------------------

* :google:`balance CSG expressions`

* can CSG expressions be balanced to a more binary tree friendly form ?

* http://www.cs.jhu.edu/~goodrich/cgc/pubs/csg.ps



CSG Building Experience
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* http://designer.mech.yzu.edu.tw/articlesystem/article/compressedfile/(2010-12-10)%20Constructive%20solid%20geometry%20and%20sweep%20representation.aspx?ArchID=1616

A CSG tree is defined as an inverted ordered binary tree whose leaf nodes are 
primitives and interior nodes are regularized set operations. 

The creation of a balanced, unbalanced, or a perfect CSG tree depends solely on the user and how 
he/she decomposes a solid into its primitives. 

The general rule to create balanced trees is to start to build the model from an  
almost central position and branch out in two opposite directions or vice versa. 

Another useful rule is that symmetric objects can lead to perfect trees 
if they are decomposed properly. 
Figure 9 shows a perfect CSG tree and Figure 10 shows an umbalance CSG tree.







Regularization
----------------

Some Regularization Problems in Ray Tracing
John Amanatides Don P. Mitchell

* http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.596.4270&rep=rep1&type=pdf
* ~/opticks_refs/csg_ray_trace_regularization.pdf


Regularization of a set is the closure of its interior.

* http://www.cs.uky.edu/~cheng/cs535/Notes/PS-Ray-1.pdf
* ~/opticks_refs/csg_regularization_PS-Ray-1.pdf

* in/on/out classification ambiguity, remedy: regular neighborhood method


Exact Ray tracing of CSG Models by Preserving Boundary Information
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Geoff Wyvill
Andrew Trotman

* http://www.cs.otago.ac.nz/homepages/andrew/papers/g4.pdf
* ~/opticks_refs/CSG_Preserving_Boundary_Info.pdf

CST : constructuve solid trimming
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CST: Constructive Solid Trimming for rendering BReps and CSG
John Hable and Jarek Rossignac

* http://www.cc.gatech.edu/~jarek/papers/CST.pdf
* ~/opticks_refs/Constructive_Solid_Trimming_CST.pdf




Experience with closed/open geometry
--------------------------------------

Some primitives such as CSG_CYLINDER and CSG_ZSPHERE have 
flags that control endcaps. Disabling caps yields primitives
with open geometry, having a bare boundary.  
Raytracing these allows you to see the "inside" surface of 
the primitive from "outside", which will appear very dark
as normals point outwards from the outer surface.

Such open geometry is however little more than a curiosity.

Open geometry does not have a well defined  "inside" and "outside", which 
means that attempting to use than as CSG sub-objects will 
yield bizarre results, typically with geometry that changes shape 
on moving viewpoint and that you see through to whats behind.

Constructive **SOLID** Geometry only works with solids 
which are by definition closed, with a boolean notion of whether 
a point is inside or outside.

Similarly the Opticks use of indices attached to boundaries 
that identify (outer material, outer surface, inner surface, inner material) 
requires closed geometry.  Without closed geometry optical photon properties
will adopt those of different materials depending on their direction, you 
will see unphysical things like some photons going faster than nearby others.

Regarding geometry modelling and whether it is OK to have open sub-objects 
that are combined in boolean combination : the answer is NO.  
Sub-objects MUST BE CLOSED AND BOUNDED for the CSG implementation to work. 

There is no need to be concerned with "internal" surfaces between sub-objects, 
boolean combination takes care of that. In fact it is best not to think of 
the sub-objects as being distinct objects at all, they are just a convenient way 
to describe combination solids. Only the combination solids have boundary indices
assigned to them.





Refs
------

* http://www.doc.ic.ac.uk/~dfg/graphics/graphics2008/GraphicsLecture10.pdf

* http://web.cse.ohio-state.edu/~parent/classes/681/Lectures/19.RayTracingCSG.pdf

* https://mit-crpg.github.io/OpenMOC/

* https://devtalk.nvidia.com/default/topic/771034/optix/constructive-solid-geometry/

* http://www.cs.utah.edu/~shirley/books/fcg2/rt.pdf

* https://www.clear.rice.edu/comp360/lectures/SurfSpeText.pdf

  Intersections of lines and Special Surfaces: cone, sphere, ...

* https://www.cg.tuwien.ac.at/courses/Rendering/VU.SS2015.html

  CG course with YouTube videos  

CSG Thesis
-----------

* http://www.en.pms.ifi.lmu.de/publications/diplomarbeiten/Sebastian.Steuer/DA_Sebastian.Steuer.pdf


CSG Modelling, postorder on non-perfect tree
-----------------------------------------------

* http://web.iitd.ac.in/~hegde/cad/lecture/L32_solidmcsg.pdf

CSG Implementations
---------------------

* https://github.com/mit-crpg/OpenMOC
* https://github.com/mit-crpg/OpenMOC/blob/6e434c8e235be2b5f010b87c15a32ba7dfd51ea8/docs/source/methods/track_generation.rst#ray-tracing

  CSG mentioned in docs, but dont find the code 

* https://github.com/search?utf8=✓&q=CSG+ray+&type=Repositories&ref=searchresults


* http://c-csg.com


CSG Tree Structure in CUDA ?
------------------------------


Series of posts from Tero Karras
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://devblogs.nvidia.com/parallelforall/thinking-parallel-part-i-collision-detection-gpu/
* https://devblogs.nvidia.com/parallelforall/thinking-parallel-part-ii-tree-traversal-gpu/

Am envisaging very small per-solid CSG trees so no need to 
get into parallel thinking complexity.

Binary tree in array
~~~~~~~~~~~~~~~~~~~~~~~

* https://en.wikipedia.org/wiki/Binary_heap

Let n be the number of elements in the heap and i be an arbitrary valid index
of the array storing the heap. 

If the tree root is at index 0, with valid indices 0 through n − 1, 
then each element a at index i has children 
at indices 2i + 1 and 2i + 2 its parent at index floor((i − 1) ∕ 2).

::

    // i: 0..n-1
    //  2i+1,2i+2, floor((i − 1) ∕ 2)

    0
    1        2
    3   4    5     6
    7 8 9 10 11 12 13 14 
    
    // hmm storing multiple trees in one array requires offsets
    // after the 1st 


Alternatively, if the tree root is at index 1, with valid indices 1 through n, 
then each element a at index i has children 
at indices 2i and 2i +1 its parent at index floor(i ∕ 2).

::

     // i: 1..n
     // 2i, 2i+1, floor(i ∕ 2)

     1                            <-- 1
     2           3                <-- 2
     4    5      6       7        <-- 4
     8 9  10 11  12 13   14 15    <-- 8  basis shapes


* NB no need for left/right pointers, 
  because the tree is complete can navigate just from the index, which is an input


CSG tree into (n,4,4) buffer ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Nodes need a flag distinguishing shape(leaf-node) from operations(op-node with boolean operation and possibly transform)

**shape-node** (dont call this primitive, reserving that word for OptiX primitives) 
 
* basis shape codes sphere/box/... (4 bits probably enough) 
* basis shape parameters 
* bbox 3*2  COULD SKIP as should be calculable from the shape parameters, but
  is convenient and have sufficient space 

**operation-node**

* operation-code union/intersection/difference (2 bits)
* transform applicability none/left/right/both (2 bits)
* 4x4 transform matrix ? 

The flags can easily fit into the spare (always: 0,0,0,1 ) space in 4x4.


test input of the tree in tboolean ? GGeoTest
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* start by adding transform handling 


Extra slicing operation ?
~~~~~~~~~~~~~~~~~~~~~~~~~~

* shape slicing ? ie restrict intersections to a range along an axis
* can be parameterized very compactly : 2bits for axis and 2 floats for range


Concatenating multiple csg trees into one buffer ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* can extend current prim/part model to hold primitves
  that are lists of parts together with primitives that
  are trees of parts combined by csg operations

::
     (n,1,4)  uint4 primBuffer 
     (n,4,4) float4 partBuffer

::

    1184 RT_PROGRAM void bounds (int primIdx, float result[6])
    1185 {
    1186   // could do this offline, but as run once only 
    1187   // its a handy place to dump things checking GPU side state
    1188   //rtPrintf("bounds %d \n", primIdx ); 
    1189 
    1190   const uint4& prim    = primBuffer[primIdx];
    1191   unsigned partOffset  = prim.x ;
    1192   unsigned numParts    = prim.y ;
    1193   unsigned primFlags   = prim.w ;



When/how to use the transforms ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Remember the user of the CSG tree is the intersect_csg 
code which looks up shape type, parameters, and transforms from it.
Presumably intersect with inverse transformed ray, as thats 
much simpler than transforming the shape.

* care with normal transforms needed if allow 
  transforms like sphere to ellipsoid

Hmm there could be multiple levels of transform up the tree.
So for a node need to look up thru the ancestors with (i-1)/2 
to collect transform matrices to multiply.

::

    int p = (i-1)/2 ; // parent of current node
    while p >= 0:
        // collect transforms
        p = (p-1)/2 



::

     03 static __device__
     04 void intersect_sphere(const quad& q0, const float& tt_min, float3& tt_normal, float& tt  )
     05 {
     ..
     11     float3 center = make_float3(q0.f);
     12     float radius = q0.f.w;
     13 
     14     float3 O = ray.origin - center;
     15     float3 D = ray.direction;
     16 
     17     float b = dot(O, D);
     18     float c = dot(O, O)-radius*radius;
     19     float disc = b*b-c;
     20 
     21     float sdisc = disc > 0.f ? sqrtf(disc) : 0.f ;
     22     float root1 = -b - sdisc ;
     23     float root2 = -b + sdisc ;
     24 
     25     bool valid_intersect = sdisc > 0.f ;   // ray has a segment within the sphere
     26 
     27     if(valid_intersect)
     28     {
     29         tt =  root1 > tt_min ? root1 : root2 ;
     30         tt_normal = tt > tt_min ? (O + tt*D)/radius : tt_normal ;
     31     }
     32 
     33 }


* http://computergraphics.stackexchange.com/questions/212/why-are-inverse-transformations-applied-to-rays-rather-than-forward-transformati



Compact 4x4
~~~~~~~~~~~~~~~~

* https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation

The representation of a rotation as a quaternion (4 numbers) 
is more compact than the representation as an orthogonal matrix (9 numbers). 

* rotation quat : 4 numbers
* translation   : 3 numbers 

Hmm probably doing both rotation and translation within a CSG solid
is rare, could split the operations to slim down nodes.

:google:`csg tree with transforms`


OpenSCAD
----------

* https://github.com/openscad/openscad/wiki/Project%3A-Survey-of-CSG-algorithms


OpenCSG : image (Z-buffer) based CSG rendering with OpenGL
--------------------------------------------------------------

* http://opencsg.org

csgjs-cpp
--------------

CSG library for C++, port of https://github.com/evanw/csg.js/

* https://github.com/dabroz/csgjs-cpp/blob/master/csgjs.cpp


pycsg : Python port of Evan Wallace's csg.js (MIT)
-----------------------------------------------------

* https://github.com/timknip/pycsg
* https://github.com/timknip/pycsg/blob/master/csg/core.py
* https://github.com/timknip/pycsg/blob/master/csg/geom.py



csg.js (MIT)
--------------

implements CSG operations on meshes elegantly and concisely using BSP trees,
and is meant to serve as an easily understandable implementation of the
algorithm.


* https://github.com/evanw/csg.js/
* http://evanw.github.io/csg.js/
* http://madebyevan.com  lots of WebGL

docs
~~~~~

* http://evanw.github.io/csg.js/docs/

All CSG operations are implemented in terms of two functions, clipTo() and
invert(), which remove parts of a BSP tree inside another BSP tree and swap
solid and empty space, respectively. To find the union of a and b, we want to
remove everything in a inside b and everything in b inside a, then combine
polygons from a and b into one solid::

    a.clipTo(b);
    b.clipTo(a);
    a.build(b.allPolygons());


The only tricky part is handling overlapping coplanar polygons in both trees.
The code above keeps both copies, but we need to keep them in one tree and
remove them in the other tree. To remove them from b we can clip the inverse of
b against a. The code for union now looks like this:

::

    a.clipTo(b);
    b.clipTo(a);
    b.invert();
    b.clipTo(a);
    b.invert();
    a.build(b.allPolygons());

Subtraction and intersection naturally follow from set operations. 
If union is A | B, subtraction is A - B = ~(~A | B) 
and intersection is A & B = ~(~A | ~B) where ~ is the complement operator.


observations
~~~~~~~~~~~~~~~

* really concise imp
* whacky triangles : probably does not matter when only using for viz
* cool webgl interface.
* simple api

::

    var a = CSG.cube();
    var b = CSG.sphere({ radius: 1.2 });
    a.setColor(1, 1, 0);
    b.setColor(0, 0.5, 1);
    return a.subtract(b);




OpenCASCADE
-------------

* https://dev.opencascade.org/index.php?q=search/node/GPU
* https://dev.opencascade.org/index.php?q=node/1173


Spatially Efficient Tree for GPU ray tracing of CSG 
-------------------------------------------------------

* :google:`Spatially Efficient Tree Layout for GPU Ray-tracing of Constructive Solid Geometry Scenes`

* https://scholar.google.com.tw/citations?view_op=view_citation&hl=en&user=23mu44wAAAAJ&citation_for_view=23mu44wAAAAJ:HE397vMXCloC


Vadim Turlapov
~~~~~~~~~~~~~~~

* https://scholar.google.com.tw/citations?user=23mu44wAAAAJ&hl=en



Geant4 CSG
-----------

::

   g4-cls G4UnionSolid
   g4-cls G4BooleanSolid 
   g4-cls G4VSolid


CSG logic picking which distance to which constituent done in eg G4UnionSolid::

    097     G4double DistanceToIn( const G4ThreeVector& p,
     98                            const G4ThreeVector& v  ) const ;
     99 
    100     G4double DistanceToIn( const G4ThreeVector& p ) const ;
    101 
    102     G4double DistanceToOut( const G4ThreeVector& p,
    103                             const G4ThreeVector& v,
    104                             const G4bool calcNorm=false,
    105                                   G4bool *validNorm=0,
    106                                   G4ThreeVector *n=0 ) const ;
    107 
    108     G4double DistanceToOut( const G4ThreeVector& p ) const ;


Pure virtuals in base G4VSolid::

    119     virtual EInside Inside(const G4ThreeVector& p) const = 0;
    120       // Returns kOutside if the point at offset p is outside the shapes
    121       // boundaries plus Tolerance/2, kSurface if the point is <= Tolerance/2
    122       // from a surface, otherwise kInside.
    123 
    124     virtual G4ThreeVector SurfaceNormal(const G4ThreeVector& p) const = 0;
    125       // Returns the outwards pointing unit normal of the shape for the
    126       // surface closest to the point at offset p.
    127 
    128     virtual G4double DistanceToIn(const G4ThreeVector& p,
    129                                   const G4ThreeVector& v) const = 0;
    130       // Return the distance along the normalised vector v to the shape,
    131       // from the point at offset p. If there is no intersection, return
    132       // kInfinity. The first intersection resulting from `leaving' a
    133       // surface/volume is discarded. Hence, it is tolerant of points on
    134       // the surface of the shape.
    135 
    136     virtual G4double DistanceToIn(const G4ThreeVector& p) const = 0;
    137       // Calculate the distance to the nearest surface of a shape from an
    138       // outside point. The distance can be an underestimate.
    139 
    140     virtual G4double DistanceToOut(const G4ThreeVector& p,
    141                    const G4ThreeVector& v,
    142                    const G4bool calcNorm=false,
    143                    G4bool *validNorm=0,
    144                    G4ThreeVector *n=0) const = 0;
    145       // Return the distance along the normalised vector v to the shape,
    146       // from a point at an offset p inside or on the surface of the shape.
    147       // Intersections with surfaces, when the point is < Tolerance/2 from a
    148       // surface must be ignored.
    149       // If calcNorm==true:
    150       //    validNorm set true if the solid lies entirely behind or on the
    151       //              exiting surface.
    152       //    n set to exiting outwards normal vector (undefined Magnitude).
    153       //    validNorm set to false if the solid does not lie entirely behind
    154       //              or on the exiting surface
    155       // If calcNorm==false:
    156       //    validNorm and n are unused.
    157       //
    158       // Must be called as solid.DistanceToOut(p,v) or by specifying all
    159       // the parameters.
    160 
    161     virtual G4double DistanceToOut(const G4ThreeVector& p) const = 0;
    162       // Calculate the distance to the nearest surface of a shape from an
    163       // inside point. The distance can be an underestimate.
    164 



github CSG
------------

* https://github.com/jtramm/ConstructiveSolidGeometry.jl
* https://github.com/jtramm/ConstructiveSolidGeometry.jl/blob/master/examples/1-Introduction.ipynb


Embree CSG : Computer Science Thesis describing Embree CSG
-------------------------------------------------------------

* https://dspace.cvut.cz/bitstream/handle/10467/65282/F3-DP-2016-Karaffova-Marketa-Efektivni_sledovani_paprsku_v_CSG_modelech.pdf?sequence=-1
* ~/opticks_refs/F3-DP-2016-Karaffova-Marketa-Efektivni_sledovani_paprsku_v_CSG_modelech.pdf


:google:`GPU CSG boolean Roth`
---------------------------------

Spatially Efficient Tree Layout for GPU Ray-tracing of Constructive Solid Geometry Scenes

* http://ceur-ws.org/Vol-1576/090.pdf


Andrew Kensler
----------------

Ray Tracing CSG Objects Using Single Hit Intersections

* http://xrt.wdfiles.com/local--files/doc%3Acsg/CSG.pdf

* http://www.cs.utah.edu/~aek/
* http://www.cs.utah.edu/~aek/research/


* http://www.cs.utah.edu/~aek/research/csgimplicits.pdf
* ~/opticks_refs/aek_csgimplicits.pdf

* http://www.cs.utah.edu/~aek/research/tree.pdf
* ~/opticks_refs/aek_tree_rotation.pdf



XRT Renderer : simpler? way of handling CSG trees
---------------------------------------------------

* http://xrt.wikidot.com/search:site/q/csg 


Kensler state tables corrected in below page...

* http://xrt.wikidot.com/doc:csg

* http://xrt.wikidot.com/downloads

XRT appears to be provided only as a Windows binary 


Question : can basis shapes be concave ? what about torii ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 

Kensler statement:

   sub-objects must be closed, non-self-intersecting 
   and have consistently oriented normals

Intuitive guess is that concave is OK and torii too but such shapes
will need to allow more loops over additional intersections, 
and the intersect algorithms would need to return all roots 
eventually as are repeatedly called with larger tmin.


XRT corrected Kensler algorithm pseudo-code
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* http://xrt.wikidot.com/doc:csg

Implementing the below in an intersect_boolean_solid program
similar to those in oxrap/cu/hemi-pmt.cu seems most appropriate.

The only sticky part is advancing tmin and re-intersecting, this
probably means have to defer the rt*Intersection 
calls to a higher level, meaning will need to pass normals
identity and t values around. 

* hmm dont want to duplicate intersect code, one version having 
  the rt*Intersection calls and the other not though ?
   
* perhaps templating trickery can do this

* hmm probably a higher level macro that conditionally 
  uses the rtPotentialIntersection rtReportIntersection
  functions based on a preprocessor switch can allow
  the same base shape intersection code to be 
  used as part of a boolean and as simple shape

TODO: refactor hemi-pmt.cu into imp headers for each shape 


::

   // 3 action tables for Union/Intersection/Subtraction 

    minA = minB = min // current nearest intersection

    //
    // rtIntersectionDistance 
    //    parametric distance from the current ray’s origin to the closest intersection point yet discovered.
    //    available to *intersection*, closest_hit, any_hit programs  
    //

    /// persumably for *intersection* this starts at the t value for the 
    /// intersection of the ray with the bounding box ?
    /// Which is why this solids primitive intersection code is being called.
    ///


    ( tA, NA ) = IntersectWithA( O, D, minA )
    ( tB, NB ) = IntersectWithB( O, D, minB )

    stateA = ClassifyEnterExitOrMiss( tA, NA )
    stateB = ClassifyEnterExitOrMiss( tB, NB )


    loop:
         action = boolean_action_table [stateA, stateB] 
         if 
                   ReturnMiss ∈ action
         then
                   return miss

         else if 
                  ReturnA ∈ action
             or ( ReturnAIfCloser ∈ action and tA <= tB ) 
             or ( ReturnAIfFarther ∈ action and tA > tB ) 
         then
             return tA, NA

         else if 
                  ReturnB ∈ action
             or ( ReturnBIfCloser ∈ action and tB <= tA )
             or ( ReturnBIfFarther ∈ action and tB > tA )
         then
             if FlipB ∈ action then NB = -NB
             return tB, NB

         else if 
                  AdvanceAAndLoop ∈ action
             or ( AdvanceAAndLoopIfCloser ∈ action and tA <= tB ) 
         then
             minA = tA
             ( tA, NA ) = IntersectWithA( O, D, minA ) 
             stateA = ClassifyEnterExitOrMiss( tA, NA )

         else if 
                 AdvanceBAndLoop ∈ action
            or ( AdvanceBAndLoopIfCloser ∈ action and tB <= tA ) 
         then
             minB = tB
             ( tB, NB ) = IntersectWithB( O, D, minB ) 
             stateB = ClassifyEnterExitOrMiss( tB, NB )
         end if

    end loop
￼




How to map the above described algorithm to OptiX ?
-----------------------------------------------------


rtTrace ?
~~~~~~~~~~

rtTrace can only be called from generate, closest_hit or miss progs
and its too high level anyhow (it needs to take the geometry node instance
as argument : usually top) ... so it is not appropriate for IntersectWithA 


Selector
~~~~~~~~~~

A selector is similar to a group in that it is a collection of higher level
graph nodes. The number of nodes in the collection is set by
rtSelectorSetChildCount, and the individual children are assigned with
rtSelectorSetChild. Valid child types are rtGroup, rtGeometryGroup,
rtTransform, and rtSelector.  The main difference between selectors and groups
is that selectors do not have an acceleration structure associated with them.
Instead, a visit program is specified with rtSelectorSetVisitProgram. This
program is executed every time a ray encounters the selector node during graph
traversal. The program specifies which children the ray should continue
traversal through by calling rtIntersectChild.  A typical use case for a
selector is dynamic (i.e. per-ray) level of detail: an object in the scene may
be represented by a number of geometry nodes, each containing a different level
of detail version of the object. The geometry groups containing these different
representations can be assigned as children of a selector. The visit program
can select which child to intersect using any criterion (e.g. based on the
footprint or length of the current ray), and ignore the others.  As for groups
and other graph nodes, child nodes of a selector can be shared with other graph
nodes to allow flexible instancing.


Intersection
~~~~~~~~~~~~~~~~

Ray traversal invokes an intersection program when the current ray encounters
one of a Geometry object’s primitives. It is the responsibility of an
intersection program to compute whether the ray intersects with the primitive,
and to report the parametric t-value of the intersection. Additionally, the
intersection program is responsible for computing and reporting any details of
the intersection, such as surface normal vectors, through attribute variables.
Once the intersection program has determined the t-value of a ray-primitive
intersection, it must report the result by calling a pair of OptiX functions,
rtPotentialIntersection and rtReportIntersection.

::

    ￼__device__ bool rtPotentialIntersection( float tmin )
    ￼__device__ bool rtReportIntersection( unsigned int material )


rtPotentialIntersection 
    takes the intersection’s t-value as an argument. 
    If the t-value could potentially be the closest intersection of the current traversal 
    the function narrows the t-interval of the current ray accordingly and returns true. 
    If the t-value lies outside the t-interval the function returns false, 
    whereupon the intersection program may trivially return.

    If rtPotentialIntersection returns true, 
    the intersection program may then set any attribute variable values 
    and call rtReportIntersection. This function takes an unsigned int specifying 
    the index of a material that must be associated with an any hit and closest hit program. 
    This material index can be used to support primitives of several different 
    materials flattened into a single Geometry object. 
    Traversal then immediately invokes the corresponding any hit program. 
    Should that any hit program invalidate the intersection via the rtIgnoreIntersection function, 
    then rtReportIntersection will return false. Otherwise, it will return true.



current partitioned intersect
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This assumes the primitive is chopped into single basis shape subparts.

Loops over all the sub-parts of the primitive invoking 
rtPotentialIntersection/rtReportIntersection multiple times... 
leaving the task narrowing down to find the closest intersect tmin to OptiX

::

    1243 RT_PROGRAM void intersect(int primIdx)
    1244 {
    1245   const uint4& solid    = solidBuffer[primIdx];
    1246   unsigned int numParts = solid.y ;
    1247 
    1248   //const uint4& identity = identityBuffer[primIdx] ; 
    1249   //const uint4 identity = identityBuffer[instance_index*primitive_count+primIdx] ;  // just primIdx for non-instanced
    1250 
    1251   // try with just one identity per-instance 
    1252   uint4 identity = identityBuffer[instance_index] ;
    1253 
    1254 
    1255   for(unsigned int p=0 ; p < numParts ; p++)
    1256   {
    1257       unsigned int partIdx = solid.x + p ;
    1258 
    1259       quad q0, q1, q2, q3 ;
    1260 
    1261       q0.f = partBuffer[4*partIdx+0];
    1262       q1.f = partBuffer[4*partIdx+1];
    1263       q2.f = partBuffer[4*partIdx+2] ;
    1264       q3.f = partBuffer[4*partIdx+3];
    1265 
    1266       identity.z = q1.u.z ;  // boundary from partBuffer (see ggeo-/GPmt)
    1267 
    1268       int partType = q2.i.w ;
    1269 
    1270       // TODO: use enum
    1271       switch(partType)
    1272       {
    1273           case 0:
    1274                 intersect_aabb(q2, q3, identity);
    1275                 break ;
    1276           case 1:
    1277                 intersect_zsphere<false>(q0,q1,q2,q3,identity);
    1278                 break ;
    1279           case 2:
    1280                 intersect_ztubs(q0,q1,q2,q3,identity);
    1281                 break ;



How would a boolean_intersect look ? 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CSG OptiX 
--------------------------------------------------------


Implemented in oxrap/cu::

   hemi-pmt.cu   # <-- NB poorly named 
   boolean-solid.h
   intersect_part.h
   intersect_boolean.h


Constructive solid geometry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://devtalk.nvidia.com/default/topic/771034/?comment=4296423

sphere/box intersection, nljones:

Your ray payload needs to contain a bit that indicates whether the ray is in a
sphere. Set it to one upon entering the sphere and zero upon leaving. 

## sign(N.D) determines if entering/leaving the shape

Your closest hit program for the sphere sends a new ray in the same direction with
this bit set. Your closest hit program for the box sends a new ray in the same
direction if the bit is one and sets the color of the ray payload if the bit is
zero.

In order to render the interface between the box and shere where they touch,
you also need to keep a bit indicating whether the ray is inside the box.


Example code for CSG in OptiX
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://devtalk.nvidia.com/default/topic/967816/?comment=4985663

dlacewell:

.. haven't thought about this too much, but for a limited number of closed
shapes, you could use per ray data (PRD) to store a hit counter, or really just
a flag, for each shape. 

## bitfield in per-ray-data with 1 or 2 bits for each basis shape could handle
## boolean operations involving small numbers of shapes (as is usual in G4 geometries)  

Use the closest hit program to either terminate the ray
and shade, or toggle the hit flag in PRD for the current shape and continue the
ray using rtTrace. 
Terminate when all hit flags are toggled on at once, meaning
that the current point is inside all the shapes.

##  rtTerminateRay only available in AnyHit, so by this dlacewell presumably means 
##  that can either accept a closest hit when per-ray-data flags are as they should
##  be for the boolean expression being evaluated OR if not (when this is not a real surface)
##  can call rtTrace again (from a modified starting position ? or tmin ) 

For some shapes, you could use the geometric normals to determine whether the
ray is entering or exiting, and then you might not need hit flags.

It may also be possible to do this with an any-hit program for a very small
number of shapes, by storing all intersections for the ray and sorting/shading
them in the ray gen program. That would be slow if there were too many shapes.


You could take optixSpherePP in the SDK and make some changes:

* add another sphere to the scene, that uses the same material
* add a geometry id variable to the sphere, and expose it as an attribute for the closest hit program
* change the closest hit program to make one of the spheres completely 
  transparent based on id, and continue the ray with rtTrace. 
* Make the spheres semi-transparent. You still shoot a new ray with rtTrace, 
  but composite the result with the current sphere color and opacity using per ray data.

Once you get that all working, then it's probably not a big jump to CSG.


IceSL
------

* https://members.loria.fr/Sylvain.Lefebvre/icesl/


CSG Simplification
-----------------------------------------------

* http://webserver2.tecgraf.puc-rio.br/~lhf/ftp/doc/sib2006b.pdf

  Spatial Partitioning to simplify CSG rendering
  Hardware-assisted Rendering of CSG Models


* http://www.cc.gatech.edu/~turk/my_papers/pxpl_csg.pdf

  CSG Tree Normalization and Pruning


CSG Ray Tracing Techniques
-----------------------------

* http://www.sciencedirect.com/science/article/pii/S0734189X86800548

  A new algorithm for object oriented ray tracing  (1986) Saul Youssef 


Ray Tracing CSG Models : implementation details
------------------------------------------------------ 

* http://web.cse.ohio-state.edu/~parent/classes/681/Lectures/19.RayTracingCSG.pdf


Cool WebGL interface allowing to edit CSG geometries
----------------------------------------------------

* http://evanw.github.io/csg.js/
* http://evanw.github.io/csg.js/docs/
* https://github.com/evanw/csg.js/


* http://learningthreejs.com/blog/2011/12/10/constructive-solid-geometry-with-csg-js/

  three.js bridge to csg.js


CSG to BREP mesh ?
-------------------

* :google:`BREP of CSG boolean solids`

* http://stackoverflow.com/questions/2002976/constructive-solid-geometry-mesh


Boole (public domain)
~~~~~~~~~~~~~~~~~~~~~~~

* http://people.mpi-inf.mpg.de/~schoemer/ECG/SS02/papers/boole2.pdf
* http://gamma.cs.unc.edu/CSG/boole.html


Solid and Physical Modelling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* http://www.cc.gatech.edu/~jarek/papers/SPM.pdf

Converting CSG models into Meshed B-Rep Models Using Euler Operators and Propagation Based Marching Cubes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* http://www.scielo.br/pdf/jbsmse/v29n4/a01v29n4.pdf
* ~/opticks_refs/csg_to_brep_marching_cubes_a01v29n4.pdf 


Merging BSP Trees Yields Polyhedral Set Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* http://www.mcs.csueastbay.edu/~tebo/papers/siggraph90.pdf

BSP : binary space partioning


Exact and Robust (Self-)Intersections for Polygonal Meshes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* http://www.graphics.rwth-aachen.de/media/papers/campen_2010_eg_021.pdf

Fast, Exact, Linear Booleans :  Gilbert Bernstein and Don Fussell
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* http://stackoverflow.com/questions/2002976/constructive-solid-geometry-mesh

* ~/opticks_refs/booleans2009.pdf
* http://www.gilbertbernstein.com/resources/booleans2009.pdf

* http://www.gilbertbernstein.com/project_boolean.html

* https://github.com/gilbo/cork   


B-rep algorithms: 

#. If A and B are the boundaries of two objects whose union, difference or
   intersection we would like to compute, find the intersection of A and B, thus
   dividing each surface into two components, one inside and one outside the other
   surface. 

#. Select the appropriate component of each surface, and 

#. stitch these together to form the correct output. 

This apparent simplicity belies the large number of special cases 
that result from the various ways the two objects can align

BSP trees afford an alternative to B-rep algorithms that avoid their
concomitant case explosion by explicitly handling all degenerate
configurations of geometry. 

One author of Fast, Exact, Linear Booleans has a project named "cork" on github
that implements mesh-based CSG: github.com/gilbo/cork. His site
gilbertbernstein.com/project_boolean.html indicates that this is not the same
method as that of the paper. 


http://gts.sourceforge.net (LGPL)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




EOU
}
csg-dir(){ echo $(local-base)/env/graphics/csg/graphics/csg-csg ; }
csg-cd(){  cd $(csg-dir); }
csg-mate(){ mate $(csg-dir) ; }
csg-get(){
   local dir=$(dirname $(csg-dir)) &&  mkdir -p $dir && cd $dir

}
