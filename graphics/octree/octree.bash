# === func-gen- : graphics/octree/octree fgp graphics/octree/octree.bash fgn octree fgh graphics/octree
octree-src(){      echo graphics/octree/octree.bash ; }
octree-source(){   echo ${BASH_SOURCE:-$(env-home)/$(octree-src)} ; }
octree-vi(){       vi $(octree-source) ; }
octree-env(){      elocal- ; }
octree-usage(){ cat << EOU

Octree
=========

Vital statistics
------------------

* http://www.txutxi.com/?p=378
* sum i=0..h  8^i = ( 8^(h+1) - 1) / 7


* max leaf at height h, 8^h  = 2^(3*h) = 1 << (3*h)

::

    octleaf0_ = lambda h:math.pow(8,h)
    octleaf1_ = lambda h:math.pow(2,3*h)
    octleaf_ = lambda h:1 << (3*h)

    octside0_ = lambda h:math.pow(2,h)
    octside_ = lambda h:1 << h 


From Ericson p309

* complete binary tree of n levels has 2^n − 1 nodes
* complete d-ary tree of n levels has (d^n − 1)/(d − 1) nodes
*  n = 0 -> 0, so perhaps   (d^(n+1) - 1)/(d - 1)    n=0 -> 1   

::

    octnum1_ = lambda n:(math.pow(8,n+1) - 1)/7 
    octnum2_ = lambda n:(math.pow(2,3*(n+1)) - 1)/7    ## funny that factors into 7 
    octnum_ = lambda n:((1 << (3*(n+1))) - 1)/7 

    h_ = lambda h:h 


    import locale
    locale.setlocale(locale.LC_ALL, 'en_US')
    locale.format("%d", 1000000, grouping=True)

    dfmt_ = lambda d:locale.format("%20d", d, grouping=True)

    oct_ = lambda h:" ".join(map(lambda fn:dfmt_(fn(h)), [h_, octside_, octleaf_, octnum_]))

            ##         h              octside_            octleaf_               octnum_
    In [101]: print "\n".join(map(oct_, range(16)))
                       0                    1                    1                    1
                       1                    2                    8                    9
                       2                    4                   64                   73
                       3                    8                  512                  585
                       4                   16                4,096                4,681
                       5                   32               32,768               37,449
                       6                   64              262,144              299,593
                       7                  128            2,097,152            2,396,745
                       8                  256           16,777,216           19,173,961
                       9                  512          134,217,728          153,391,689
                      10                1,024        1,073,741,824        1,227,133,513
                      ------------------------------------------------------------------
                      11                2,048        8,589,934,592        9,817,068,105
                      12                4,096       68,719,476,736       78,536,544,841
                      13                8,192      549,755,813,888      628,292,358,729
                      14               16,384    4,398,046,511,104    5,026,338,869,833
                      15               32,768   35,184,372,088,832   40,210,710,958,665

    In [103]: dfmt_(1 << 32)
    Out[103]: '       4,294,967,296'



* octree implicit indices: 8*i + 1, ... 8*i + 8 like binary trees,


Gigavoxels
-------------

GigaVoxels: A Voxel-Based Rendering Pipeline For Efficient Exploration Of Large And Detailed Scenes

* http://maverick.inria.fr/Membres/Cyril.Crassin/thesis/


Efficient Sparse Voxel Octree, Laine and Karras
-------------------------------------------------

* https://research.nvidia.com/publication/efficient-sparse-voxel-octrees
* ~/opticks_refs/Efficient_Sparse_Voxel_Octrees_laine2010tr1_paper.pdf


GPU Octree
-----------

GPU-based Adaptive Octree Construction Algorithms
* https://www.cse.iitb.ac.in/~rhushabh/publications/octree
* ~/opticks_refs/GPU_octree.pdf



Out-Of-Core Construction of Sparse Voxel Octrees - reference implementation 
-----------------------------------------------------------------------------

* streaming approach, bizarre stuff about empty nodes i didnt follow
* https://github.com/Forceflow/ooc_svo_builder/blob/master/src/svo_builder/OctreeBuilder.cpp

* http://graphics.cs.kuleuven.be/publications/BLD14OCCSVO/
* http://graphics.cs.kuleuven.be/publications/BLD14OCCSVO/BLD14OCCSVO_paper.pdf
* ~/opticks_refs/Sparse_Octree_Morton_Updated_BLD14OCCSVO_paper.pdf



EOU
}
octree-dir(){ echo $(local-base)/env/graphics/octree/graphics/octree-octree ; }
octree-cd(){  cd $(octree-dir); }
octree-mate(){ mate $(octree-dir) ; }
octree-get(){
   local dir=$(dirname $(octree-dir)) &&  mkdir -p $dir && cd $dir

}
