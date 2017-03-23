# === func-gen- : graphics/isosurface/dualcontouringsample/dcs fgp graphics/isosurface/dualcontouringsample/dcs.bash fgn dcs fgh graphics/isosurface/dualcontouringsample
dcs-src(){      echo graphics/isosurface/dualcontouringsample/dcs.bash ; }
dcs-source(){   echo ${BASH_SOURCE:-$(env-home)/$(dcs-src)} ; }
dcs-vi(){       vi $(dcs-source) ; }
dcs-env(){      elocal- ; }
dcs-usage(){ cat << EOU



http://ngildea.blogspot.co.uk/2014/11/implementing-dual-contouring.html

https://github.com/nickgildea/DualContouringSample



edgevmap
   relates edge index 0..11 to corner index 0..7
   foreach of the 12 cube edges, 4 for each of the 3 axes 


                 3:011    E3    7:111
                   +------------+
                   |            |  
                   |            |
                E5 |     +Z     | E7
                   |            | 
                   |            |
                   +------------+   
                 1:001   E1    5:101


         2:010   E2   6:110
         Y +------------+
           |            |  
           |            |
        E4 |            | E6
           |            | 
           |            |
           +------------+ X
         0:000  E0     4:100


   Last 4 edges difficult to draw


   0: 000     
   1: 001 
   2: 010 
   3: 011 
   4: 100  
   5: 101
   6: 110
   7: 111






EOU
}

dcs-dir(){ echo $(local-base)/env/graphics/isosurface/dualcontouringsample/DualContouringSample ; }
dcs-cd(){  cd $(dcs-dir); }
dcs-get(){
   local dir=$(dirname $(dcs-dir)) &&  mkdir -p $dir && cd $dir

   local url=https://github.com/nickgildea/DualContouringSample
   [ ! -d $(basename $url) ] && git clone $url

}
