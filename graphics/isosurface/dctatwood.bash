# === func-gen- : graphics/isosurface/dctatwood fgp graphics/isosurface/dctatwood.bash fgn dctatwood fgh graphics/isosurface
dctatwood-src(){      echo graphics/isosurface/dctatwood.bash ; }
dctatwood-source(){   echo ${BASH_SOURCE:-$(env-home)/$(dctatwood-src)} ; }
dctatwood-vi(){       vi $(dctatwood-source) ; }
dctatwood-env(){      elocal- ; }
dctatwood-usage(){ cat << EOU


* http://www.tatwood.net/articles/7/dual_contour
* http://www.tatwood.net/wp-content/uploads/dualcontour.zip

EOU
}
dctatwood-dir(){ echo $(local-base)/env/graphics/isosurface/dctatwood/dualcontour ; }
dctatwood-cd(){  cd $(dctatwood-dir); }
dctatwood-mate(){ mate $(dctatwood-dir) ; }
dctatwood-get(){
   local dir=$(dirname $(dctatwood-dir)) &&  mkdir -p $dir && cd $dir

    
   [ ! -f dualcontour.zip ] && curl -L -O http://www.tatwood.net/wp-content/uploads/dualcontour.zip

   [ ! -d dualcontour ] && unzip dualcontour.zip

}
