# === func-gen- : graphics/mayavi/mayavi fgp graphics/mayavi/mayavi.bash fgn mayavi fgh graphics/mayavi src base/func.bash
mayavi-source(){   echo ${BASH_SOURCE} ; }
mayavi-edir(){ echo $(dirname $(mayavi-source)) ; }
mayavi-ecd(){  cd $(mayavi-edir); }
mayavi-dir(){  echo $LOCAL_BASE/env/graphics/mayavi/mayavi ; }
mayavi-cd(){   cd $(mayavi-dir); }
mayavi-vi(){   vi $(mayavi-source) ; }
mayavi-env(){  elocal- ; }
mayavi-usage(){ cat << EOU


Mayavi: 3D scientific data visualization and plotting in Python
==================================================================

* https://docs.enthought.com/mayavi/mayavi/index.html

* https://docs.enthought.com/mayavi/mayavi/installation.html

* https://github.com/enthought/mayavi/issues/532



macOS black screen
--------------------

* https://github.com/enthought/mayavi/issues/881


mayavi.mlab : aka "mlab"
--------------------------

* https://docs.enthought.com/mayavi/mayavi/mlab.html#simple-scripting-with-mlab

When using IPython with mlab::
  
   ipython --gui=qt


* https://docs.enthought.com/mayavi/mayavi/mlab.html#moving-the-camera



EOU
}
mayavi-get(){
   local dir=$(dirname $(mayavi-dir)) &&  mkdir -p $dir && cd $dir

}
