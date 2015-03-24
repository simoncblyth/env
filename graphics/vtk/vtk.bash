# === func-gen- : graphics/vtk/vtk fgp graphics/vtk/vtk.bash fgn vtk fgh graphics/vtk
vtk-src(){      echo graphics/vtk/vtk.bash ; }
vtk-source(){   echo ${BASH_SOURCE:-$(env-home)/$(vtk-src)} ; }
vtk-vi(){       vi $(vtk-source) ; }
vtk-env(){      elocal- ; }
vtk-usage(){ cat << EOU

VTK Visualization Toolkit
==========================

Aimed at data visualization rather than more generic 3D graphics.

* http://www.vtk.org/

* http://www.kitware.com/media/html/KitwareOnAppleOSXItJustWorksInMacPorts.html





macports
--------

::

    port info vtk   


EOU
}
vtk-dir(){ echo $(local-base)/env/graphics/vtk/$(vtk-name); }
vtk-cd(){  cd $(vtk-dir); }
vtk-mate(){ mate $(vtk-dir) ; }
vtk-vers(){ echo 6.0.0 ; }
vtk-name(){ echo VTK$(vtk-vers) ; }
vtk-url(){  echo http://www.vtk.org/files/release/6.0/vtk-$(vtk-vers).tar.gz ; }
vtk-get(){
   local dir=$(dirname $(vtk-dir)) &&  mkdir -p $dir && cd $dir

   local url=$(vtk-url)
   local tgz=$(basename $url)
   [ ! -f "$tgz" ] && curl -L -O $url 

   local nam=$(vtk-name)
   [ ! -d "$nam" ] && tar zxvf $tgz


}
