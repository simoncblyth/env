# === func-gen- : graphics/optix7/plotoptix fgp graphics/optix7/plotoptix.bash fgn plotoptix fgh graphics/optix7 src base/func.bash
plotoptix-source(){   echo ${BASH_SOURCE} ; }
plotoptix-edir(){ echo $(dirname $(plotoptix-source)) ; }
plotoptix-ecd(){  cd $(plotoptix-edir); }
plotoptix-dir(){  echo $LOCAL_BASE/env/graphics/optix7/plotoptix ; }
plotoptix-cd(){   cd $(plotoptix-dir); }
plotoptix-vi(){   vi $(plotoptix-source) ; }
plotoptix-env(){  elocal- ; }
plotoptix-usage(){ cat << EOU


Data visualisation in Python based on NVIDIA OptiX 7.1 ray tracing framework.
===============================================================================

PlotOptiX is basically an interface to RnD.SharpOptiX library which we are
developing and using in our Studio. RnD.SharpOptiX offers much more
functionality than it is now available through PlotOptiX. 

* https://github.com/rnd-team-dev/plotoptix
* https://rnd.team/project/plotoptix
* https://www.instagram.com/rnd.team.studio/
* https://plotoptix.rnd.team

Feature list includes:

* configurable multi-GPU support

Argh, it is CSharp based windows-centric : need mono on Linux.




EOU
}
plotoptix-get(){
   local dir=$(dirname $(plotoptix-dir)) &&  mkdir -p $dir && cd $dir

}
