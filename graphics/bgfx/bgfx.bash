# === func-gen- : graphics/bgfx/bgfx fgp graphics/bgfx/bgfx.bash fgn bgfx fgh graphics/bgfx src base/func.bash
bgfx-source(){   echo ${BASH_SOURCE} ; }
bgfx-edir(){ echo $(dirname $(bgfx-source)) ; }
bgfx-ecd(){  cd $(bgfx-edir); }
bgfx-dir(){  echo $LOCAL_BASE/env/graphics/bgfx/bgfx ; }
bgfx-cd(){   cd $(bgfx-dir); }
bgfx-vi(){   vi $(bgfx-source) ; }
bgfx-env(){  elocal- ; }
bgfx-usage(){ cat << EOU

bgfx
=======

Cross-platform, graphics API agnostic, "Bring Your Own Engine/Framework" style
rendering library.

* https://github.com/bkaradzic/bgfx
* https://bkaradzic.github.io/bgfx/overview.html

  Lots of backends 

* https://github.com/jpcy/bgfx-minimal-example

Uses shader language subset ?

See Also
---------

* llgl-
* dileng-



EOU
}
bgfx-get(){
   local dir=$(dirname $(bgfx-dir)) &&  mkdir -p $dir && cd $dir
   [ ! -d bgfx ] && git clone https://github.com/bkaradzic/bgfx 

}
