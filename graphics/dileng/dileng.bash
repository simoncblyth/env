# === func-gen- : graphics/dileng/dileng fgp graphics/dileng/dileng.bash fgn dileng fgh graphics/dileng src base/func.bash
dileng-source(){   echo ${BASH_SOURCE} ; }
dileng-edir(){ echo $(dirname $(dileng-source)) ; }
dileng-ecd(){  cd $(dileng-edir); }
dileng-dir(){  echo $LOCAL_BASE/env/graphics/dileng/DiligentEngine ; }
dileng-cd(){   cd $(dileng-dir); }
dileng-vi(){   vi $(dileng-source) ; }
dileng-env(){  elocal- ; }
dileng-usage(){ cat << EOU

Diligent Engine
==================

* https://github.com/DiligentGraphics/DiligentEngine
* http://diligentgraphics.com/diligent-engine/
* http://diligentgraphics.com


* https://github.com/Immediate-Mode-UI/Nuklear

See Also
---------

* llgl-
* bgfx-



EOU
}
dileng-get(){
   local dir=$(dirname $(dileng-dir)) &&  mkdir -p $dir && cd $dir

   [ ! -d DiligentEngine ] && git clone https://github.com/DiligentGraphics/DiligentEngine

}
