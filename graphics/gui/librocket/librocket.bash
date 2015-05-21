# === func-gen- : graphics/gui/librocket/librocket fgp graphics/gui/librocket/librocket.bash fgn librocket fgh graphics/gui/librocket
librocket-src(){      echo graphics/gui/librocket/librocket.bash ; }
librocket-source(){   echo ${BASH_SOURCE:-$(env-home)/$(librocket-src)} ; }
librocket-vi(){       vi $(librocket-source) ; }
librocket-env(){      elocal- ; }
librocket-usage(){ cat << EOU


libRocket is the C++ user interface package based on the HTML and CSS
standards. It is designed as a complete solution for any project's interface
needs.

MIT license


EOU
}
librocket-dir(){ echo $(local-base)/env/graphics/gui/libRocket ; }
librocket-cd(){  cd $(librocket-dir); }
librocket-mate(){ mate $(librocket-dir) ; }
librocket-get(){
   local dir=$(dirname $(librocket-dir)) &&  mkdir -p $dir && cd $dir

   git clone https://github.com/libRocket/libRocket

}
