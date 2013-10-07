# === func-gen- : graphics/collada/meshtool fgp graphics/collada/meshtool.bash fgn meshtool fgh graphics/collada
meshtool-src(){      echo graphics/collada/meshtool.bash ; }
meshtool-source(){   echo ${BASH_SOURCE:-$(env-home)/$(meshtool-src)} ; }
meshtool-vi(){       vi $(meshtool-source) ; }
meshtool-env(){      elocal- ; }
meshtool-usage(){ cat << EOU

MESHTOOL
========

By the PyCollada author. Some visualisation options based on Panda3D. 
Also uses pycollada to do conversions.


EOU
}
meshtool-dir(){ echo $(local-base)/env/graphics/collada/meshtool ; }
meshtool-cd(){  cd $(meshtool-dir); }
meshtool-mate(){ mate $(meshtool-dir) ; }
meshtool-get(){
   local dir=$(dirname $(meshtool-dir)) &&  mkdir -p $dir && cd $dir

   git clone https://github.com/pycollada/meshtool

}
