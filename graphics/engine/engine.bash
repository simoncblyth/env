# === func-gen- : graphics/engine/engine fgp graphics/engine/engine.bash fgn engine fgh graphics/engine
engine-src(){      echo graphics/engine/engine.bash ; }
engine-source(){   echo ${BASH_SOURCE:-$(env-home)/$(engine-src)} ; }
engine-vi(){       vi $(engine-source) ; }
engine-env(){      elocal- ; }
engine-usage(){ cat << EOU

Engines
=========

* http://blog.digitaltutors.com/unity-udk-cryengine-game-engine-choose/


* Unity
* Unreal Engine 4 
* Source 2


Valve : Source 2
-------------------

* http://blog.digitaltutors.com/valve-enters-game-engine-race-source-2/





EOU
}
engine-dir(){ echo $(local-base)/env/graphics/engine/graphics/engine-engine ; }
engine-cd(){  cd $(engine-dir); }
engine-mate(){ mate $(engine-dir) ; }
engine-get(){
   local dir=$(dirname $(engine-dir)) &&  mkdir -p $dir && cd $dir

}
