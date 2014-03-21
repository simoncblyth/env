# === func-gen- : graphics/collada/fcollada/fcollada fgp graphics/collada/fcollada/fcollada.bash fgn fcollada fgh graphics/collada/fcollada
fcollada-src(){      echo graphics/collada/fcollada/fcollada.bash ; }
fcollada-source(){   echo ${BASH_SOURCE:-$(env-home)/$(fcollada-src)} ; }
fcollada-vi(){       vi $(fcollada-source) ; }
fcollada-env(){      elocal- ; }
fcollada-usage(){ cat << EOU

FCOLLADA
==========

* https://collada.org/mediawiki/index.php/FCollada

From  http://www.tutorgigpedia.com/ed/COLLADA

FCollada (C++) - A utility library available from Feeling Software. In contrast
to the COLLADA DOM, Feeling Software's FCollada provides a higher-level
interface. FCollada is used in ColladaMaya, ColladaMax, and several commercial
game engines. The development of the open source part was discontinued by
Feeling Software in 2008. The company continues to support its paying customers
and licenses with improved versions of its software.



EOU
}
fcollada-dir(){ echo $(local-base)/env/graphics/collada/fcollada/graphics/collada/fcollada-fcollada ; }
fcollada-cd(){  cd $(fcollada-dir); }
fcollada-mate(){ mate $(fcollada-dir) ; }
fcollada-get(){
   local dir=$(dirname $(fcollada-dir)) &&  mkdir -p $dir && cd $dir

}
